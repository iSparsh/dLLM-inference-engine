// M1 correctness-first implementation. ATen's CUDA softmax/max preserve the
// exact bf16 reference behavior; a custom CUDA kernel performs GPU-only commit.
// M2 can replace the full softmax with a windowed parallel reduction.
#include "checks.h"
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <tuple>

namespace {
__global__ void commit_kernel(const __nv_bfloat16* confidence,
                              const int64_t* guess, int64_t* tokens,
                              const int32_t* window, int32_t* stats,
                              __nv_bfloat16 tau, int64_t mask_id) {
  // Single-thread decision is intentionally simple for M1; all data stay on GPU.
  if (blockIdx.x || threadIdx.x) return;
  int start = window[0], end = window[1];
  float best = -INFINITY;
  int forced = -1;
  for (int i = 0; i < 32; ++i) {
    if (i >= start && i < end && tokens[i] == mask_id) {
      int row = i ? i - 1 : 0;
      float p = __bfloat162float(confidence[row]);
      if (p > best) { best = p; forced = i; }
    }
  }
  int count = 0;
  for (int i = 0; i < 32; ++i) {
    if (i >= start && i < end && tokens[i] == mask_id) {
      int row = i ? i - 1 : 0;
      if (__bfloat162float(confidence[row]) > __bfloat162float(tau) || i == forced) {
        tokens[i] = guess[row];
        ++count;
      }
    }
  }
  bool window_done = true, block_done = true;
  for (int i = 0; i < 32; ++i) {
    if (tokens[i] == mask_id) {
      block_done = false;
      if (i >= start && i < end) window_done = false;
    }
  }
  stats[0] = count;
  stats[1] = window_done;
  stats[2] = block_done;
}
at::Tensor select_commit(at::Tensor logits, at::Tensor tokens,
                         at::Tensor window, double tau, int64_t mask_id) {
  check(logits, logits.device(), at::kBFloat16, {32, 151936}, "logits");
  check(tokens, logits.device(), at::kLong, {32}, "tokens");
  check(window, logits.device(), at::kInt, {2}, "window");
  c10::cuda::CUDAGuard guard(logits.device());
  // Softmax in the logits dtype: ATen computes fp32 internally and rounds
  // every probability to bf16 before its max/tie-breaking reduction.
  auto probs = at::softmax(logits, -1, c10::nullopt);
  auto result = probs.max(-1);
  auto confidence = std::get<0>(result).contiguous();
  auto guess = std::get<1>(result).contiguous();
  auto stats = at::empty({3}, window.options());
  commit_kernel<<<1, 1, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(confidence.data_ptr<at::BFloat16>()),
      guess.data_ptr<int64_t>(), tokens.data_ptr<int64_t>(),
      window.data_ptr<int32_t>(), stats.data_ptr<int32_t>(),
      __float2bfloat16_rn((float)tau), mask_id);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return stats;
}
}
TORCH_LIBRARY_IMPL(dllm, CUDA, m) { m.impl("select_commit", select_commit); }
