// docs/INTERFACES.md: Return absolute block-causal grouped-query attention
// (§3.3).
//
// v1: simple and obviously correct. One thread block per (query row, query
// head), 128 threads. Keys are processed in tiles of 128: thread t computes the
// full q·k dot product for key j0+t, the tile updates an fp32 online softmax
// (running max m, running sum l), then thread t accumulates output dimension t
// over the tile's values. All math is fp32; the output is rounded to bf16 once.
#include "checks.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

namespace {
constexpr int kHeads = 12;
constexpr int kGroup = 6; // query heads per KV head
constexpr int kDim = 128;
constexpr int kThreads = 128; // one thread per output dim, one key per tile slot
constexpr int kWarps = kThreads / 32;

template <bool IsMax> __device__ float block_reduce(float x, float *scratch) {
  for (int o = 16; o > 0; o >>= 1) {
    float y = __shfl_xor_sync(0xffffffff, x, o);
    x = IsMax ? fmaxf(x, y) : x + y;
  }
  if (threadIdx.x % 32 == 0)
    scratch[threadIdx.x / 32] = x;
  __syncthreads();
  x = scratch[0];
  for (int w = 1; w < kWarps; ++w)
    x = IsMax ? fmaxf(x, scratch[w]) : x + scratch[w];
  __syncthreads(); // scratch is reused by the next reduction
  return x;
}

__global__ void __launch_bounds__(kThreads)
    block_attention_v1(const __nv_bfloat16 *__restrict__ q, int64_t q_s0,
                       int64_t q_s1, const __nv_bfloat16 *__restrict__ k,
                       int64_t k_s0, int64_t k_s1,
                       const __nv_bfloat16 *__restrict__ v, int64_t v_s0,
                       int64_t v_s1, const int *__restrict__ q_start, int m_rows,
                       int block_size, float scale,
                       __nv_bfloat16 *__restrict__ out) {
  const int h = blockIdx.x, row = blockIdx.y, t = threadIdx.x;
  const int start = *q_start; // device-resident: graph-safe
  const int p = start + row;
  // Visible keys form one prefix [0, end): the current block and all earlier
  // ones, never past the last query row (later slots hold stale data).
  const int end = min(start + m_rows, (p / block_size + 1) * block_size);

  const __nv_bfloat16 *kh = k + (h / kGroup) * k_s0;
  const __nv_bfloat16 *vh = v + (h / kGroup) * v_s0;

  __shared__ float qs[kDim];
  __shared__ float probs[kThreads];
  __shared__ float scratch[kWarps];
  qs[t] = __bfloat162float(q[row * q_s0 + h * q_s1 + t]) * scale;
  __syncthreads();

  float m = -INFINITY, l = 0.f, acc = 0.f;
  for (int j0 = 0; j0 < end; j0 += kThreads) {
    const int j = j0 + t;
    float s = -INFINITY;
    if (j < end) {
      const __nv_bfloat16 *kr = kh + j * k_s1;
      float dot = 0.f;
#pragma unroll 8
      for (int d = 0; d < kDim; ++d)
        dot = fmaf(qs[d], __bfloat162float(kr[d]), dot);
      s = dot;
    }
    // Every tile holds at least one visible key (j0 < end), so m_new is finite.
    const float m_new = fmaxf(m, block_reduce<true>(s, scratch));
    const float e = j < end ? expf(s - m_new) : 0.f;
    probs[t] = e;
    const float corr = expf(m - m_new); // exp(-inf) = 0 on the first tile
    l = l * corr + block_reduce<false>(e, scratch); // also syncs probs
    acc *= corr;
    const int n = min(kThreads, end - j0);
    for (int jj = 0; jj < n; ++jj)
      acc = fmaf(probs[jj], __bfloat162float(vh[(j0 + jj) * v_s1 + t]), acc);
    m = m_new;
    __syncthreads(); // probs is overwritten by the next tile
  }
  out[(row * kHeads + h) * kDim + t] = __float2bfloat16(acc / l);
}

at::Tensor block_attention(at::Tensor q, at::Tensor k_cache, at::Tensor v_cache,
                           at::Tensor q_start, int64_t block_size) {
  check(q, q.device(), at::kBFloat16, {-1, 12, 128}, "q");
  check(k_cache, q.device(), at::kBFloat16, {2, -1, 128}, "k_cache");
  check(v_cache, q.device(), at::kBFloat16, {2, k_cache.size(1), 128},
        "v_cache");
  check(q_start, q.device(), at::kInt, {1}, "q_start");
  TORCH_CHECK(block_size > 0 &&
                  (q.size(0) == 1 || q.size(0) == 32 || q.size(0) == 64),
              "unsupported block_size or M");
  c10::cuda::CUDAGuard guard(q.device());
  const auto m_rows = q.size(0);
  auto out = at::empty({m_rows, kHeads, kDim}, q.options());
  auto bf = [](const at::Tensor &x) {
    return reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr<at::BFloat16>());
  };
  block_attention_v1<<<dim3(kHeads, m_rows), kThreads, 0,
                       c10::cuda::getCurrentCUDAStream()>>>(
      bf(q), q.stride(0), q.stride(1), bf(k_cache), k_cache.stride(0),
      k_cache.stride(1), bf(v_cache), v_cache.stride(0), v_cache.stride(1),
      q_start.data_ptr<int>(), static_cast<int>(m_rows),
      static_cast<int>(block_size), 1.f / sqrtf(static_cast<float>(kDim)),
      reinterpret_cast<__nv_bfloat16 *>(out.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
} // namespace
TORCH_LIBRARY_IMPL(dllm, CUDA, m) {
  m.impl("block_attention", block_attention);
}
