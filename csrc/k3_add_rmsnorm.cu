#include "checks.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {
constexpr int H = 1536;
__global__ void rms_kernel(const __nv_bfloat16* x, __nv_bfloat16* residual,
                           const __nv_bfloat16* weight, __nv_bfloat16* out,
                           int64_t xs, int64_t rs, int64_t os, float eps) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  __shared__ float sums[256];
  float sq = 0.0f;
  // The updated residual must round to bf16 BEFORE the reduction.
  for (int col = tid; col < H; col += blockDim.x) {
    int64_t ridx = (int64_t)row * rs + col;
    float r = __bfloat162float(residual[ridx]);
    if (x) {
      r += __bfloat162float(x[(int64_t)row * xs + col]);
      residual[ridx] = __float2bfloat16_rn(r);
      r = __bfloat162float(residual[ridx]);
    }
    sq += r * r;
  }
  sums[tid] = sq;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) sums[tid] += sums[tid + offset];
    __syncthreads();
  }
  float inv = rsqrtf(sums[0] / (float)H + eps);
  for (int col = tid; col < H; col += blockDim.x) {
    float r = __bfloat162float(residual[(int64_t)row * rs + col]);
    __nv_bfloat16 norm = __float2bfloat16_rn(r * inv);
    float result = __bfloat162float(norm) * __bfloat162float(weight[col]);
    out[(int64_t)row * os + col] = __float2bfloat16_rn(result);
  }
}
at::Tensor add_rmsnorm(std::optional<at::Tensor> x, at::Tensor residual,
                       at::Tensor weight, double eps) {
  check(residual, residual.device(), at::kBFloat16, {-1, H}, "residual");
  check(weight, residual.device(), at::kBFloat16, {H}, "weight");
  if (x) check(*x, residual.device(), at::kBFloat16, {residual.size(0), H}, "x");
  c10::cuda::CUDAGuard guard(residual.device());
  auto out = at::empty({residual.size(0), H}, residual.options());
  if (residual.size(0)) {
    rms_kernel<<<residual.size(0), 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      x ? reinterpret_cast<const __nv_bfloat16*>(x->data_ptr<at::BFloat16>()) : nullptr,
      reinterpret_cast<__nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      x ? x->stride(0) : 0, residual.stride(0), out.stride(0), (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}
}
TORCH_LIBRARY_IMPL(dllm, CUDA, m) { m.impl("add_rmsnorm", add_rmsnorm); }
