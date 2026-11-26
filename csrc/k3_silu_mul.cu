#include "checks.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {
__global__ void silu_mul_kernel(const __nv_bfloat16* in, __nv_bfloat16* out,
                                int64_t in_stride, int64_t out_stride, int64_t n) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  int64_t row = idx / 8960;
  int64_t col = idx % 8960;
  float g = __bfloat162float(in[row * in_stride + col]);
  float u = __bfloat162float(in[row * in_stride + col + 8960]);
  float s = g / (1.0f + expf(-g));
  __nv_bfloat16 rounded_s = __float2bfloat16_rn(s);
  out[row * out_stride + col] = __float2bfloat16_rn(__bfloat162float(rounded_s) * u);
}
at::Tensor silu_mul(at::Tensor gate_up) {
  check(gate_up, gate_up.device(), at::kBFloat16, {-1, 17920}, "gate_up");
  c10::cuda::CUDAGuard guard(gate_up.device());
  auto out = at::empty({gate_up.size(0), 8960}, gate_up.options());
  int64_t n = gate_up.size(0) * 8960;
  if (n) {
    silu_mul_kernel<<<(n + 255) / 256, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(gate_up.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      gate_up.stride(0), out.stride(0), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}
}
TORCH_LIBRARY_IMPL(dllm, CUDA, m) { m.impl("silu_mul", silu_mul); }
