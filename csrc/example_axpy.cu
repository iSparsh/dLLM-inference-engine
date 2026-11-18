// Teaching example: allocate an output, guard the device, launch on PyTorch's
// stream.
#include "checks.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

namespace {
__global__ void axpy(const float *x, const float *y, float *out, float a,
                     int64_t n) {
  auto i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = fmaf(a, x[i], y[i]);
}
at::Tensor example_axpy(at::Tensor x, at::Tensor y, double alpha) {
  check(x, x.device(), at::kFloat, {-1}, "x");
  check(y, x.device(), at::kFloat, {x.numel()}, "y");
  c10::cuda::CUDAGuard guard(x.device());
  auto out = at::empty_like(x);
  if (x.numel()) {
    axpy<<<(x.numel() + 255) / 256, 256, 0,
           c10::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), out.data_ptr<float>(), alpha,
        x.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}
} // namespace
TORCH_LIBRARY_IMPL(dllm, CUDA, m) { m.impl("example_axpy", example_axpy); }
