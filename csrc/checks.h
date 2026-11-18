#pragma once
#include <ATen/ATen.h>
#include <torch/library.h>

// A negative extent denotes a caller-selected size, e.g. M or Lmax.
inline void check(const at::Tensor &t, at::Device device, at::ScalarType dtype,
                  std::initializer_list<int64_t> shape, const char *name) {
  TORCH_CHECK(t.is_cuda() && t.device() == device, name,
              " must be on the same CUDA device");
  TORCH_CHECK(t.scalar_type() == dtype &&
                  t.dim() == static_cast<int64_t>(shape.size()),
              name, " has incorrect dtype or rank");
  int i = 0;
  for (auto extent : shape) {
    TORCH_CHECK(extent < 0 || t.size(i) == extent, name,
                " has incorrect extent in dim ", i);
    ++i;
  }
  TORCH_CHECK(t.stride(-1) == 1, name, " requires contiguous last dimension");
}
