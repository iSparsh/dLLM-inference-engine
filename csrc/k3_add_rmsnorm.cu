// docs/INTERFACES.md: Optionally add x into residual; return weighted RMSNorm
// (§3.1).
#include "checks.h"

namespace {
at::Tensor add_rmsnorm(std::optional<at::Tensor> x, at::Tensor residual,
                       at::Tensor weight, double eps) {
  check(residual, residual.device(), at::kBFloat16, {-1, 1536}, "residual");
  check(weight, residual.device(), at::kBFloat16, {1536}, "weight");
  if (x)
    check(*x, residual.device(), at::kBFloat16, {residual.size(0), 1536}, "x");
  TORCH_CHECK(false,
              "not implemented yet: add_rmsnorm; see docs/INTERFACES.md");
}
} // namespace
TORCH_LIBRARY_IMPL(dllm, CUDA, m) { m.impl("add_rmsnorm", add_rmsnorm); }
