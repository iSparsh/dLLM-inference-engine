// docs/INTERFACES.md: Apply bf16 SiLU to the first half and multiply by the
// second (§3.4).
#include "checks.h"

namespace {
at::Tensor silu_mul(at::Tensor gate_up) {
  check(gate_up, gate_up.device(), at::kBFloat16, {-1, 17920}, "gate_up");
  TORCH_CHECK(false, "not implemented yet: silu_mul; see docs/INTERFACES.md");
}
} // namespace
TORCH_LIBRARY_IMPL(dllm, CUDA, m) { m.impl("silu_mul", silu_mul); }
