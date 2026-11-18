// docs/INTERFACES.md: Apply §3.5 shifted bf16 selection; update tokens and
// return three device counters.
#include "checks.h"

namespace {
at::Tensor select_commit(at::Tensor logits, at::Tensor tokens,
                         at::Tensor window, double tau, int64_t mask_id) {
  check(logits, logits.device(), at::kBFloat16, {32, 151936}, "logits");
  check(tokens, logits.device(), at::kLong, {32}, "tokens");
  check(window, logits.device(), at::kInt, {2}, "window");
  TORCH_CHECK(false,
              "not implemented yet: select_commit; see docs/INTERFACES.md");
}
} // namespace
TORCH_LIBRARY_IMPL(dllm, CUDA, m) { m.impl("select_commit", select_commit); }
