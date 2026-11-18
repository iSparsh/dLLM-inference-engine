// docs/INTERFACES.md: Rotate Q in place and write rotated K and original V
// (§3.2).
#include "checks.h"

namespace {
void rope_kv_write(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor k_cache,
                   at::Tensor v_cache, at::Tensor q_start, at::Tensor cos,
                   at::Tensor sin) {
  check(q, q.device(), at::kBFloat16, {-1, 12, 128}, "q");
  check(k, q.device(), at::kBFloat16, {q.size(0), 2, 128}, "k");
  check(v, q.device(), at::kBFloat16, {q.size(0), 2, 128}, "v");
  check(k_cache, q.device(), at::kBFloat16, {2, -1, 128}, "k_cache");
  check(v_cache, q.device(), at::kBFloat16, {2, k_cache.size(1), 128},
        "v_cache");
  check(q_start, q.device(), at::kInt, {1}, "q_start");
  check(cos, q.device(), at::kBFloat16, {k_cache.size(1), 128}, "cos");
  check(sin, q.device(), at::kBFloat16, {k_cache.size(1), 128}, "sin");
  TORCH_CHECK(false,
              "not implemented yet: rope_kv_write; see docs/INTERFACES.md");
}
} // namespace
TORCH_LIBRARY_IMPL(dllm, CUDA, m) { m.impl("rope_kv_write", rope_kv_write); }
