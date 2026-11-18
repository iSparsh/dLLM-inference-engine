// docs/INTERFACES.md: Return absolute block-causal grouped-query attention
// (§3.3).
#include "checks.h"

namespace {
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
  TORCH_CHECK(false,
              "not implemented yet: block_attention; see docs/INTERFACES.md");
}
} // namespace
TORCH_LIBRARY_IMPL(dllm, CUDA, m) {
  m.impl("block_attention", block_attention);
}
