#include <torch/library.h>

TORCH_LIBRARY(dllm, m) {
  m.def("add_rmsnorm(Tensor? x, Tensor(a!) residual, Tensor weight, float eps) "
        "-> Tensor");
  m.def("rope_kv_write(Tensor(a!) q, Tensor k, Tensor v, Tensor(b!) k_cache, "
        "Tensor(c!) v_cache, Tensor q_start, Tensor cos, Tensor sin) -> ()");
  m.def("block_attention(Tensor q, Tensor k_cache, Tensor v_cache, Tensor "
        "q_start, int block_size) -> Tensor");
  m.def("silu_mul(Tensor gate_up) -> Tensor");
  m.def("select_commit(Tensor logits, Tensor(a!) tokens, Tensor window, float "
        "tau, int mask_id) -> Tensor");
  m.def("example_axpy(Tensor x, Tensor y, float alpha) -> Tensor");
}
