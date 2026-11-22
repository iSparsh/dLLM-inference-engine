#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/selection
echo '=== Extract golden fixtures ==='
bash tools/fetch_golden.sh
echo '=== Compile CUDA and run the selection/K3 correctness tests ==='
uv run pytest tests/test_ops.py -k 'silu_mul or add_rmsnorm or select_commit' -v -rs
# -rs exposes unexpected skips; also fail if an M1 op is skipped.
echo '=== K3 CUDA Graph microbenchmarks ==='
for op in silu_mul add_rmsnorm; do
  for backend in ours torch; do
    uv run python bench/bench_op.py --op "$op" --backend "$backend" --graph --out-dir results/selection
  done
done
echo '=== Live K1 parity (requires locally cached dLLM model) ==='
uv run pytest tests/test_select_live.py -v -rs
