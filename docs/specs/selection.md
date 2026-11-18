# Slice: Selection and Fused Ops

**ECE 759 Final Project · Work slice spec**
Milestones: M0 Thu 11/19 · M1 Sun 11/22 · M2 Sun 11/29 · M3 Wed 12/2 · slides 12/3 · report section 12/6

**What you hand in** (on branch `slice/selection`, merged to `main` by pull request):
- `csrc/k3_silu_mul.cu`, `csrc/k3_add_rmsnorm.cu`, `csrc/k1_select_commit.cu`, passing `tests/test_ops.py` for all three.
- Benchmark results in `results/selection/` (`bench/bench_op.py --out-dir results/selection`).
- The cross-GPU baseline script `bench/run_baselines.sh` and its results for every GPU the team has, in `results/baselines/`.
- An Nsight Compute summary for K1 (`docs/report/selection-profile.md`).
- Your report section draft in `docs/report/selection.md`, and 2–3 slides.

**Hardware:** any NVIDIA GPU with compute capability 8.6 or 8.9, or Euler, for the kernels. Problem 5 (baselines) needs runs on every NVIDIA GPU the team has; you write the script, and each GPU's owner runs it.

---

## Background

You need very little knowledge of language models for this slice. The [GUIDE](../GUIDE.md) sections 2–6 give the wider picture; here is what matters for your three kernels.

**The model's output at each position** is a row of 151,936 scores (logits), one per vocabulary token. A diffusion pass produces 32 such rows, one per slot of the current 32-token block. Most slots are still blank (`MASK`); the decoder must decide, for the blanks in the current group of 8 slots, which guesses to keep.

**K1, the selection step, is the diffusion model's decision rule.** The reference code computes the LM head for all 32 rows, then a bf16 softmax over all 151,936 entries of the active window's rows, a gather, a mask, a comparison and a boolean scatter, with several CPU-GPU synchronizations along the way. (Our PyTorch reference op softmaxes all 32 rows; same result, more work.) On its own this is cheap, about 0.04 ms of GPU time per pass. Its real cost is the synchronizations: they stop the engine from recording the whole pass as one CUDA Graph. You will do it in a few GPU kernels with no synchronization. The exact rule is in [INTERFACES.md §3.5](../INTERFACES.md), and it has two traps that make it a good exercise in getting parallel numerics exactly right:

1. **The winner is decided after rounding.** The reference rounds each probability to bf16 and takes the *first* index with the largest rounded probability. Two logits that differ slightly can round to the same probability, and then the lower index wins even if its logit is smaller. So "argmax of the logits" is not quite right.
2. **The threshold is bf16 too.** τ = 0.9 is rounded to bf16 (0.8984375) before the strict comparison.

Your kernel must commit *exactly* the same tokens as the reference on every recorded step, not approximately.

**Only 8 rows matter.** The active window is 8 slots, and slot `i` reads logits row `max(i-1, 0)`. So K1 needs at most 8 of the 32 rows (7 distinct ones for the first window, where slots 0 and 1 both read row 0): about 2.4 MB of the 9.7 MB of logits. Reading only those is part of the design.

**K3, the fused small ops.** Between the big matrix multiplies, each layer runs several small elementwise and normalization steps. In PyTorch each is a separate kernel launch with its own trip through memory. You will fuse two of them:
- `add_rmsnorm`: residual add, then RMSNorm (a sum-of-squares reduction over 1536 values per row, an rsqrt, a scale), with HF's exact casting order.
- `silu_mul`: `silu(gate) * up`, purely elementwise.

Per pass there are 57 norm calls (two per layer plus the final norm) and 28 SiLU × mul calls. Together with the other small ops they take about 1.4 ms of a 17.6 ms pass, and most of the ~1,400 kernel launches.

**Your three ops** (full contracts in [INTERFACES.md §3.1, 3.4, 3.5](../INTERFACES.md)):

```
add_rmsnorm(x[M,1536] or None, residual[M,1536] (in place), weight[1536], eps) -> normed[M,1536]
silu_mul(gate_up[M,17920]) -> out[M,8960]
select_commit(logits[32,151936], tokens[32] (in place), window: int32[2] on device, tau, mask_id)
    -> stats: int32[3] on device  (n_committed, window_done, block_done)
```

All three must be **graph-safe**: no `.item()`, no host synchronization, launch on the current CUDA stream. The engine records each pass as a CUDA Graph and replays it.

---

## Problem 1 (M0). Set up and read

a) Set up your environment ([ENVIRONMENT.md](../setup/ENVIRONMENT.md)) and check it:
```
uv run python tools/smoke_test.py
uv run pytest -q
```

b) Read `dllm/ops/reference.py` for `add_rmsnorm`, `silu_mul` and `select_commit`, and `dllm/reference_decode.py` to see where `select_commit` sits in the decode loop. Read `csrc/ops.cpp` (the op schemas), `csrc/example_axpy.cu` (a complete working kernel showing the launch pattern), then your three stub files.

c) Extract the golden tensors with `tools/fetch_golden.sh` (they are committed as an archive; or regenerate them with `uv run python tools/dump_golden.py`). For `select_commit` they hold three real logits cases. Traces (`tools/dump_traces.py`) record decisions, not logits, so for step-by-step checks you'll use a live comparison (Problem 3).

d) Create your branch `slice/selection` and push it.

## Problem 2 (M1). SiLU × mul and add + RMSNorm

a) Implement `csrc/k3_silu_mul.cu`. Use vectorized loads (8 bf16 values per 128-bit load) where the address and row stride allow it (16-byte aligned), with a scalar fallback otherwise; a contiguous last dimension alone doesn't guarantee alignment. Reproduce the reference rounding: `silu(gate)` rounded to bf16, then multiplied by `up` in bf16.

b) Implement `csrc/k3_add_rmsnorm.cu`. One thread block per row is a natural start: each thread adds its share of `x` into `residual` (bf16), accumulates the sum of squares in fp32, a block-wide reduction (warp shuffles, then shared memory) produces the mean, and each thread scales its values. Follow HF's order exactly: fp32 normalize, cast to bf16, then multiply by the bf16 weight. Support `x = None` (plain RMSNorm).

c) Pass the tests and measure both backends for `M ∈ {1, 32, 64}`:
```
uv run pytest tests/test_ops.py -k "silu_mul or add_rmsnorm" -v
uv run python bench/bench_op.py --op silu_mul --backend ours --graph --out-dir results/selection
uv run python bench/bench_op.py --op silu_mul --backend torch --graph --out-dir results/selection
uv run python bench/bench_op.py --op add_rmsnorm --backend ours --graph --out-dir results/selection
uv run python bench/bench_op.py --op add_rmsnorm --backend torch --graph --out-dir results/selection
```
The test summary must say *passed*; *skipped* means your op wasn't run. Report time and achieved bandwidth for both backends against the GPU's peak bandwidth (2 × memory clock × bus width / 8: 256 GB/s on the 4060 Laptop, ~280 GB/s on Euler's RTX 4000 SFF Ada, 936 GB/s on the 3090).

## Problem 3 (M1). Parallel selection, correct first

a) Write a first, simple, exactly correct `csrc/k1_select_commit.cu`. Suggested structure, one kernel per stage:
   1. For each needed row (at most 8): find the maximum logit and the fp32 sum of `exp(l - max)`. One thread block per row, with a block-wide reduction.
   2. For each eligible slot: compute the rounded maximum probability and find the **first** index whose rounded probability equals it (a second pass over the row, with a min-reduction over matching indices).
   3. One small block: apply the bf16 threshold, force the most confident eligible slot (leftmost on ties), write tokens, compute `stats`.

b) Pass the tests. They include hand-made tie cases, an empty window, and the three real logits cases from the golden files:
```
uv run pytest tests/test_ops.py -k select_commit -v
```
c) Then check every step of real generations. `dllm.reference_decode.decode` takes an `observe` hook that is called with each pass's logits and tokens just before selection. Write a test (`tests/test_select_live.py`) that, inside the hook, runs both the reference op and yours on copies of `tokens` and asserts identical tokens and stats, over 5 GSM8K-100 prompts at τ ∈ {0.7, 0.9, 1.0}. Any mismatch is a bug: the committed set must be identical.

## Problem 4 (M2). Parallel selection, fast

a) **Spread each row across the GPU.** One thread block per row uses only 8 blocks. Split each row into chunks handled by many blocks, each producing a partial (max, sum) that a second step combines (the same online-softmax merge used in attention). Choose the chunk size to fill the SMs.

b) **Read the logits once if you can.** Stage 2 re-reads the row. Find a way to avoid most of it, for example by having stage 1 keep each chunk's candidates near the maximum, and show whether it helps. Any shortcut must provably keep the earliest index among equal rounded probabilities, or fall back to re-reading the row; add a test for the boundary case.

c) **Fewer launches.** With CUDA Graphs, launch cost is small, but every extra kernel still costs a few microseconds. Try fusing stages (for example, a last-block-done pattern for the final selection) and measure.

d) Report time vs the PyTorch reference path (the `torch` backend) and achieved bandwidth on the bytes you actually read. Profile the final version with Nsight Compute:
```
mkdir -p results/selection
ncu --set full -k regex:<your kernel's name> -o results/selection/k1 \
    .venv/bin/python bench/bench_op.py --op select_commit --backend ours --iters 3
```
(`--iters` below 20 runs the op without timing it, for the profiler. Run once without `-k` to see your kernels' names.)
Summarize in `docs/report/selection-profile.md`.

## Problem 5 (M3). Cross-GPU baselines

The final comparison needs the same baselines measured on every GPU: the 4060 Laptop, the 3090, and Euler's RTX 4000 Ada. Write one script that each GPU's owner can run unchanged.

a) `bench/run_baselines.sh` should:
   - build and run the cuBLAS step-cost sweep `baselines/pilot/cuda/gemm_tm.cu` (it already detects the GPU and reports the LM head separately; build with `nvcc -O3 -arch=native gemm_tm.cu -lcublas`) for `M = 1 … 512`;
   - build llama.cpp from source if it isn't there (`git clone https://github.com/ggml-org/llama.cpp && cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release && cmake --build llama.cpp/build --config Release -j`), convert Qwen2.5-1.5B-Instruct to a bf16 GGUF once (`python llama.cpp/convert_hf_to_gguf.py <model dir> --outtype bf16 --outfile qwen2.5-1.5b-bf16.gguf`), and run `llama-bench -m qwen2.5-1.5b-bf16.gguf -ngl 99 -fa 1 -p 128 -n 128 -r 3` (synthetic generation throughput);
   - run `tools/profile_reference.py` for the HF AR baseline and the NVIDIA dLLM reference (τ = 0.9; the threshold applies to the dLLM only) on 10 GSM8K-100 prompts;
   - write everything as results rows through `dllm.results`, with GPU clocks recorded.
   Reference values on the 4060 Laptop (performance profile): llama-bench tg128 = 76.0 tok/s; cuBLAS 12.81 / 13.57 / 13.87 / 16.50 ms at M = 1 / 32 / 64 / 128.

b) Run it on Euler yourself (a Slurm job under 45 minutes), and get it run on the 4060 and the 3090 by their owners. Commit the results.

c) Make one chart: pass time vs tokens per pass (`M`) for all three GPUs, normalized to `M = 1`, with each GPU's flat region marked. This is the "where does the knee sit on each GPU" result in the report.

## Problem 6 (M3). Final numbers and write-up

a) Final kernel benchmarks on your GPU and on Euler, in `results/selection/`.

b) Report section (`docs/report/selection.md`, about 1–1.5 pages): K1's exact-rounding problem and how the parallel version keeps it exact, the speedups, the K3 kernels' contribution, and the cross-GPU knee chart.

c) 2–3 slides for the video.

---

## Seams

| You need | From | When |
|---|---|---|
| Reference ops, tests, bench harness | starter code | now |
| Golden tensors and traces | Engine slice (tools exist; anyone can run them) | M0 |
| A driver that can call your ops | Engine slice | M2 |
| GPU owners to run `run_baselines.sh` | each NVIDIA-GPU owner | M3 |

| You provide | To | When |
|---|---|---|
| Correct `silu_mul`, `add_rmsnorm`, `select_commit` | Engine slice | M1 |
| Fast versions | Engine slice | M2 |
| Cross-GPU baseline results | Engine slice (final grid), report | M3 |

## Hints

- Write the tie-case test you are most worried about first, then the kernel.
- bf16 conversion: `__float2bfloat16_rn` rounds to nearest even, which is what PyTorch does.
- Exponentials: compute `expf(l - max)` in fp32. Don't use fast-math intrinsics until the exact tests pass, then check they still pass.
- `window` and `tokens` live on the device. The kernel reads the window bounds itself; the host never learns them during a replay.
- For timing, always use `bench/bench_op.py` (warm-up, median of ≥ 20 runs, clocks recorded).
