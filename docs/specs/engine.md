# Slice: Engine (lead)

**ECE 759 Final Project · Work slice spec**
Milestones: M0 Thu 11/19 · M1 Sun 11/22 · M2 Sun 11/29 · M3 Wed 12/2 · slides 12/3 · report section 12/6

This slice is the critical path: every other slice either plugs into the engine or measures it. Its first job is to keep the others unblocked (golden tensors, traces, a driver they can switch their ops into), and only then to make the engine fast.

**What you hand in** (on branch `slice/engine`, merged to `main` by pull request):
- `dllm/engine/`: the driver (decode loop, static KV cache, CUDA Graph capture, per-op backend switch, AR mode, P1 and whichever of P2/P3 the data supports).
- The published golden tensors and the trace tool runs the other slices need.
- `bench/run_grid.py` and the final benchmark grid in `results/grid/`, plus the ablation waterfall chart.
- The reference profile (`docs/research/reference-profile.md`).
- Report sections: the engine design, the waterfall, the overall results. Slides for the same.

**Hardware:** a local NVIDIA GPU (the RTX 4060 Laptop is the main target).

---

## Background

[PLAN.md](../PLAN.md) §4–5 and [INTERFACES.md](../INTERFACES.md) §3.6 are the background for this slice. The starter code already contains:
- `dllm/reference_decode.py`: a re-implementation of the reference decode loop that matches `model.generate()` token for token (verified on 5 prompts × 3 thresholds);
- `dllm/engine/single_pass.py`: a working spike of the static-cache layer loop (gate and up weights already packed) with one pass captured as a CUDA Graph. In the first experiment it matched the HF forward exactly and took 16.1 ms per pass vs 17.2 ms for the HF forward (4060, performance profile, 96-token prefix, cache capacity 160);
- `dllm/engine/eos.py`: `eos_ready`, the graph-safe stop check;
- `dllm/ops/`: the op registry, with PyTorch reference versions of all five ops;
- `tools/profile_reference.py` and its results in [research/reference-profile.md](../research/reference-profile.md): the reference takes 17.6 ms per pass on the performance profile, 13.8 of it in matrix multiplies already at the cuBLAS floor.

**Watch out:** the PyTorch reference `block_attention` always works over the whole cache capacity. At capacity 4096 it costs ~9 ms per pass. Until K2 lands, run stage 1 with the cache capacity bucketed to the current length (for example one captured graph per 256-token bucket), or with SDPA over the valid prefix only.

Start from those; don't rewrite them.

---

## Problem 1 (M0). Unblock everyone

a) Generate the golden tensors with `tools/dump_golden.py`, check their size, and publish them as a GitHub release asset so `tools/fetch_golden.sh` works for everyone. Post the URL in the team chat.

b) Check that `tools/dump_traces.py` works on Euler (the Accuracy slice needs traces), and that `scripts/euler/setup_env.sh` succeeds on Euler end to end.

c) Rerun `tools/profile_reference.py` on the 4060 (performance profile) so the reference numbers in `docs/research/reference-profile.md` are yours, and keep the same settings for the final run.

## Problem 2 (M1). Driver stage 1: correct, graph-captured, stock ops

a) Turn the spike into the driver: a decode loop that follows INTERFACES.md §3.6 exactly, running every pass through the static-cache layer loop with all ops on the `torch` backend.

b) Capture the denoise pass and the commit pass as CUDA Graphs (prefill can stay eager). Keep one small device-to-host copy per pass: K1's `stats` plus the `eos_ready` flag, written into one buffer.

c) **Parity:** generated token IDs (including the stop token) identical to `reference_decode.py` on the parity prompts at τ ∈ {1.0, 0.9, 0.7}, plus one exactly-32-token prompt. Create `tests/test_engine.py` for this; the other slices run it after switching their ops on.

d) **AR mode:** the same driver with Qwen2.5-1.5B-Instruct weights. The prompt is prefilled with ordinary causal attention; each decode step is `M = 1`, `block_size = 1`, plain greedy. Parity with HF `generate(do_sample=False, repetition_penalty=1.0)`.

e) First two waterfall bars, end to end on 10 GSM8K-100 prompts at τ = 0.9: the reference `generate()`, then stage 1 (static cache + graphs, stock ops).

## Problem 3 (M2). Integrate the kernels

a) As each CUDA op lands (Attention and Selection slices), switch it on in the driver (recapture the graphs after switching), rerun parity, and record one waterfall bar per op group: K3 fused ops, K2 attention, K1 selection.

b) Pack the Q/K/V weights and biases into one projection (gate and up are already packed in the spike) and record the effect. Expect it to be small: the matrix multiplies are already near the cuBLAS floor.

c) Keep the per-op backend switch working, so any bar of the waterfall can be reproduced with a command line.

## Problem 4 (M2). P1, P2, P3

a) **P1:** stop at the first fully decoded EOS without the commit pass. The reference already does this in `generate()`; make sure the driver does, and measure it against a "finish the block" baseline.

b) **P2 and P3:** read the Accuracy slice's hit-rate note (Wed 11/25). Build P2 if the T0 guess hits ≥ ~70%: it runs two consecutive blocks as `M = 64` rows, which `block_attention` already supports. Build P3 only if `h` ≳ 0.4 *and* there is time to design it: its two halves are the same block at the same positions, so it needs a two-branch attention variant and separate scratch K/V per branch (see INTERFACES.md §3.3). Both must produce output identical to the plain driver, including when a guess is rejected and when the stop token appears. If a hit rate is too low, report it and skip the build.

## Problem 5 (M3). The final grid

a) `bench/run_grid.py` runs, at batch size 1 on 10 GSM8K-100 prompts (plus one prose prompt set):
   - {dLLM, AR} × {reference, our engine}, the dLLM at τ ∈ {1.0, 0.9, 0.7};
   - llama.cpp AR throughput from the Selection slice's baseline script (`llama-bench` measures synthetic generation speed; running llama.cpp on the actual prompts would need its own runner with the same template and stop rules, which is optional).
   It writes results rows only; charts read those files.

b) Run it on the 4060 (performance profile, clocks logged), on Euler, and get it run on the 3090.

c) Charts: the waterfall (ms per pass, reference → each optimization), tokens/s for the full grid, and the combined speed-vs-accuracy chart with the Accuracy slice. Report total wall time, generated tokens and passes alongside ms per pass: P1–P3 remove passes without changing ms per pass.

## Problem 6. Write-up

Report sections: engine design, the waterfall, overall results and conclusion. Slides for the same, and the overall structure of the video.

---

## Seams

| You provide | To | When |
|---|---|---|
| Golden tensors, traces, Euler setup working | Attention, Selection, Accuracy | M0 |
| A driver with a per-op switch | Attention, Selection | M1 (stage 1), M2 |
| Engine with `ours` backend | Accuracy (accuracy check) | M2 |

| You need | From | When |
|---|---|---|
| Correct and fast CUDA ops | Attention, Selection | M1, M2 |
| P2/P3 hit rates | Accuracy | Wed 11/25 |
| Cross-GPU baselines, llama.cpp numbers | Selection | M3 |
| GSM8K accuracy per setting | Accuracy | M2 |
