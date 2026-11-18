# Slice: Attention Path

**ECE 759 Final Project · Work slice spec**
Milestones: M0 Thu 11/19 · M1 Sun 11/22 · M2 Sun 11/29 · M3 Wed 12/2 · slides 12/3 · report section 12/6

**What you hand in** (on branch `slice/attention`, merged to `main` by pull request):
- `csrc/k3_rope_kv_write.cu` and `csrc/k2_block_attention.cu`, passing `tests/test_ops.py` for both ops.
- Benchmark results in `results/attention/` (written by `bench/bench_op.py --out-dir results/attention`).
- An Nsight Compute report summary for your final K2 (`docs/report/attention-profile.md`: a table, not the raw `.ncu-rep`).
- Your report section draft in `docs/report/attention.md`, and 2–3 slides.

**Hardware:** any NVIDIA GPU with compute capability 8.6 or 8.9 (RTX 30xx/40xx), or Euler. Everything here can be developed and measured on Euler alone.

---

## Background

You don't need to know how the whole model works. You need to know what one attention step computes, what the KV cache is, and what's special about this model. The [GUIDE](../GUIDE.md) sections 2–6 give the wider picture.

**Attention in one paragraph.** Each token position produces 12 query vectors `q` (one per query head) and 2 key/value pairs `k`, `v` (one per KV head), 128 numbers each. The output for a query is a weighted average of the values of the positions it is allowed to see, with weights `softmax(q·k / √128)` over those positions. That's it: a dot product per visible key, a softmax, and a weighted sum.

**The KV cache.** Keys and values of finished (committed) blocks never change, so they are stored once in a buffer per layer and read on every later pass. The current block's slots are scratch: each pass overwrites them. Our engine allocates the buffer once at full size (`Lmax` = 4096 slots) and writes into it in place:

```
k_cache, v_cache : bf16[2 KV heads, Lmax slots, 128]     (one pair per layer; 28 layers)
```

**Grouped-query attention (GQA).** This model has 12 query heads but only 2 KV heads. Query heads 0–5 all read KV head 0, heads 6–11 read KV head 1. So one load of a key or value can serve 6 query heads.

**What's special here: block-causal attention.** The diffusion model writes text in blocks of 32 positions. A position may see every position in its own block (in both directions) and everything in earlier blocks, but nothing in later blocks. With absolute positions `p` (query) and `j` (key):

```
visible(p, j)  =  j // 32  <=  p // 32        and   j < q_start + M
```

`q_start` is the absolute position of the first query row and `M` is the number of query rows. With block size 1 the same rule is ordinary causal attention, which is how the normal (AR) model runs. Your kernel takes `block_size` as a parameter and serves both.

**What the reference does today, and why it's slow.** Once per pass the NVIDIA code builds a boolean mask on the CPU and copies it to the GPU. Then for every layer it:
1. concatenates the whole cache with the new block (`torch.cat`), copying it,
2. copies each KV head 6 times so that every query head has its own (`repeat_kv`),
3. calls PyTorch's attention (SDPA) with the explicit mask, which rules out its fastest backend.

Counting logical memory streams, that is roughly 15× what the math needs (see `docs/research/fastdllm-v2-survey.md` §1.3). We measured what it costs per pass (all 28 layers, as CUDA Graph replays on the RTX 4060 Laptop):

| Context length | HF path (cat + repeat_kv + masked SDPA) | SDPA alone on prepared inputs (a lower bound for a library call) |
|---:|---:|---:|
| 128 | 0.9 ms | 0.3 ms |
| 1,024 | 3.1 ms | 0.7 ms |
| 4,096 | 10.6 ms | 1.9 ms |

At the start of a GSM8K answer the context is ~130 tokens and attention is a small part of a 17.6 ms pass; by the end of a long answer it is several milliseconds, and for long prompts it dominates. Your kernel reads the cache in place, loads each KV head once per tile for its 6 query heads, applies the mask by index arithmetic, and runs as one launch (or two, with a split-K reduction). The goal is to get close to the lower-bound column, or below it.

**Why the parallelism is tricky.** At batch size 1 there are only 2 KV heads. If each thread block handles one KV head, only 2 of the GPU's 24–82 streaming multiprocessors (SMs) do any work. The standard fix is **split-K** (also called flash-decoding): split the key range into chunks, let each thread block compute a partial softmax-weighted sum over its chunk (keeping its running max `m` and sum `l`), then combine the partial results in a second, small step. You will choose the chunk size and see how it trades parallelism against the cost of combining. Che Chang (TA) flagged exactly this trade-off for another team's attention kernel; it's a good thing to measure and discuss.

**Your two ops** (full contracts in [INTERFACES.md §3.2–3.3](../INTERFACES.md)):

```
rope_kv_write(q[M,12,128] (in place), k[M,2,128], v[M,2,128], k_cache, v_cache (in place),
              q_start: int32[1] on device, cos[Lmax,128], sin[Lmax,128])
block_attention(q[M,12,128], k_cache[2,Lmax,128], v_cache[2,Lmax,128],
                q_start: int32[1] on device, block_size) -> out[M,12,128]
```

`M` is 1 (AR), 32 (one block) or 64 (two consecutive blocks, for the engine's P2 strategy). P3 would need a two-branch variant that is not part of this spec unless the Engine slice asks for it (INTERFACES.md §3.3). Both ops must be **graph-safe**: they read `q_start` from device memory, never call `.item()`, and launch on the current CUDA stream. The engine records each pass as a CUDA Graph and replays it with different values in the same buffers.

---

## Problem 1 (M0). Set up and read

a) Set up your environment ([ENVIRONMENT.md](../setup/ENVIRONMENT.md)) and check it:
```
uv run python tools/smoke_test.py
uv run pytest -q
```
On Euler, run `sbatch scripts/euler/setup_env.sh` once, then `sbatch scripts/euler/run_tests.sh`.

b) Read `dllm/ops/reference.py`: the functions `rope_kv_write` and `block_attention`. These are the ground truth; your kernels must match them. Read `csrc/ops.cpp` (the op schemas), `csrc/example_axpy.cu` (a complete working kernel showing the launch pattern: device guard, current stream, launch check), then the two stub files you will fill in.

c) Get the golden tensors (real inputs and outputs from the model) with `tools/fetch_golden.sh <url from the team chat>`, or make them yourself with `uv run python tools/dump_golden.py` on any NVIDIA GPU or Euler. Load one case and print the shapes:
```
uv run python -c "from dllm import golden; m, t = golden.load('artifacts/golden/dllm-denoise-13-block_attention'); print(m); print({k: v.shape for k, v in t.items()})"
```

d) Create your branch `slice/attention` and push it.

## Problem 2 (M1). RoPE + KV-cache write

This is the warm-up: an elementwise kernel, but with exact rounding rules.

a) Implement `csrc/k3_rope_kv_write.cu` per the contract. For each query row `i`, position `p = q_start + i`, and each head, rotate the 128-vector with the rotate-half rule using row `p` of `cos`/`sin`, rounding each product to bf16 and adding in bf16 (as the reference does). Write rotated `q` back in place. Write rotated `k` and unchanged `v` into the cache slots `p`. Leave every other cache slot untouched.

b) Pass the tests (the summary must say *passed*; *skipped* means your op wasn't run):
```
uv run pytest tests/test_ops.py -k rope_kv_write -v
```

c) Measure it and report achieved bandwidth (bytes read + written / time) for `M ∈ {1, 32, 64}`:
```
uv run python bench/bench_op.py --op rope_kv_write --backend ours --graph --out-dir results/attention
uv run python bench/bench_op.py --op rope_kv_write --backend torch --graph --out-dir results/attention
```
It moves very little data, so it will be latency-bound. What matters is that it replaces several PyTorch launches with one. Report the time for both backends.

## Problem 3 (M1). Block attention, correct first

a) Write a first version of `csrc/k2_block_attention.cu` that is simple and obviously correct. Suggested shape: one thread block per (query row, query head); each block loops over the visible keys of its KV head (`h // 6`), computes scores in fp32 with an online softmax (running max and sum), and accumulates the weighted values. Don't optimize yet.

b) Pass the tests for every combination the contract requires (`M ∈ {1, 32, 64}`, `block_size ∈ {1, 32}`, several `q_start` values including ones that are not multiples of 32, and the golden cases):
```
uv run pytest tests/test_ops.py -k block_attention -v
```

c) Record the v1 time for `M ∈ {1, 32, 64}` at context lengths `L ∈ {128, 512, 1024, 2048, 4096}`:
```
uv run python bench/bench_op.py --op block_attention --backend ours --sweep-L --graph --out-dir results/attention
```
This is the baseline your optimization will beat.

## Problem 4 (M2). Block attention, fast

Now make it fast. Work in steps and measure after each one; the sequence of measurements is a result for the report.

a) **Share KV loads across the 6 query heads.** Have each thread block handle all 6 query heads of one KV head for a tile of query rows, staging tiles of K and V in shared memory so each tile is read from global memory once per thread block.

b) **Split the key range (split-K).** Launch several thread blocks per (KV head, query tile), each over a chunk of keys, writing fp32 partial outputs plus their running max `m` and sum `l`. Then combine them in a second kernel (or with a last-block-done reduction): rescale each partial by `exp(m_i − m)` before summing. A chunk can be entirely masked for some rows (its `l` is 0); handle that without producing NaNs. Sweep the chunk size and pick one per GPU; explain the trade-off.

c) **Skip what isn't visible.** Whole key tiles past `q_start + M` are never loaded. Inside the current block, the mask is all-visible for block size 32, so the mask only matters at block boundaries.

d) *Optional:* use tensor cores (WMMA or `mma.sync`, bf16 inputs with fp32 accumulation) for the `q·k` and `p·v` products. With `M = 32` and 6 heads per KV head you have 192 query rows per KV head, which is enough to fill a tensor-core tile. Rounding the probabilities to bf16 for the `p·v` product adds error; check you stay within the tolerance.

e) Compare against the baselines at `M ∈ {1, 32, 64}` and `L ∈ {128, 512, 1024, 2048, 4096}`, all as graph replays:
```
uv run python bench/bench_op.py --op block_attention --baseline hf_cat      --sweep-L --graph --out-dir results/attention
uv run python bench/bench_op.py --op block_attention --baseline sdpa_nomask --sweep-L --graph --out-dir results/attention
uv run python bench/bench_op.py --op block_attention --backend ours         --sweep-L --graph --out-dir results/attention
```
   - `hf_cat` is what the HF reference does (cat + repeat_kv + masked SDPA, at the real length);
   - `sdpa_nomask` is SDPA alone on prepared inputs without a mask: a lower bound for a library call. It is a timing reference only; it computes different values, since it ignores the mask;
   - `--backend torch` is our full-capacity PyTorch reference: a correctness reference, slow by design.
   Plot time vs `L` for all of them. Run the same sweep on Euler (RTX 4000 SFF Ada) so the result is reproducible.

f) Profile your final version with Nsight Compute at `M = 32`, `L = 1024`:
```
mkdir -p results/attention
ncu --set full -k regex:<your kernel's name> -o results/attention/k2 \
    .venv/bin/python bench/bench_op.py --op block_attention --backend ours --m 32 --L 1024 --iters 3
```
(`--iters` below 20 runs the op that many times without timing it, which is what you want under the profiler. On Euler use `~/dllm-venv/bin/python`. Run once without `-k` to see your kernel's actual name.)
Summarize: achieved memory throughput, occupancy, warp stall reasons, and what limits it. Put the table in `docs/report/attention-profile.md`.

## Problem 5 (M2). Inside the engine

a) With the Engine slice, switch both ops to `ours` in the driver (`ops.set_backend("block_attention", "ours")`, same for `rope_kv_write`, before the graphs are captured) and check that generation still matches the reference token for token on the parity prompts:
```
uv run pytest tests/test_engine.py -v      # the Engine slice's parity test
```

b) Report the change in end-to-end ms per pass with your two ops switched on, everything else unchanged. This is your bar in the waterfall chart.

## Problem 6 (M3). Final numbers and write-up

a) Final benchmark runs on your GPU and on Euler, saved in `results/attention/`.

b) Report section (`docs/report/attention.md`, about 1–1.5 pages): the problem (what the reference does), your design (decomposition, split-K, shared loads), the step-by-step speedups from Problem 4, the time-vs-context-length plot against both baselines, the Nsight Compute summary, and what limits it now.

c) 2–3 slides for the video.

---

## Seams

| You need | From | When |
|---|---|---|
| Reference ops, tests, bench harness | starter code | now |
| Golden tensors | Engine slice (tool exists; anyone can run it) | M0 |
| A driver that can call your ops | Engine slice | M2 |

| You provide | To | When |
|---|---|---|
| Correct `rope_kv_write`, `block_attention` | Engine slice | M1 |
| Fast versions | Engine slice | M2 |
| ms per pass with your ops on | Engine slice (waterfall chart) | M2 |

## Hints

- Start every kernel with the input checks already in the stub; a wrong stride is the most common bug.
- The tests compare against an fp32 computation of the same math, not against bf16 PyTorch bit for bit. Accumulate in fp32 and you'll be inside the tolerance.
- `q_start` lives in device memory. Read it at the top of the kernel; don't copy it to the host.
- Cache slots past `q_start + M` can hold stale values from earlier passes; your kernel must never read them.
- For timing, always use `bench/bench_op.py`; it warms up, takes the median of ≥ 20 runs, and records GPU clocks.
- The CUDA ops compile on first use (`dllm.ops.cuda.load()`), which takes a minute or two. On Euler that happens inside your job; run nothing on the login node.
