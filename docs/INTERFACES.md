# Interfaces

This is the contract between the pieces of work. If your code matches what is written here, it fits with everyone else's, and nobody has to wait on anybody. Change this file only by pull request, and tell the team when you do.

Status: **v1**. The semantics below were verified against the reference code on 2026-11-17 in two experiments: a re-implementation of the decode loop matched `model.generate()` token for token on 5 prompts × 3 thresholds, and a layer loop built from these ops matched the HF forward exactly (max abs diff 0) on a 96-token prefix plus a 32-row block. The checked-in tests repeat smaller versions of both checks (`tests/test_reference_ops.py`, `tests/test_reference_decode.py`).

---

## 1. Repository layout

```
dllm/                     Python package (import dllm)
  config.py               model ids, special token ids, block sizes
  models.py               load the HF models and tokenizer
  prompts.py              the one GSM8K prompt every run uses
  reference_decode.py     the reference decode loops (dLLM and AR), with trace hooks
  ops/
    __init__.py           the op registry: ops.<name>(...) and set_backend()
    reference.py          PyTorch versions of every op: the ground truth
    cuda.py               loads our compiled kernels (torch.ops.dllm.*)
  engine/                 the driver (today: a single-pass static-cache spike, and eos_ready)
  golden.py               save/load saved real tensors
  trace.py                per-pass trace records
  results.py              results records + run metadata
  timing.py               CUDA-event timing, clock logging
csrc/                     our CUDA kernels, one file per op, plus the registration file
tests/                    pytest: every op, ours vs reference
bench/                    microbenchmarks per op, and the end-to-end runners
eval/                     GSM8K harness and analysis
tools/                    download models, dump golden tensors and traces, profile the reference, smoke test
scripts/euler/            Slurm jobs
docs/                     plan, guide, interfaces, specs, setup
```

## 2. Model constants

| Name | Value |
|---|---|
| dLLM | `Efficient-Large-Model/Fast_dLLM_v2_1.5B` (needs `trust_remote_code=True`) |
| AR | `Qwen/Qwen2.5-1.5B-Instruct` |
| layers / hidden / MLP | 28 / 1536 / 8960 |
| heads | 12 query, 2 key/value (GQA, 6 query heads per KV head), head dim 128 |
| vocab | 151,936 |
| dtype | bf16 |
| RoPE θ | 1,000,000 |
| RMSNorm ε | 1e-6 |
| dLLM special ids | MASK 151665, stop 151645 |
| AR stop ids | 151645 or 151643 |
| Decoding | temperature 0 for both; the AR model is plain greedy (no repetition penalty, even though its checkpoint config sets 1.1) |
| block size / sub-block size | 32 / 8 |

## 3. The ops

Every op exists twice: a PyTorch version in `dllm/ops/reference.py` (the ground truth) and our CUDA version, registered as `torch.ops.dllm.<name>`. Today the CUDA versions are stubs that raise "not implemented yet", and their tests skip until they are filled in. The engine calls ops only through the registry (`reference_decode.py` and `tools/dump_golden.py` call the PyTorch ops directly on purpose, so they always produce ground truth):

```python
from dllm import ops
ops.set_backend("block_attention", "ours")    # or "torch"; per op
out = ops.block_attention(q, k_cache, v_cache, q_start, block_size)
```

Rules for every CUDA op:
- **Graph-safe.** No host-device sync, no `.item()`, no data-dependent shapes. Values that change between passes (like `q_start`) are read from a device tensor, never passed as a Python int.
- **Writes in place where marked.** Outputs marked *in place* are written into caller-owned buffers; the driver preallocates them once.
- **bf16 in, bf16 out, fp32 accumulate**, unless stated otherwise.
- **Contiguous inner dimension.** Inputs may be views with strides; the last dimension is always contiguous. Check and raise otherwise.
- **Tested** against the reference with the tolerance below, on random inputs and on the saved real tensors (section 4).

Shapes: `M` = query rows in this pass (1 for AR, 32 for a block, 64 for two consecutive blocks). `Lmax` = cache capacity (default 4096). `V` = 151,936. Always `0 ≤ q_start ≤ Lmax − M`.

### 3.1 `add_rmsnorm` (K3)

```
add_rmsnorm(x: bf16[M, 1536], residual: bf16[M, 1536] (in place), weight: bf16[1536], eps: float)
  -> normed: bf16[M, 1536]
```
`residual += x` in bf16 (the residual stream is never kept in fp32). Then on the summed residual: convert to fp32, square, mean over the 1536 values, add ε, rsqrt, multiply the fp32 values, **cast to bf16, then multiply by the bf16 weight**. `x` may be `None` when `residual` already holds the whole input (the first layer's input norm, where `residual` is the token embeddings): then it is a plain RMSNorm of `residual`. Everywhere else, including the final norm (where `x` is the last MLP output), pass `x`.
Tolerance: every element of `normed` within 1 bf16 ulp in ≥ 99.9% of elements; the updated `residual` exactly equal (it is a plain bf16 add).

### 3.2 `rope_kv_write` (K3)

```
rope_kv_write(q: bf16[M, 12, 128] (in place), k: bf16[M, 2, 128], v: bf16[M, 2, 128],
              k_cache: bf16[2, Lmax, 128] (in place), v_cache: bf16[2, Lmax, 128] (in place),
              q_start: int32[1] (device), cos: bf16[Lmax, 128], sin: bf16[Lmax, 128])
```
Applies rotary embedding to `q` (in place) and `k` at absolute positions `q_start .. q_start+M-1`, and writes rotated `k` and unchanged `v` into cache slots `[q_start, q_start+M)`. Rotate-half form: with `x1 = x[..., :64]`, `x2 = x[..., 64:]`, `rotate_half(x) = [-x2, x1]` and `out = x*cos + rotate_half(x)*sin`, each product rounded to bf16 and added in bf16. `cos`/`sin` are generated in fp32 for θ = 1e6 by the driver and **stored as bf16**: `cos, sin: bf16[Lmax, 128]`. Positions are the same in denoise and commit passes. A denoise pass writes its block's K/V into slots `[q_start, q_start+M)` as scratch; only a commit pass advances the driver's committed length.
Tolerance: `q` and the written K slots within 1 bf16 ulp in ≥ 99.9% of elements; written V slots exact; cache slots outside the range unchanged.

### 3.3 `block_attention` (K2)

```
block_attention(q: bf16[M, 12, 128], k_cache: bf16[2, Lmax, 128], v_cache: bf16[2, Lmax, 128],
                q_start: int32[1] (device), block_size: int) -> out: bf16[M, 12, 128]
```
Query row `i` sits at absolute position `p = q_start + i`. It attends to cache slots `j` in `[0, q_start + M)` with `j // block_size <= p // block_size`. So `block_size = 32` is the dLLM rule (bidirectional inside a block, causal across blocks) and `block_size = 1` is ordinary causal attention for AR mode. The bound `j < q_start + M` matters: the cache is allocated at full capacity, and slots past the current block hold stale data (the driver initializes the cache to zeros so they are at least finite). Query head `h` uses KV head `h // 6`. Scale `1/sqrt(128)`, softmax in fp32.
Must support `M ∈ {1, 32, 64}` and `q_start + M ≤ Lmax`. Prefill (long prompts) may use the PyTorch path.
Tolerance: the tests compute the same math in fp32 as the oracle. Your output must be within 2e-2 of it (max abs diff), and no further from it than the stock bf16 PyTorch op is.
**Performance note.** The PyTorch reference always works over the whole `Lmax` cache, so it costs ~9 ms per pass at `Lmax = 4096` whatever the context; the engine must not run it at full capacity. See `docs/research/reference-profile.md` for the HF path's cost by context length.
**Not covered yet: P3 branches.** P3 (two speculative versions of the *same* block) needs both halves at the same positions, the same prefix, and each half attending only to its own block. Consecutive `M = 64` rows (two different blocks) cannot express that. If the P3 hit rate justifies building it, the Engine slice will add a two-branch variant here first.

### 3.4 `silu_mul` (K3)

```
silu_mul(gate_up: bf16[M, 17920]) -> out: bf16[M, 8960]
```
`out = silu(gate) * up`, where `gate = gate_up[:, :8960]` and `up = gate_up[:, 8960:]`. `silu(gate)` is rounded to bf16, then multiplied by `up` in bf16, as HF does. HF has separate gate and up projections; the driver concatenates their weights (gate first) once at load time so a single cuBLAS call produces `gate_up`.
Tolerance: 1 bf16 ulp in ≥ 99.9% of elements.

### 3.5 `select_commit` (K1)

```
select_commit(logits: bf16[32, V], tokens: int64[32] (in place), window: int32[2] (device),
              tau: float, mask_id: int) -> stats: int32[3] (device)
```
One denoise step of the reference decoder, entirely on the GPU. `logits` are the raw LM-head outputs for the 32 block positions; `window = [start, end)` is the active 8-slot group.
1. **Shift.** Block slot `i` is predicted by logits row `max(i-1, 0)`.
2. **Probabilities exactly as the reference rounds them.** For each slot, softmax over the row is accumulated in fp32 and each probability is **rounded to bf16**. The guess is the **first** vocabulary index whose rounded probability is the largest, and the confidence is that rounded value. This is not the same as the raw-logit argmax: bf16 logits `[0, 0.001]` round to probabilities `[0.5, 0.5]`, so index 0 wins. A fast version finds the row max and the fp32 sum of `exp(l - max)` in one pass, then finds the first index whose rounded probability equals the rounded maximum.
3. **Eligible slots** are those in `[start, end)` whose token is still `mask_id`. All others count as confidence −∞.
4. **Commit** every eligible slot whose confidence is strictly greater than **τ rounded to bf16** (so τ = 0.9 means > 0.8984375, and the smallest passing value is 0.90234375). Also always commit the eligible slot with the highest confidence, the leftmost on ties, even when τ = 1.
5. Write the committed guesses into `tokens`. `stats = [n_committed, window_done, block_done]`, where `window_done` means no `mask_id` is left in `[start, end)` and `block_done` means none is left in the block. If the window has no eligible slot, nothing is committed and `stats = [0, 1, block_done]`.

End-of-text handling stays in the driver (`dllm.engine.eos.eos_ready`, graph-safe): it stops when a committed stop token has no MASK before it. The first token of the next block comes from the commit pass's last logits row by plain argmax of the logits, not from this op.
Correctness: identical committed set and tokens to the reference on every recorded step (section 4), not just "close".

## 3.6 Decode-loop rules the driver must reproduce

Verified against `model.generate()` (batch size 1, `block_size=32`, `small_block_size=8`, `use_block_cache=False`, temperature 0). `dllm/reference_decode.py` implements exactly these rules and is the reference for both the driver and the accuracy harness.
- **Prefill.** For a prompt of `N > 32` tokens, the first `32·⌊N/32⌋` tokens are prefilled into the cache in one pass. If `N` is a multiple of 32, the argmax of the last logits row is appended as the first generated token. The remaining `N mod 32` prompt tokens start the first block, followed by MASKs up to 32.
- **Short prompts (`N < 32`)** share the first block with the generated tokens; nothing is prefilled.
- **Edge case `N = 32` exactly:** the reference does *not* prefill, and the first block is 32 MASKs that never see the prompt (a quirk of `generate()`). The parity tests include one 32-token prompt as a regression check, and the driver reproduces the quirk.
- **Blocks** are aligned to absolute positions that are multiples of 32.
- **Denoise passes** run all 32 block slots without advancing the committed cache (the committed prefix is read-only; the block's own cache slots are scratch that each pass overwrites), then `select_commit` on the active window. Windows advance left to right, 8 slots each.
- **Commit pass** when the block has no MASK left: run the 32 final tokens again, writing their K/V into the cache, and append the plain argmax of the last row as the next block's first token.
- **Stop:** once a generated stop token (151645) is committed and no MASK precedes it in the generated text, stop without the commit pass. The output includes that EOS and drops everything after it.
- **Budget:** the reference runs `⌊max_new_tokens/32⌋` blocks, so it can return slightly more or fewer tokens than asked. Always report the actual generated length.

## 4. Saved real tensors ("golden" files)

`tools/dump_golden.py` runs the reference models on a fixed GSM8K prompt and saves 27 cases (~41 MB): for each of `add_rmsnorm`, `rope_kv_write`, `block_attention`, `silu_mul`, inputs captured from real HF passes (a dLLM denoise pass, a dLLM commit pass and an AR step, at layers 0 and 13) with the outputs the PyTorch reference op produces from them; plus three `select_commit` cases with raw logits from a real pass at τ = 0.7, 0.9 and 1.0. One case per directory:

```
artifacts/golden/<case>/              e.g. dllm-denoise-13-block_attention, dllm-select-0.9
  meta.json              {"op": "select_commit", "model": "dllm", "kind": "denoise", "layer": null, "pass": 10,
                          "q_start": 64, "M": 32, "block_size": 32, "prompt_id": "gsm8k-0005",
                          "git_sha": "...", "scalars": {"tau": 0.7, "mask_id": 151665}}
  tensors.safetensors    the op's tensor arguments by name, plus "expected_<name>" for its output and every in-place argument
```

`dllm.golden.load(case_dir) -> (meta, tensors)`. `tests/test_ops.py` runs every case it finds, with the op name in the test ID (`pytest -k select_commit` selects them). The cases are committed as one archive, `tests/fixtures/dllm-golden-v1.tar.gz` (32 MB; its root holds the case folders). Run `tools/fetch_golden.sh` once to extract it into `artifacts/golden/` (gitignored), or regenerate the cases with `uv run python tools/dump_golden.py` on any NVIDIA GPU or Euler. If you regenerate them for a new version, commit a new archive under a new name rather than overwriting this one.

## 5. Per-pass traces

`tools/dump_traces.py` records every pass of a generation as one JSON line, using the harness's GSM8K prompt (`dllm/prompts.py`). Used to measure how often P2/P3 guesses would be right. Records hold decisions, not logits.

- `pass` counts from 0 after prefill, across the whole generation; denoise and commit records interleave in order. `block` counts blocks. Positions are block-local (0–31).
- `committed`: slots filled on this pass. `guesses`: slots still masked *after* this pass's commits, with this pass's argmax and confidence (bf16-rounded, as K1 computes them).
- Commit records have `window: null`, empty `committed`/`guesses`, and `t0_guess` (raw argmax of the last row in the block's final denoise pass) and `t0_actual` (from the commit pass).

```json
{"run_id": "...", "prompt_id": "gsm8k-0042", "model": "dllm", "tau": 0.9,
 "block": 3, "pass": 7, "kind": "denoise",              // or "commit"
 "window": [8, 16], "q_start": 160,
 "committed": [[9, 1234, 0.97], [10, 88, 0.95]],       // [position, token, confidence]
 "guesses": [[11, 4021, 0.62], [12, 13, 0.41]],        // argmax + confidence for every still-masked position in the block
 "t0_guess": 2211, "t0_actual": 2211}                   // commit passes only
```

## 6. Results

Every timed or graded run appends JSON lines to `results/<experiment>/<run_id>.jsonl` through `dllm.results.ResultWriter`, which also writes `results/<experiment>/<run_id>.meta.json` once (including `slurm_job_id` when run under Slurm). There are two kinds of rows: **generation rows**, one per prompt, timed by wall clock end to end; and **microbenchmark rows** from `bench/bench_op.py`, one per shape, timed with CUDA events (`--out-dir` picks the folder, default `results/bench_op/`).

```json
// meta: written once per run
{"run_id": "...", "git_sha": "...", "host": "...", "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU",
 "driver": "...", "torch": "...", "clocks": {"sm_mhz": 2370, "mem_mhz": 8001, "power_limit_w": 80},
 "engine": "reference" | "ours" | "llamacpp" | "hf", "model": "dllm" | "ar",
 "tau": 0.9, "block_size": 32, "sub_block": 8, "ops": {"block_attention": "ours", ...}}
// one row per prompt
{"run_id": "...", "prompt_id": "gsm8k-0042", "prompt_tokens": 112, "gen_tokens": 231,
 "passes": 131, "ms_total": 2010.4, "ms_per_pass": 15.3, "tok_per_s": 114.9,
 "pred": "18", "gold": "18", "correct": true}
```

Tokens are counted up to and including the first stop token, for both models. Charts read only these files. `power_limit_w` may be `null` when the driver doesn't report it.

## 7. GSM8K subset

`eval/gsm8k_100.jsonl`: 100 problems from the GSM8K test split, chosen with `random.Random(759).sample(range(1319), 100)`, sorted by index. Each line: `{"id": "gsm8k-0042", "index": 42, "question": "...", "answer": "18"}`. Generated by `tools/make_gsm8k_subset.py` and committed, so every machine uses the identical list.

## 8. Timing

- **Warm the GPU up first.** An idle laptop GPU runs at about 210 MHz SM / 405 MHz memory and takes a moment to ramp up; short timings taken cold are 3–10× too slow. `dllm.timing.benchmark` runs the op back to back for 1 s before measuring. Check the recorded clocks.
- CUDA events around the measured region, then the median of ≥ 20 runs.
- Compare kernels as CUDA Graph replays (`bench/bench_op.py --graph`), because that is how the engine runs them; without a graph, tiny ops are dominated by launch overhead.
- Achieved bandwidth = logical bytes the op must move (inputs read + outputs written, counted once) / median time. That is not measured DRAM traffic; use Nsight Compute for that.
- Laptops on the performance power profile, plugged in. Record GPU clocks with every run (`dllm.timing.gpu_state()`; the scripts do this).

## 9. Euler

- Partition `instruction`, 1 GPU (`--gres=gpu:1`), at most 45 minutes per job.
- Modules: `gcc/14.3.0` and `nvidia/cuda/13.0.0` (loaded by `scripts/euler/job_template.sh`). Our virtual environment lives at `~/dllm-venv`, models at `~/hf_cache` (`HF_HOME`).
- Start every job from `scripts/euler/job_template.sh` and submit it from the repo root. Our CUDA ops build inside the job on first use; run nothing on the login node.

## 10. Git

- `main` always works. Each piece of work develops on its own branch (`slice/<name>`) and merges by pull request.
- Commit and push small and often; the course grades the commit history.
- Never commit model weights, golden tensors, traces or large results. Results under 1 MB that a chart uses may be committed.
