# Project Plan: Fast dLLM Inference on Edge GPUs

ECE 759, Fall 2026. Team: Sparsh Mishra, Munasib Ilham, M Sadman Sakib, Ayesha Shafique.
Work division and interfaces come next and will be added as a section here.

Source material in this folder:
- `final-project-kick-off.pdf`, `ECE_759_Proposal.pdf`, `ECE759 - Final Project Proposal Feedback.pdf`, `DLLM-Idea-Prose.pdf`
- `research/pilot-measurements.md`: every number we measured on the 4060 in October
- `research/fastdllm-v2-survey.md`: line-by-line audit of the NVIDIA reference code, the GitHub issues and the related papers

---

## 1. Dates

| Date | What |
|---|---|
| **12/2** | Results frozen. Final benchmark runs done. |
| **~12/4** | Record the presentation video |
| **12/7 23:59** | Final report PDF; code frozen |

The kick-off slides say grading looks at **steady commits from everyone**. Commit and push small, often.

---

## 2. What changed since the proposal, and why

The proposal said we would write a full inference engine from scratch with no PyTorch. Both TAs called that too ambitious:

- **Chih-Chun Chang:** reuse the HF tokenizer and weights, focus on the CUDA kernels, report the laptop GPU, rerun kernel benchmarks on Euler.
- **Che Chang:** reproducing model outputs is already a huge effort; reusable cache state is tricky; matching cuBLAS/PyTorch is hard. Build *"a minimal inference driver around the selected dLLM, then stretch."*

So we keep the goal (an optimized dLLM inference engine, compared against the best AR engine) and change the approach:

- **HF and PyTorch do the plumbing:** weights, tokenizer, the extension build, and **cuBLAS for every matrix multiply**. We don't try to beat cuBLAS. Our own measurements say we shouldn't (see 4.2).
- **Our code is everything that is specific to diffusion decoding**:
  - the decode loop and its KV cache
  - CUDA Graph capture
  - custom CUDA kernels for the dLLM-only operations (block-causal attention, parallel token selection) and fused small ops
  - new decoding strategies that use the GPU's spare compute

These are the parts where PyTorch eager is slow and where no library has a kernel.

Sparsh will explain this change to the TAs in person. The report must cite the feedback as the reason for it.

---

## 3. Research questions

**RQ1. Engine:** how much of the reference dLLM's per-step overhead can our driver and kernels remove **without changing the output**?
- Target: 22 ms → ~14.5–16 ms per step on the 4060.
- Shown as an ablation waterfall (ms/step after each optimization). You can read more about it [here](https://medium.com/@rajilini/ablation-study-what-is-it-in-machine-learning-0a1d362b366d). It is basically a fancy way to say that we measure performance of the system as we add features.

**RQ2. dLLM vs AR:** with both on fast engines, is the dLLM worth it?
- **Headline:** our dLLM engine vs **llama.cpp** running Qwen2.5-1.5B-Instruct (the best AR engine).
- **Calibration:** our own engine in AR mode (block size 1, Qwen2.5 weights, same kernels). If it lands within ~10% of llama.cpp, our engine is "flagship-grade", and the dLLM's advantage is clearly due to the decoding method, not a better or worse engine.
- Metrics: tok/s, ms per token, tokens per forward pass, GSM8K accuracy, all across the dLLM confidence threshold τ. Shown as a speed-vs-accuracy chart.

**RQ3. Forward passes:** at batch 1 the GPU is memory-bound, so a 64-row pass costs about the same as a 32-row pass. Can that free compute cut the **number of passes** per answer?
- Measure the opportunity offline first, then build only what the data supports (see 5.3).

Expected outcome, from the pilot numbers at ~13–15 ms/step:
- dLLM: ~130 tok/s on math and ~80 tok/s on prose.
- AR: ~70 tok/s on our engine and 64.6 tok/s on llama.cpp.

That would be a real but task-dependent win. RQ3 is what could widen it.

---

## 4. Background you need (short)

### 4.1 The two models
- **AR:** `Qwen/Qwen2.5-1.5B-Instruct`, producing one token per forward pass.
- **dLLM:** `Efficient-Large-Model/Fast_dLLM_v2_1.5B`, the same Qwen2.5 architecture lightly fine-tuned into a *block diffusion* model.
- Architecture: 28 layers, hidden size 1536, 12 query heads / 2 KV heads, head dim 128, vocab 151,936, bf16, ~3.09 GB of weights.

How the dLLM generates (reference code: HF `modeling.py` `generate()`):
1. Text is produced in blocks of 32 tokens. A block starts as `[T0, MASK × 31]`, where T0 comes from the previous block.
2. Each **denoise pass** runs all 32 block positions through the model. Block positions attend to the committed cache and to each other (bidirectional inside the block, causal across blocks).
3. The logits are shifted by one position: position *i* is predicted from output *i−1*. The work proceeds through sub-blocks of 8 positions.
4. Every masked position whose top probability exceeds the threshold τ is **committed**. The single most confident position is always committed, so every pass makes progress.
5. When the block is fully decoded, one **commit pass** writes its K/V into the cache and yields the next block's T0.

Higher τ commits fewer tokens per pass, which is slower but more accurate. τ = 1 means one token per pass.

### 4.2 Why speed is all about memory traffic (measured on the 4060, 256 GB/s)
- Every forward pass reads all ~3.09 GB of weights, so ~12 ms per pass is the floor.
- cuBLAS time for a full step barely changes with the number of tokens per pass: 1.02× at 8 tokens, 1.06× at 32 and 1.08× at 64, against 1 token. It only climbs from 128 (1.31×) and 256 (2.26×). **Up to ~64 tokens per pass is basically free.**
- A naive custom matmul kernel fell from 242 GB/s at 1 token to 24 GB/s at 32. That is why we keep cuBLAS.

### 4.3 Where the reference's 22 ms per pass goes

| Part | Time | Fix |
|---|---|---|
| Matrix multiplies (cuBLAS) | ~11.6 ms | none (floor) |
| Other GPU work in ~1,364 kernels | ~4.2 ms | our kernels and fusion |
| CPU launch gaps and syncs | ~6.5 ms | static cache + CUDA Graph |

What the other GPU work does wrong:
- It rebuilds a `[32, L+32]` boolean mask on the CPU every pass.
- It runs `torch.cat` over the whole KV cache in every layer, then `repeat_kv` (a 6× copy), then SDPA with an explicit mask. That is about 15× the necessary KV traffic.
- It runs a full-vocab bf16 softmax for 32 rows when only 8 are used.
- It forces 5 or more CPU-GPU syncs per pass.

Details with file:line references are in `research/fastdllm-v2-survey.md` §1.

### 4.4 Measured tokens per pass (reference, τ = 0.9)

| Prompt | Tokens per pass | Throughput |
|---|---|---|
| Math (GSM8K) | ~1.73–2.1 | 78 tok/s |
| Prose | ~1.2 | 53–56 tok/s, slower than llama.cpp's 64.6 |

GitHub issue #81 reports the same no-speedup result on short answers.

---

## 5. What we build

### 5.1 Driver (Python)
1. **Load with HF** (`trust_remote_code`, `transformers==4.53.x`) to get the weights and tokenizer. No custom loader, no custom model class.
2. **Our own layer loop** (~50 lines): `F.linear` (cuBLAS) plus our ops, reading the HF modules' weights. It is checked **layer by layer** against HF's own modules, so there is no "reproduce the whole model blind" problem.
3. **Our own decode loop**, matching `generate()` exactly: token shift, threshold, forced argmax, commit pass, EOS handling.
4. **Static KV cache.** Preallocated, written in place. Denoise passes overwrite slots `[L, L+32)`, and the commit pass is the same computation with the final tokens. The cache length L lives on the GPU.
5. **CUDA Graph** of one pass: fixed shapes, no CPU syncs inside. One small "done?" flag is read per pass.
6. **Stage 1 uses only stock PyTorch ops**, with a device-side mask in place of the CPU-built one, captured in a graph. That takes the risk out of graph capture early and is the first bar of the waterfall. Our kernels then replace ops one at a time through a per-op backend switch (`torch` | `ours`).
7. **AR mode:** the same driver with Qwen2.5 weights and block size 1.

### 5.2 CUDA kernels (C++/CUDA, built as a PyTorch extension)
Write each kernel straight away as the best version we can, rather than a sequential → OpenMP → CUDA progression. Each one gets:
- a unit test against the PyTorch op it replaces, with a stated tolerance
- a microbenchmark and an Nsight Compute profile (bandwidth vs roofline)
- a before/after ms-per-pass number in the driver

| | Kernel | Replaces |
|---|---|---|
| **K1** | **Parallel selection:** per row, argmax and Σexp over 151,936 logits (so `p_max = 1/Σ`, no full softmax), then threshold, forced argmax, scatter into the sequence, done/EOS flag, all on the GPU | full softmax, gather, `where`, boolean indexing, 5+ syncs |
| **K2** | **Block-causal attention:** understands grouped KV heads (each KV head serves 6 query heads), so K/V are loaded once; split along the KV length with a final reduction (bs=1 with 2 KV heads leaves the GPU idle otherwise); implicit mask; reads the static cache in place; L read from device memory; q_len ∈ {1, 32, 64} | `torch.cat` + `repeat_kv` + masked SDPA (~15× KV traffic) |
| **K3** | **Fused small ops:** RMSNorm + residual, RoPE + KV-cache write, SiLU·mul | most of the ~1,400 launches |

K2 is worth ~0.5 ms at L = 300, ~1.7 ms at 1k and ~3.4 ms at 2k (upper-bound estimates; nsys confirms). q_len = 1 is AR mode; 64 is for P2/P3 below.

**K1 gotcha:** the reference compares a **bf16-rounded** probability against τ. An fp32 kernel flips decisions at the margin unless it emulates that rounding. Test that the selected sets match the reference over many real passes.

### 5.3 New decoding strategies (RQ3). Measure the opportunity offline first

| | Strategy | Exact? | Expected gain | Build if |
|---|---|---|---|---|
| **P1** | Stop at the first fully decoded EOS and skip the final commit pass | exact | 3–6% of passes | always (easy) |
| **P2** | Fuse the commit pass with the next block's first denoise pass, using a **guessed** T0 (from the last denoise pass), verified afterwards | exact once verified | 3.5–6% of passes | the T0 guess is right ≥ ~70% of the time |
| **P3** | **Speculative step pipelining:** run step t and a guessed step t+1 in one 64-row pass; keep the second half if the guess was right | exact once verified | ~1.2–1.45× if the hit rate h is 0.3–0.6 | h ≳ 0.4 |
| **P4** | Approximate commit rules: a stricter τ₂ beyond the current sub-block, or committing a position whose guess held steady over 2 passes | changes output | 10–20% fewer passes for 1–2 points of accuracy | time allows |

The hit rates for P2 and P3 come from the trace dump (E3). **They are a report figure whether or not we build the feature.**

---

## 6. What we measure

| | Measurement | Where |
|---|---|---|
| **E0** | nsys / torch.profiler on one reference pass: GPU-busy vs wall time, top kernels by time. Sets how much the driver can recover. **Do first.** | 4060 |
| **E1** | cuBLAS knee: full-step time at M = 1…256 tokens per pass (`gemm_tm.cu` from the pilot already does this) | 4060, 3090, Euler |
| **E2** | GSM8K accuracy (below) | Euler |
| **E3** | Per-pass trace dump: commit set (position, token), confidences, guessed vs actual T0. Feeds P2–P4 analysis. | any GPU |
| **E4** | Final grid: {AR, dLLM} × {reference, our engine} at batch 1, plus llama.cpp; ablation waterfall; speed-vs-accuracy chart | 4060, 3090, Euler |

**GSM8K protocol:**
- **100 problems chosen with a fixed random seed**, not by hand, so the set can't drift easy or hard.
- Settings: AR, plus dLLM at τ ∈ {1.0, 0.95, 0.9, 0.8, 0.7, 0.5}.
- Batch size 1, because accuracy depends on batch size in the reference (issue #39: 63.0% at bs 64 vs 57.3% at bs 4).
- At n = 100 the 95% error bar is about ±10 points. Every setting runs on the **same** 100 problems, so we can report per-problem agreement (AR right / dLLM wrong counts), which says more than two separate percentages.
- Stretch goal: full 1,319 problems for the two headline settings (AR and dLLM τ = 0.9).
- Count tokens only up to EOS, for both models.
- Use the same chat template and answer extraction for both models.

**Measurement hygiene:**
- Laptop on the performance power profile. The pilot ran at a 45 W cap and saw memory drop to 6001 MHz under HF eager, plus 10–14% run-to-run variance.
- Log SM and memory clocks with every run.
- Warm up first and repeat each run.

---

## 7. Hardware

| GPU | Arch | Mem BW | Memory | Who / where |
|---|---|---|---|---|
| RTX 4060 Laptop | Ada sm_89 | 256 GB/s | 8 GB | Sparsh. The main edge target. |
| RTX 3090 | Ampere sm_86 | 936 GB/s | 24 GB | teammate (Windows + WSL2). The high-bandwidth contrast: its floor is ~3.3 ms, so overhead dominates there. |
| RTX 4000 SFF Ada | Ada sm_89 | 280 GB/s | 20 GB | **Euler `instruction` partition.** Same architecture and similar bandwidth to the 4060, so it doubles as the reproducible edge GPU and is the GPU for teammates without one. |

Euler facts, from `scripts/euler/probe_euler.sh` (job 122802):
- `-p instruction --gres=gpu:1`, **max 45 min per job**.
- Modules `nvidia/cuda/13.0.0` and `gcc/14.3.0`. Driver 580 (CUDA 13.0).
- System Python 3.12, no torch preinstalled. **Compute nodes have internet** (HF and pypi reachable).
- Home is CephFS; node-local NVMe at `/scratch`.
- Compiling on the login node is OK. **Running** anything there is not.
- Research/interactive partitions also have A100/H100/A4500. Ask the TA whether we may use them; don't plan around them.

GSM8K-100 on one Euler GPU, estimated from pilot per-problem times:

| Setting | Time |
|---|---|
| AR | ~10 min |
| dLLM τ = 1 | ~11 min |
| dLLM τ = 0.9 | ~5 min |

**Each setting fits in one job.** The full 1,319 needs about 4 shards for AR and about 2 for τ = 0.9.

**Pinned environment** (from the pilot):
- Python 3.12, `torch==2.14.1+cu130`, **`transformers==4.53.x`** (5.x breaks NVIDIA's remote code).
- Build kernels for `sm_86` and `sm_89`.

---

## 8. Not doing (and why)
- **Our own matmul:** cuBLAS is at the floor up to 64 tokens per pass, and our naive kernel collapsed (4.2).
- **The sequential/OpenMP/CUDA kernel progression:** we put the effort into the best kernels and new decoding strategies instead.
- **Sub-block cache (`use_block_cache`), dKV/dLLM/Sparse/d2/Elastic-Cache, token/layer skipping:**
  - they save arithmetic, which is free here
  - most are approximate
  - skipping makes outputs diverge
- **Multi-block concurrent decoding:** it needs retraining.
- **Tool-calling benchmarks:** Fast-dLLM v2 scored 0/30 on BFCL.
- **Standalone C++ runner; batch > 1; paged KV.**
- **Stretch only:** 8-bit weights. It is the biggest per-pass lever left, but it is not dLLM-specific.

---

## 9. Schedule

| Window | Goals |
|---|---|
| **11/17–11/23** | Commit pilot code as the baseline. E0 profile. E1 on all 3 GPUs. E3 trace dump and P2/P3 hit-rate analysis. Driver stage 1 (static cache + graph, stock ops). K1–K3 developed against saved real tensors. GSM8K-100 harness and the reference τ sweep on Euler. |
| **11/24–11/30** (Thanksgiving 11/26) | K1–K3 inside the driver; waterfall chart. AR mode in our engine. P1, P2. P3 build if h ≳ 0.4. |
| **12/1–12/2** | Final grid runs (E4), charts. Results frozen. |
| **12/3–12/4** | Slides, record video. |
| **12/5–12/7** | Report. |

Each row is a complete result on its own. If we stop early, the report still has a full story up to that point.

---

## 10. Risks

| Risk | Mitigation |
|---|---|
| CUDA Graph capture through our driver fights us (aliasing between replays, as in PR #51; cuBLAS picking different kernels) | Stage 1 de-risks it in week 1. Fallback: `torch.compile(mode="reduce-overhead")` with our ops registered through `torch.library`. |
| GPU-busy time is already ~20 ms, so graphs recover little | E0 tells us on day 1; shift effort to K2/K3 accordingly. |
| K1's selected tokens drift from the reference (bf16 rounding) | Compare selection sets on real traces; emulate the rounding. |
| The P3 hit rate is low | Report the measurement as a result and skip the build. |
| n = 100 too small to separate settings | Paired per-problem analysis; full set for the headline settings if time allows. |
| Windows teammates | WSL2 + CUDA works; if Nsight Compute doesn't work in WSL, profile on Euler. |

---

## 11. Next steps
1. **Interfaces and ownership** (next session): kernel contracts plus saved real tensors, the op registry, the trace format, the results JSONL schema, the Euler job template, the environment setup per OS. Every owner must be able to work unblocked.
2. Starter code / repo skeleton (separate session).
3. Sparsh: talk to the TA about the scope change, and whether the research partition is allowed.
