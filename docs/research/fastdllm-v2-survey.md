# Fast-dLLM v2 (1.5B) at batch size 1: reference-code audit, literature survey, and ranked "wins"

Prepared 2026-11-17 for the ECE 759 dLLM team. Nothing was committed and no code was run (this sandbox has no `torch`).
Every number below is either quoted from a source (with a link) or is **my arithmetic from the numbers in the task brief**. Arithmetic is labelled *(est.)*.

## Evidence legend
- **[code]** – I read the file in full and cite file:line. Local copies of everything I read are in `./src/` next to this file, and the line numbers refer to those copies.
  - HF `modeling.py` = `src/modeling.py`, from the HF repo at revision `25093b6f63300adfd57f72145083c8a528fe4f16` (last modified 2026-01-22). Source: <https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_1.5B/blob/main/modeling.py>.
  - GitHub files are from `main`, fetched 2026-11-17. Repo: <https://github.com/NVlabs/Fast-dLLM>.
- **[paper]** – a claim from a paper or blog. Papers and blogs were read through a page-fetch summariser, so **re-check any figure/table number against the PDF before citing it in the report.**
- **[abstract]** – I only saw the abstract or a search snippet.
- *(est.)* – my own derivation.

---

## 0. TL;DR (read this first)

1. **Per-step cost at bs=1 is already close to the roofline on GPU time. The reference code just wraps it in a lot of overhead.**
   - The weight floor is 3.087 GB, which is 12.06 ms at 256 GB/s *(est.)*. The tied embedding / `lm_head` is 467 MB of that, or 15%, which is 1.8 ms.
   - A well-behaved cuBLAS stack should reach about 80–85% of peak bandwidth, so roughly 14–16 ms *(est.)*.
   - You measured 22 ms. So **CUDA graph + static cache + fusion can recover about 6–8 ms (about 1.4×) at most on the 4060**. After that, ms/step is stuck near the floor unless weight bytes shrink (quantisation, not dLLM-specific).
   - On the 3090 the floor is 3.3 ms *(est.)*, so the same 6–7 ms of CPU overhead is the *dominant* cost, and graphs are worth much more there.
2. **Almost all the diffusion-specific headroom is in passes per token, not ms per pass.**
   - Passes per block = S denoise steps + 1 commit pass (exact formulas in §1.2).
   - At your 2.1 tokens/step (math), a 32-token block costs about 14.8 + 1 passes, which is about 2.0 tokens/pass *(est.)*.
   - At 1.2 tokens/step (prose) it is about 1.19 tokens/pass.
   - AR at bs=1 pays the same ~12 ms weight read per token. So **on prose, dLLM is at parity with AR; on math it is about 1.9×**. Either way, tokens/pass is what the project is selling.
3. **Caching tricks that save FLOPs buy ~0 on a 256 GB/s part at batch 1.**
   - GEMM time is set by weight bytes. Going from 8 to 32 to 64 rows costs 1.2 / 4.9 / 9.9 ms of compute at an assumed 20 TFLOPS *(est.)*, all hidden under the 12 ms memory time. The knee is near M≈64–100.
   - This kills `use_block_cache`, dKV/dLLM/Sparse/d2/Elastic-Cache, and token/layer skipping for your hardware (§6).
4. **What the reference does badly that you can fix exactly**
   - Per step it has 5+ host syncs (§1.4).
   - It rebuilds an `[32, L+32]` bool mask on the CPU and copies it to the GPU every forward.
   - It does `torch.cat` of the whole cache per layer, then `repeat_kv` (6× materialisation), then SDPA with an explicit mask.
   - I estimate about **15× the KV bytes** of traffic per step (§1.3). That is about 1.7 ms at L=1000 and 3.4 ms at L=2000 on the 4060, versus 0.11 / 0.22 ms for a GQA-aware kernel.
5. **Ranked wins (details in §5):**
   - W1 graph + static in-place KV + fused selection (exact, the baseline plan).
   - W2 GQA-aware split-KV block-causal attention kernel (exact, matters at L ≥ 1k).
   - W3 stop at the first fully-decoded EOS and skip the commit pass (exact, easy).
   - W4 fuse the commit pass with the next block's first denoise pass, with a verified T0 guess (exact, small).
   - W5 speculative step pipelining, two denoise steps per weight read (exact with verification, research-grade, measure the hit rate offline first).
   - W6 spend the free compute on tokens/step (approximate, modest).
   - W7 W8 weight quantisation (approximate, biggest ms/step lever, not dLLM-specific).
6. **Reference-code facts to design against** (all [code]):
   - Defaults: `block_size=32`, `small_block_size=8`, `threshold=1` in `generate()` (i.e. parallel decoding OFF), `use_block_cache=False`.
   - `eval.py` defaults to `threshold=0.9`, but `eval_script.sh` overrides it to `threshold=1` for GSM8K, Minerva and IFEval.
   - **The published eval path never turns on `use_block_cache`.**
   - Each block runs S denoise passes plus one commit pass.
   - Confidence is a **bf16** softmax over the full vocab.
   - `lm_head` is applied to all 32 rows even though only 8 are used.

---

## 1. What the reference code does per denoising step (bs=1)

### 1.1 Three code paths
| Path | Where | Used by | Notes |
|---|---|---|---|
| `Fast_dLLM_QwenForCausalLM.generate()` | HF `modeling.py:652-831` | model card quick-start, bs=1 | Cleanest reference for bs=1. Stops early at EOS. |
| `batch_sample()` | GitHub `v2/generation_functions.py:16-168` | `v2/eval.py:86,253-277` (lm-eval harness, batched) | Per-row `finished_flag`, python loops over rows, `key_cache[...]` surgery (`:153-155`). Does **not** stop the block at EOS, it finishes all masks. |
| `mdm_sample_with_visualization()` | `v2/generation_functions.py:170-323` | `v2/app.py:73-125` (Gradio) | Always full recompute, no block cache option, `small_block_size=block_length` (`app.py:104`, so SB=32 by default in the demo). |

Defaults in `generate()` ([code] `modeling.py:653-668`): `threshold=1`, `small_block_size=8`, `block_size=32`, `top_p=0.95`, `temperature=0`, `use_block_cache=False`.
`generation_config.json` (temperature 0.7, top_p 0.8, rep. penalty 1.1) is **not** used by this loop. The HF model card calls `generate(..., small_block_size=8, threshold=0.9)` ([README of the 1.5B repo](https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_1.5B/blob/main/README.md)). The model config pins `transformers 4.53.1` (<https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_1.5B/blob/main/config.json>).

### 1.2 The loop, step by step ([code] `modeling.py` `generate()`)
1. **Prefill** (`:684-702`).
   - If `len(prompt) > 32`, one forward over `prompt[:, : L//32*32]` with `use_cache=True, update_past_key_values=True`.
   - If `L % 32 == 0`, `argmax(logits[:, -1])` is appended as one token (`:698-700`).
   - The remaining prompt tail (0–31 tokens) stays in `input_ids` and becomes the start of the first block.
   - *Edge case:* a prompt of exactly 32 tokens fails `32 > 32`, so there is no prefill, the block is built as 32 masks after the prompt (`:711-716`), and the forward sees only `x_t[:, -32:]`, which is masks only. The prompt is silently dropped. Avoid 32-token prompts in equivalence tests.
2. **Block start** (`:709-719`).
   - `x_init = [input_ids | MASK × (32 − len % 32)]`.
   - For every block after the first, `len % 32 == 1`, because the previous commit pass appended the AR-style next token. So **each later block is `[T0, MASK×31]`**, with T0 given for free by the previous commit pass.
3. **Denoise loop** (`:749-815`).
   - It loops over small blocks (SB=8, so 4 per block). Inside, `while masks remain in this small block:`
   - **Forward on all 32 block positions** (`:790-795`): `input_ids=x_t[:, -32:]`, `past_key_values=<committed prefix>`, `update_past_key_values=False`.
     - 32 query rows, K/V length L+32. The prefix cache is *not* modified.
     - `lm_head` runs on **all 32 rows**, because `logits_to_keep=0` gives `slice(-0, None)`, which is everything (`:636-637`). Only 8 rows are then used (`:798`).
   - **Token shift** (`:797`): `logits = cat([logits[:, :1], logits[:, :-1]])`. The prediction for position *i* comes from the output at *i−1*.
   - **Sampling** (`:805`, `:834-866`).
     - At `temperature=0`, `sample_with_top_p` does `softmax(logits)` over the full 151,936 vocab in the **input dtype (bf16)**, then `argmax`. It returns the full `p_1t` (`:838-841`). No sort and no top-p at T=0.
     - Confidence is `gather(p_1t, x_1)` (`:807`). At T=0 this equals the max probability.
   - **Selection** (`:808-815`):
     - `x1_p = where(mask_idx, conf, -inf)`.
     - `unmask = x1_p > threshold`, plus a forced `argmax` (guarantees progress).
     - `x_t[:, start:end][unmask] = x_1[unmask]`.
   - **Quirk to reproduce:** `x1_p` is bf16 and is compared to a Python float. I expect (not verified; no torch here) PyTorch compares in fp32, so the effective threshold on τ=0.9 is "the bf16-rounded probability is ≥ 0.9023", which is about p ≳ 0.9004. A fused fp32 kernel will flip decisions at the margin unless you emulate the bf16 rounding of the softmax output. **Write a unit test that diffs selection sets against the reference across many steps.**
4. **Commit pass** (`:729-747`), once all masks in the block are filled.
   - Forward on the final 32 tokens with `update_past_key_values=True`.
   - K/V are appended to the `DynamicCache` (a `torch.cat` per layer, `:226-228`).
   - `argmax(logits[:, -1])` gives T0 of the next block.
   - If EOS is found with no masks before it (`:722-725`), the loop breaks **before** the commit pass. So the last block has no commit pass.

**Pass count per block** (bs=1, SB=8): `S` denoise passes + 1 commit pass.
- Lower bound: S ≥ 4, because each of the 4 small blocks needs at least one step. That caps you at about 7.75 tokens/step at SB=8.
- Block yield: 32 tokens (T0 + 31 decoded).
- With your pilot: tokens/pass = 32 / (31/r + 1).
  - r = 2.1 gives 2.03.
  - r = 1.2 gives 1.19.
  - *(est.; I assume r counts denoise steps only, since your brief does not say.)*

### 1.3 What a forward pass costs in the reference
| Item | Evidence | Consequence |
|---|---|---|
| Mask is built **on CPU every forward**: `torch.arange` on CPU → `eval_block_diff_mask` → `.to(device)` | `modeling.py:424-432, 496` | Per forward, an H2D copy plus a bool `[32, L+32]` tensor (about 129k elements at L=4000). Mask is `block_q >= block_kv`, i.e. bidirectional inside a block, causal across blocks (`:89-94`). |
| Prefix cache is a HF `DynamicCache`; non-commit path does `torch.cat((past_k, k), -2)` and the same for V, **per layer, per step** | `:229-231` | Copies the whole cache every step. Shapes change every block, so no CUDA-graph capture. |
| Attention goes through HF `sdpa` with an explicit mask and `is_causal=False` | `:237-250` | With a mask, SDPA cannot use the flash backend. HF's `sdpa_attention_forward` (v4.53.1) always calls `repeat_kv` on K and V (`transformers/integrations/sdpa_attention.py:40-42`, [link](https://github.com/huggingface/transformers/blob/v4.53.1/src/transformers/integrations/sdpa_attention.py)). That is a 6× materialisation for 12 q-heads / 2 kv-heads. Newer transformers versions may differ (not checked). |
| RMSNorm upcasts to fp32 in eager PyTorch | `:266-271` | Many small kernels per layer. This is a large part of your ~1,400 kernels per step. |

KV traffic estimate *(est.)*:
- The cache is 28 layers × 2 (K,V) × 2 heads × 128 × 2 B = **28,672 B/token**.
- Per layer, per step: cat (read C + write C) + repeat (read C + write 6C) + SDPA read (up to 6C) ≈ **15C**.

| L (tokens) | cache | reference traffic (≈15×) | at 256 GB/s | single-pass GQA kernel (1×) |
|---|---|---|---|---|
| 300 | 8.6 MB | ~130 MB | 0.5 ms | 0.03 ms |
| 1000 | 28.7 MB | ~430 MB | 1.7 ms | 0.11 ms |
| 2000 | 57 MB | ~860 MB | 3.4 ms | 0.22 ms |
| 4000 | 115 MB | ~1.7 GB | 6.7 ms | 0.45 ms |

This is an upper bound, since L2 may dedupe some SDPA re-reads. **Measure it with nsys before promising it.**

### 1.4 Host–device syncs and CUDA-graph blockers ([code], `generate()` unless noted)
| Where | Op | Why it syncs / blocks capture |
|---|---|---|
| `:758` | `if mask_idx[:, start:end].sum() == 0` | tensor→bool, every inner iteration |
| `:760-763` | `stop_token in x_t[...]`, `.nonzero()[0][1]`, `.sum()==0` | `in` calls `.item()`; `nonzero` has a data-dependent shape |
| `:726-729` | `mask_idx.sum() == 0` | per outer iteration |
| `:766` | `(x_t[:, …] == mask_id).any()` (block-cache only) | per iteration |
| `:815` | `x_t[:, start:end][unmask_idx] = x_1[unmask_idx]` | boolean-mask gather and `index_put_`, which use `nonzero` |
| `:496` + `:424-432` | CPU mask build and `.to(device)` | host work and H2D every forward |
| `:230-231`, `:226-228` | `DynamicCache` + `torch.cat` | new allocations, changing shapes |
| `generation_functions.py:33,59,61,75-77,143-146` (`batch_sample`) | `.item()`, `.all()`, python loops over rows | many more syncs; irrelevant to bs=1 |

So the reference has at least 5 syncs per step. Because each sync drains the queue, the CPU launch cost of the next step cannot overlap with the GPU tail of the current one.

**Skeptical note.** A CPU that launches 1,400 kernels in 6.5 ms should normally run ahead of a 15 ms GPU step, so launch cost should largely overlap within a step. Your 22 ms versus a 12 ms floor therefore implies *either* many tiny kernels are launch-bound (gaps between them), *or* GPU time is itself about 20 ms (SDPA-with-mask kernels, fp32 norms, `repeat_kv`). **Run `nsys` first and compare GPU-busy time with wall time per step. Graphs only recover (wall − GPU-busy).**

After graphs, a per-step readback of "all done?" costs on the order of 0.05–0.1 ms *(est.)*, under 1% of a 15 ms step. So **do not over-engineer sync elimination** once the step is a graph. One tiny pinned-memory flag per step is fine.

### 1.5 Caching levels: exact vs approximate
| Level | What is reused | Exact? | Evidence | Default in eval? |
|---|---|---|---|---|
| **A. Prompt + previous blocks (`past_key_values`)** | K/V of committed tokens, written by the commit pass with the *final* tokens | **Exact** relative to recomputing the whole sequence with the same mask. Block q attends only blocks ≤ q (`:89-94`), so earlier K/V cannot depend on later tokens. (Not bit-exact: different GEMM shapes.) | [code]; training mask `M_BC` at `:83-84` | always on |
| **B. No cache inside the block** | nothing; all 32 positions recomputed each step | **Exact** | [code] `:790-798` | the default (`use_block_cache=False`) |
| **C. `use_block_cache` / "sub-block cache"** | K/V of the other 24 block positions from an earlier pass are reused and only the current 8 are overwritten (`replace_position`, `:211-222`, `:765-788`) | **Approximate (stale K/V)** | [code]; [issue #83](https://github.com/NVlabs/Fast-dLLM/issues/83) (open, unanswered) raises exactly this. S2D2 says sub-block caching "is not lossless here" and disables it for Fast-dLLM v2 [paper](https://arxiv.org/html/2603.25702). The v2 paper says it has "no observable effect on accuracy" (Sec 4.3, Fig 6a) [paper](https://arxiv.org/html/2509.26328) | **off**: `eval.py:59` `use_block_cache=False`; never set in `eval_script.sh` |
| **D. v1 PrefixCache / DualCache (LLaDA/Dream)** | K/V of prompt, and of the masked suffix, refreshed once per block | **Approximate** (bidirectional attention over the whole sequence has no exact cache) | [paper](https://arxiv.org/html/2505.22618) Sec 3.2; [code] `v1/llada/generate.py:238` (refresh = full forward per block), `:270` (block-only forward with `replace_position`) | n/a to v2 |

How the block cache works, in the code:
- The 8-token path is only taken when the first token of the current small block is already unmasked (`:766`).
- Reason: with the token shift, position `start` would need the logit from `start−1`, which the 8-token forward does not compute.
- Otherwise a full 32-token "refresh" pass runs.
- So even when enabled, each small block costs 1 full pass + k cheap passes.
- On a bandwidth-bound part the cheap pass costs the same as the full pass (§3).

### 1.6 Defaults the eval scripts actually use
- `eval.py` `__init__`: `use_block_cache=False`, `small_block_size=8`, `bd_size=32`, `threshold=0.9` (`eval.py:59-62`).
- `eval_script.sh` (`:7-29`):
  - MMLU and GPQA use bs 1 with default args. In MMLU only one letter is predicted, so the threshold has no effect ([issue #38](https://github.com/NVlabs/Fast-dLLM/issues/38)).
  - **GSM8K, Minerva and IFEval run with `threshold=1,show_speed=True`**, bs 32. That means one committed token per step per sample, i.e. no confidence-parallelism.
  - The v2 paper itself says threshold 1 is the default and 0.9 is used for the speed figures (Sec 4.1/4.2) [paper](https://arxiv.org/html/2509.26328).
- `app.py`: `threshold=0.9`, `top_p=0.9`, `small_block_size=block_length` (default 32).
- v1 `llada/eval_gsm8k.sh`: `block_length=32`, `threshold=0.9` or `factor=1.0`; dual-cache runs use `steps=length`.

---

## 2. GitHub issues and PRs worth knowing
List pulled with `gh` on 2026-11-17: 58 issues, 28 PRs. Links: `https://github.com/NVlabs/Fast-dLLM/issues/<n>` (PRs under `/pull/<n>`).

| # | State | What | Why it matters to us |
|---|---|---|---|
| [#81](https://github.com/NVlabs/Fast-dLLM/issues/81) | open, 0 replies | 1.5B at τ=0.9 shows **no speedup on short generations** (~20–30 tokens) vs Qwen2.5-1.5B | Same situation as your prose result. User's hypothesis: ≈1 token/step, heavier step. No maintainer answer. |
| [#83](https://github.com/NVlabs/Fast-dLLM/issues/83) | open | `use_block_cache` only overwrites 8 positions' K/V; others go stale. Approximation or bug? | Confirms level C is approximate; unanswered. |
| [#32](https://github.com/NVlabs/Fast-dLLM/issues/32), [#50](https://github.com/NVlabs/Fast-dLLM/issues/50) | closed / open | vLLM support for v2. Maintainer said "next month" on 2025-10-16; #50 (Dec 2025) points to vllm-omni | No vLLM path delivered in this repo. |
| [#39](https://github.com/NVlabs/Fast-dLLM/issues/39) | open | Accuracy depends on batch size. A commenter reports the official **1.5B** GSM8K = 63.0 at bs 64 vs 57.3 at bs 4 | Your bs=1 baseline may not match published accuracy. Run your own reference accuracy. |
| [#64](https://github.com/NVlabs/Fast-dLLM/issues/64) | open | `sample_with_top_p` samples only `p_1t[0]`, so bs>1 with T>0 breaks (confirmed at `modeling.py:864`) | Irrelevant at bs=1, but T>0 sampling is only valid for bs=1. |
| [#36](https://github.com/NVlabs/Fast-dLLM/issues/36) | open | "What is dual cache for in v2?" Commenters note a different N-mask padding changes when the model decides to stop | Stopping behaviour depends on how many masks follow, which affects any early-exit design. |
| [#52](https://github.com/NVlabs/Fast-dLLM/issues/52) | open | NVTX profile (LLaDA v1): dual-cache + parallel only ever shows "refresh", never "reuse" | Even v1's cache savings may be small in practice. |
| [#49](https://github.com/NVlabs/Fast-dLLM/issues/49), [#15](https://github.com/NVlabs/Fast-dLLM/issues/15) | open / closed | Reproduced tokens/s are lower than the paper's (81.5 vs 114.1 on an A800); throughput counting differences. [PR #16](https://github.com/NVlabs/Fast-dLLM/pull/16) (merged) unified token counting | Treat published tok/s with care. Count tokens up to EOS only, same as the AR twin. |
| [#7](https://github.com/NVlabs/Fast-dLLM/issues/7) | closed | v1 batch>1: changing the pad-token attention mask slows inference | Explicit masks defeat fast attention paths, same issue as §1.3. |
| [#19](https://github.com/NVlabs/Fast-dLLM/issues/19) | closed | Cache is refreshed after each block, otherwise performance degrades; maintainers confirm already-unmasked tokens are not updated | v1 detail. |
| [PR #26](https://github.com/NVlabs/Fast-dLLM/pull/26) | merged | `torch.compile`-compatible LLaDA v1 `generate.py`. Author claims ~3× if compiled (unverified) | Only v1 (LLaDA); nothing for the v2 model. |
| [PR #51](https://github.com/NVlabs/Fast-dLLM/pull/51) | open | torch-compilable RoPE for LLaDA; reports a CUDA-graph "output overwritten" error and **19% faster forward** (RTX 4090, 3K prompt / 512 gen) | Hits the same output-aliasing issue you will meet with graphs. |
| [PR #70](https://github.com/NVlabs/Fast-dLLM/pull/70) | closed, unmerged | Token-level and layer-level skipping on **v2** (cosine similarity between consecutive hidden states) | See §6: FLOPs −12–37% but outputs diverge heavily. |
| [PR #41](https://github.com/NVlabs/Fast-dLLM/pull/41) | closed | "reusing cached block activations", rewrite of v2 `generation_functions.py` | Could not evaluate (19 files, no description). |
| [PR #72](https://github.com/NVlabs/Fast-dLLM/pull/72) | closed | SmoothQuant / DuQuant experiments on LLaDA | Quantisation for dLLMs is only explored in closed community code. |
| [PR #74](https://github.com/NVlabs/Fast-dLLM/pull/74), [#76](https://github.com/NVlabs/Fast-dLLM/pull/76) | merged 2026-04/05 | SGLang support (vendored under `third_party/sglang`), `HierarchyBlock`, `SpeculativeBlock`, W8A8-FP8 for **Fast-dVLM** | `third_party/sglang/python/sglang/srt/models/fast_dllm_v2.py` exists too (610 lines), so an SGLang path for the v2 LLM exists in-repo. Not run. |

PR #70 data ([results.json](https://github.com/NVlabs/Fast-dLLM/pull/70), self-reported, v2 7B, one prompt set):

| Policy / threshold | FLOPs reduction | text overlap with baseline | BERT-F1 |
|---|---|---|---|
| token_skip 0.95 | 33.7% | 0.097 | 0.88 |
| token_skip 0.995 | 15.4% | 0.075 | 0.893 |
| layer_skip 0.95 | 16.0% | 0.022 | 0.897 |

The "overlap" metric is whatever that PR defines, and I did not read its definition. The take-away is that FLOP-saving skips change the output.

---

## 3. Roofline reasoning used throughout *(est.)*

| Quantity | Value | Source |
|---|---|---|
| params per layer | 46.8 M (q 1536², kv 2×1536×256, o 1536², MLP 3×1536×8960) | config [link](https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_1.5B/blob/main/config.json) |
| 28 layers | 1.310 B | matches model card "non-embedding 1.31B" |
| tied embedding / lm_head | 233.4 M params = **467 MB** | vocab 151936 × 1536 |
| total weights bf16 | **3.087 GB** | |
| floor @256 GB/s (4060 Laptop) | **12.06 ms** (lm_head alone 1.82 ms) | your brief |
| floor @936 GB/s (3090) | 3.30 ms (lm_head 0.50 ms) | public spec, from memory |
| FLOPs per forward, M rows | 2 × 1.5436e9 × M = 98.8 GFLOP at M=32 | |
| compute time at an *assumed* 20 TFLOPS (4060) | M=8: 1.2 ms, M=32: 4.9 ms, M=64: 9.9 ms, M=128: 19.8 ms | **the 20 TFLOPS is an assumption, not measured** |

- Arithmetic intensity of a weight GEMM is about M FLOP/B, so the ridge is around M ≈ 80–115 on both GPUs *(est.)*. M = 8 vs 32 is flat and M = 64 is near the knee.
- **Action:** microbenchmark `M ∈ {8,16,32,64,96,128}` on both GPUs. It is a 20-line script and it settles half of the argument below.
- On the 3090 (82 SMs) the flat region is wider. But compute at 32 rows is 1.4–3 ms, which is comparable to the 3.3 ms memory floor, so the 3090 is closer to balanced and kernel overhead dominates.
- v2's own bs=1 number on an A100 (7B): **102.5 tokens/s**, rising to 217.5 at bs=4 ([project page](https://nvlabs.github.io/Fast-dLLM/v2/) [paper]). The A100 weight floor for 7B bf16 is about 7.5 ms, so an AR ceiling of ~130 tok/s *(est.)*. Context for the "2.5×" headline: it is measured against an AR baseline in a particular framework, and the headline setting is not stated in text I could access.

---

## 4. What the papers say

### Fast-dLLM v2 ([arXiv 2509.26328](https://arxiv.org/abs/2509.26328)) [paper]
- Block size 32, sub-block 8. Threshold default is 1 (disabled); speed experiments use 0.9 (Sec 4.1/4.2).
- Hardware: A100 and H100. Fig 4 (GSM8K, τ=0.9): 39.1 → 101.7 tok/s (2.6×; model size not stated in the text I saw). Fig 5 (7B vs Qwen2.5-7B-Instruct, with sub-block cache): up to 1.5× on A100 at bs 64 and up to 1.8× on H100. Fig 1(b) compares bs 1 and 4. The Fig 1(a) headline is 2.54× vs Qwen2.5-7B-Instruct.
- Ablations: sub-block size (Table 3) GSM8K 62.8 (SB 2) → 60.2 (SB 32); HumanEval peaks at SB 8 (43.9). Block-size mismatch (Table 4) GSM8K 62.0 → 58.5 at mismatched inference block size.
- Not reported: NFE / tokens-per-forward, a limitations section, whether HF or vLLM was the framework, and any consumer-GPU or bs=1 breakdown beyond the project page.

### Fast-dLLM v1 ([arXiv 2505.22618](https://arxiv.org/abs/2505.22618)) [paper]
- A100 80GB, single batch, "no inference-acceleration frameworks" (Fig 1 caption).
- Threshold τ=0.9 default; factor strategy (Sec 3.3). Cache refreshed each block, stale-KV cosine-similarity argument (Fig 3).
- LLaDA GSM8K-256: 6.7 tok/s → 54.4 with cache+parallel (8.1×); cache alone 3.2×, parallel alone 2.5× (Table 1).
- Stated limitations: bidirectional attention "precludes a fully equivalent KV cache"; at larger batch sizes PrefixCache no longer keeps up with LLaMA as the latter turns compute-bound (Sec 4.3, App C.5); factor decoding costs 1–3% accuracy.

---

## 5. Candidate wins, ranked

Ranking = expected benefit on your bandwidth-bound laptop, confidence in that benefit, and whether it fits in about 2 weeks. "Δ" is vs the reference's 22 ms/pass, 2.1 (math) / 1.2 (prose) tokens/step.

### W1. Whole-pass CUDA graph + static in-place KV + fused selection (exact) — **foundation, do first**
- **dLLM-specific parts:**
  - Denoise passes never commit, so with a static cache **write the 32 block K/V slots in place** at `[L, L+32)` every pass. They are simply overwritten next pass. The commit pass is the *same graph* with the final tokens. Then `L += 32`. This removes `cat` and `DynamicCache` and gives fixed shapes.
  - The mask is pure arithmetic on `(q, kv, L)` and needs no tensor.
  - Selection (bf16-emulated confidence, threshold, forced argmax, scatter into `x_t`, "small-block done?" flag) lives on the device.
  - Confidence at T=0 needs only `(argmax, Σexp(l−l_max))`, so `p_max = 1/Σ`. No full softmax tensor is needed, just a reduction over 151,936 values × 32 rows (about 9.7 MB read).
  - Make `L` a device scalar, or bucket graphs by L rounded to 256.
- **Parallelism/CUDA work:** a fused argmax+sumexp reduction kernel (or fold it into the lm_head epilogue). Fused RMSNorm+residual, RoPE+KV-write, SiLU·mul. cuBLAS for the GEMMs (per TA feedback).
- **Effect:** 22 → about 14.5–16 ms on the 4060 *(est.)*, i.e. −27–34% ms/step, or 1.4–1.5×. Graphs remove CPU launch cost but **not** the per-kernel ramp/tail gap of about 1–3 µs, which is why fusion still matters: 1,400 kernels × a couple of µs is 2–4 ms of bubbles *(est.)*. Larger relative gain on the 3090 (floor 3.3 ms).
- **Exact?** Yes up to bf16 rounding. The confidence is bf16-rounded in the reference; emulate it (§1.2).
- **Difficulty:** medium. **Risks:** output-aliasing between graph replays (see PR #51); cuBLAS kernel selection can change with M and between eager and graph; the selection-parity test above.

### W2. GQA-aware, split-KV block-causal attention kernel (exact)
- **dLLM-specific:** the 32 queries all share the same K/V. With 6 q-heads per KV head that is **192 query rows per KV head** against L+32 keys, so it is a small GEMM that tensor cores can handle, unlike AR decode (6 rows per KV head). With bs=1 and only 2 KV heads, one CTA per head leaves SMs idle (24 on the 4060 Laptop, 82 on the 3090), so **split along the KV length (flash-decoding style) and reduce**. Mask is implicit: block-causal with a bidirectional 32×32 tail.
- **Effect:** removes the cat + `repeat_kv` + explicit-mask path (≈15× → 1× KV traffic): ~0.5 ms at L=300, 1.7 ms at L=1000, 3.4 ms at L=2000, 6.7 ms at L=4000 *(est., upper bound)*. For chat prompts under about 500 tokens it is a small win; it matters for long contexts and RAG-style prompts.
- **Exact?** Yes (fp32 accumulate, online softmax; compare with SDPA within tolerance).
- **Difficulty:** medium-hard. **Risks:** 32-row query tile with head_dim 128 needs careful smem use on 8 GB / Ada; numerics vs SDPA.

### W3. Stop on the first fully-decoded EOS; skip the final commit pass (exact, easy)
- **dLLM-specific:** the eval path (`batch_sample`) keeps decoding every small block after an EOS has been committed. `generate()` already breaks as soon as no masks remain before the first EOS (`:722-725`) and skips the commit pass.
- **Effect:** on average about half a block (up to 31 positions of steps), so roughly 3–6% of passes on a 256-token answer *(est.)*. dInfer reports 15–40% efficiency gain from early termination at 1024-token generation length [paper](https://arxiv.org/html/2510.08666).
- **Exact?** Yes for the text up to EOS. **Difficulty:** easy.
- **Risk:** *comparison hygiene.* Count only tokens up to EOS for both dLLM and AR, otherwise the speedup is an artifact.

### W4. Fuse the commit pass with the next block's first denoise pass (exact when the guess hits)
- **Idea:** the commit pass produces T0 (the next block's first token). Instead of waiting, guess T0 from the last denoise pass's logits at position 31, run one 64-row pass `[block k final tokens | T0_guess, MASK×31]`, and accept it if the commit-pass `argmax` equals the guess. Block k+1's rows attend to block k's K/V computed in the same pass, which is exact because they are final tokens.
- **dLLM-specific:** needs the block-causal structure, and the commit pass exists only because the K/V of the final tokens must be recomputed.
- **Effect:** saves about one of S+1 passes per block: about 6% on math (S ≈ 15), about 3.5% on prose (S ≈ 26) *(est., at a hit rate near 90% which I have not measured)*. The 64-row pass may cost +0–2 ms near the ridge on the 4060.
- **Exact?** Yes, with verification. **Difficulty:** medium (the kernel needs to take two blocks of query rows; this falls out of W2 if it takes `n_blocks` as a parameter).
- **First step:** log guess-vs-actual T0 over 50 prompts; kill the idea if the hit rate is under about 70%.

### W5. Speculative step pipelining: two denoise steps per weight read (exact when verified; research item)
- **Idea:** memory-bound means 64 rows cost about the same as 32. Run `[state_t | state_{t+1}^guess]` in one pass, where the guess is the commit set predicted from step t−1's confident positions. If the actual selection at step t equals the guess (same positions *and* same tokens), the second half is exactly step t+1's output, so you advanced two steps for one weight read. A miss discards the second half.
- **dLLM-specific:** AR has no step chain to speculate on this way. Closest relatives are Jacobi/lookahead decoding.
- **Effect:** about (1+h)/(1+c), with h the hit rate and c the extra cost of 64 vs 32 rows. At h = 0.3–0.6 and c ≈ 0.1, that is ≈ 1.2–1.45× *(est.)*, and it subsumes W4. h is unknown and could be much lower, because commit sets vary in size (1 to 8).
- **Exact?** Yes semantically; not bitwise (different GEMM shapes). **Difficulty:** hard.
- **First step (cheap):** log per-step commit sets from the reference over a few dozen prompts and compute h offline. Only build it if h ≳ 0.4. Run this measurement even if you do not build the feature; it is a good report figure.

### W6. Spend the free compute on tokens/step (approximate)
- **Ideas** (all just selection-kernel logic once W1 exists):
  - (a) Commit confident tokens beyond the current sub-block with a stricter τ₂. The reference already computes logits for all 32 rows and throws 24 of them away.
  - (b) Stability commit: commit a position whose argmax was unchanged over the last 2 steps. Related: [Prophet](https://arxiv.org/abs/2508.19982), which commits early using the top-2 confidence gap (reported ≤3.4× fewer steps on LLaDA/Dream, **not tested on block-diffusion models**) [abstract].
  - (c) Per-position or adaptive thresholds.
- **Evidence it is modest:**
  - S2D2's Table 2 on Fast-dLLM v2 (H100): SB 8 → SB 32 moves GSM8K speedup 3.3× → 3.5× while accuracy drops 83.2 → 78.3 [paper](https://arxiv.org/html/2603.25702).
  - dInfer's hierarchical/credit decoding changes tokens-per-forward by about +6%/+17% in one table and −19%/−12% in another [paper](https://arxiv.org/html/2510.08666).
  - So expect ≤10–20% fewer passes for ≥1–2 points of accuracy.
- **Exact?** No, changes outputs. **Difficulty:** easy to build, expensive to evaluate: use a 200-sample GSM8K subset (what S2D2 used), at your own bs=1 baseline.

### W7. Weight-only 8-bit (FP8 on Ada, INT8 on Ampere), maybe lm_head too (approximate, not dLLM-specific)
- **Effect:** halves weight bytes, so the floor goes 12.1 → about 6–7 ms on the 4060 *(est.)*. This is the largest per-pass lever available on a bandwidth-bound GPU.
- **Caveat:** Fast-dVLM's W8A8-FP8 gave 319 → 350 tok/s (+10%) on top of SGLang in their table, so do not expect the full 2× [paper](https://github.com/NVlabs/Fast-dLLM/blob/main/fast_dvlm/README.md). Quantisation also perturbs the confidences that drive the thresholds, so re-measure tokens/step.
- **Difficulty:** medium (a dequant-in-GEMM kernel is real work; cuBLAS FP8 on Ada needs scaling plumbing). Treat as a stretch tier.

### Not dLLM-specific but required for a fair comparison
Give the AR twin the same treatment (static cache, graph, fused kernels). Otherwise "dLLM vs AR" is mostly "optimised vs unoptimised".

---

## 6. Things NOT worth doing (and why)

| Idea | Why not |
|---|---|
| **`use_block_cache` / 8-token forwards** | Approximate (stale K/V; S2D2 disables it as lossy, issue #83 unanswered). Saves FLOPs only: M=8 vs 32 is 1.2 vs 4.9 ms of compute under a 12 ms memory time. Never used by the published eval script. |
| **dKV-Cache, dLLM-Cache, Sparse-dLLM, d2Cache, Elastic-Cache, v1 DualCache** | Built for full-sequence bidirectional dLLMs (LLaDA/Dream). Fast-dLLM v2's block-causal mask already gives an exact prefix cache. What remains (in-block, bidirectional) is approximate and compute-only. Sparse-dLLM evicts KV; at L ≤ 2k, KV bytes are ≤ 0.5–1% of weight bytes per pass (57 MB vs 3.1 GB at L=2000), so there is nothing to save. [dKV](https://arxiv.org/abs/2505.15781), [dLLM-Cache](https://arxiv.org/abs/2506.06295), [Sparse-dLLM](https://arxiv.org/abs/2508.02558), [d2Cache](https://arxiv.org/abs/2509.23094), [Elastic-Cache](https://arxiv.org/abs/2510.14973) [abstract only]. |
| **Token-skip / layer-skip (PR #70)** | FLOP-only, and the outputs diverge (§2 table). |
| **Multi-block concurrent decoding** | Requires post-training (Multi-block Teacher Forcing) [abstract](https://arxiv.org/abs/2606.29215). v2's training mask never shows a noisy previous block as context. |
| **S2D2 / Fast-dVLM self-speculation as-is** | S2D2 at bs=1 on H100 is *slower* on GSM8K (342 → 310 tok/s) and only helps at bs 32/64 [paper](https://arxiv.org/html/2603.25702). Fast-dVLM's 2.63 tokens/NFE comes from a model trained for the causal mode [paper](https://github.com/NVlabs/Fast-dLLM/blob/main/fast_dvlm/README.md). One extra verification pass per round also eats most of the gain on a ~12 ms floor. |
| **Vocabulary pruning of `lm_head`** | The 1.8 ms (15%) is tempting, but a shortlist changes the softmax denominator, hence the confidences that drive the thresholds. No exact bound. |
| **Replacing cuBLAS GEMMs** | Per TA feedback, matching cuBLAS is hard, and GEMMs are bandwidth-bound anyway. Put your custom CUDA into attention, norms/RoPE/SwiGLU fusion and selection. |
| **Custom prefill attention** | Prefill is GEMM-dominated (about 3.1 GFLOP/token); attention is a small share at L ≤ 2k. Reference quirks don't matter for TTFT here. |
| **Batch>1, continuous batching, paged KV, TP/EP** | Out of scope for bs=1 on 8 GB. (dInfer and JetEngine numbers come from 8×H800 / bs=128 and do not transfer.) |
| **llama.cpp as a target** | Its diffusion example documents Dream, LLaDA and RND1 with block-length scheduling; I found no Fast-dLLM v2 / Qwen block-diffusion architecture listed ([README](https://github.com/ggml-org/llama.cpp/blob/master/examples/diffusion/README.md)). I did not read the sampler loop. |
| **Comparing against the paper's 2.5× / dInfer 10× / SGLang numbers** | Different models (7B, LLaDA-MoE, LLaDA2-flash), GPUs (A100/H100/H800/H20), batch sizes and frameworks. At best, cite them as context. |

---

## 7. Related-work table (for the report)
"BW-bound bs=1" = would it help your 256 GB/s case?

| Work | Idea (2 lines) | Exact? | Targets | Helps BW-bound bs=1? | Applies to block-diffusion v2? |
|---|---|---|---|---|---|
| Fast-dLLM v1 [paper](https://arxiv.org/abs/2505.22618) | Block-wise PrefixCache/DualCache with per-block refresh; threshold (τ=0.9) and factor parallel decoding | Cache: approx. Threshold: approx. (guarantee only under a theorem's assumptions) | Compute (cache); step count (parallel) | Parallel decoding: yes. Cache: no | Parallel decoding already in v2. Cache superseded by block-causal exact prefix cache. |
| dKV-Cache [abstract](https://arxiv.org/abs/2505.15781) | "Delayed" per-token KV caching, tokens are cached one step after they are decoded | Approx. ("almost lossless" Decode, lossier Greedy) | Compute | No | Targets full-attention dLLMs. |
| dLLM-Cache [abstract](https://arxiv.org/abs/2506.06295) | Long-interval prompt cache + short-interval partial response update by feature similarity | Approx. | Compute (up to 9.1× FLOPs) | No | Prompt is already exact-cached in v2. |
| Sparse-dLLM [abstract](https://arxiv.org/abs/2508.02558) | Attention-guided KV eviction + sparse attention | Approx. | KV memory / attention compute | Only at long L | Not shown for block-diffusion. |
| d²Cache [abstract](https://arxiv.org/abs/2509.23094) | Two-stage token selection decides which KV to refresh | Approx. | Compute | No | LLaDA/Dream only. |
| Elastic-Cache [abstract](https://arxiv.org/abs/2510.14973) | Attention-drift test decides when/where (from which layer) to refresh KV | Approx. | Compute | No | LLaDA family. |
| FreeCache / FlashDLM [search](https://arxiv.org/html/2505.21467v2) | Freeze KV of finalised blocks; guide diffusion with a small AR model | Approx. | Compute; step count (guider) | Guider only | The "freeze finalised block" part is what v2's block-causal mask gives exactly. |
| dInfer [paper](https://arxiv.org/html/2510.08666) | Modular engine: hierarchical/credit decoding, iteration smoothing, vicinity KV-refresh (all approximate); torch.compile + CUDA graphs, kernel fusion, loop unrolling (+5–10%), early termination (+15–40%), host-sync removal (all exact) | mixed | Both | **System part: yes.** Algorithms: modest | System ideas transfer. Measured on LLaDA-MoE, bs=1, 8×H800, block 64. |
| SGLang dLLM [blog](https://www.lmsys.org/blog/2025-12-19-diffusion-llm/) | Reuses chunked prefill, block-diffusion attention (bidirectional within a block), CUDA graph, `LowConfidence` algorithm | exact | system | Partly | Blog says Fast-dLLM v2 is "planned"; the NVlabs repo later vendored SGLang with `HierarchyBlock` ([hierarchy_block.py](https://github.com/NVlabs/Fast-dLLM/blob/main/third_party/sglang/python/sglang/srt/dllm/algorithm/hierarchy_block.py)). Numbers: H20 TP8 LLaDA2-flash, claims 1.9× vs AR at small batch. |
| S2D2 [paper](https://arxiv.org/html/2603.25702) | Self-speculative: draft with block diffusion, verify with the same model in block-size-1 AR mode, routed to avoid wasted verify passes | Approx. (hybrid trajectory, not equal to AR) | Step count | Marginal at bs=1 | Yes (Fast-dLLM v2 is a listed target; the plain causal mask suffices since it is right-shifted). |
| Fast-dVLM `SpeculativeBlock` [code](https://github.com/NVlabs/Fast-dLLM/blob/main/third_party/sglang/python/sglang/srt/dllm/algorithm/speculative_block.py) | Draft with bidirectional mask, verify with a causal mask in the same CUDA graph, accept the longest matching prefix | exact w.r.t. the AR mode | Step count | One extra pass per round | Needs a model trained for the causal mode. Reports 2.63 tokens/NFE on the VLM. |
| Prophet [abstract](https://arxiv.org/abs/2508.19982) | Commit all remaining tokens early when the top-2 gap is large | Approx. | Step count | Yes (fewer passes) | Evaluated on LLaDA / Dream only. |
| Multi-Block DLMs [abstract](https://arxiv.org/abs/2606.29215) | Decode several blocks concurrently; TPF 3.47 → 6.19 | Approx., needs post-training | Step count | Yes | Needs training; not v2. |
| SDAR / JetEngine [search](https://arxiv.org/pdf/2510.06303) | Block-diffusion engine, CUDA graphs for bs 1..128, FA2/FA3+Triton | n/a | System | Partly | Design reference only: bs=128 numbers on H200/A800. |

---

## 8. Do these first (days 1–3, before writing kernels)
1. `nsys` one reference step on the 4060. Report GPU-busy vs wall per step, plus top kernels by time. This bounds what W1 can deliver.
2. Microbenchmark the cuBLAS GEMMs at `M ∈ {8,16,32,64,96,128}` for the 5 matrix shapes (and lm_head) on the 4060 and the 3090. This settles where the knee is, and so how "free" W4/W5 are.
3. Instrument the reference to dump, per step, the commit set `(pos, token)`, the per-row confidence, and the guess-vs-actual T0. Compute: tokens/step distribution, W4 hit rate, W5 hit rate h, and how often τ=0.9 vs a bf16-aware compare would disagree.
4. Pin `transformers==4.53.1` and PyTorch for the reference. The reference code indexes `DynamicCache` internals (`generation_functions.py:153-155`, `modeling.py:216,229-231`), and newer transformers may change this (unchecked).
5. Run your own reference accuracy (GSM8K subset) at bs=1. Published accuracy is batch-dependent ([issue #39](https://github.com/NVlabs/Fast-dLLM/issues/39)).

## 9. Known uncertainties in this document
- I could not run anything. All ms/step figures are derived from your brief, not measured.
- Paper numbers come via a summariser; check against PDFs before citing.
- I did not verify the bf16 comparison semantics (§1.2) or the SDPA kernel actually chosen for the masked path (§1.3).
- SGLang `HierarchyBlock` uses a **causal** (`DECODER`) attention for the final KV-write forward (`hierarchy_block.py:149`), while the HF reference's commit pass uses the block-bidirectional mask (`modeling.py:89-94, 496`). I did not resolve whether this difference is intentional. **Treat the HF code as ground truth** for your own equivalence tests.
- Hardware facts for the 3090 (936 GB/s, 82 SMs) and the 4060 Laptop (24 SMs) are from memory; verify with `nvidia-smi`/`deviceQuery`.
