# Pilot measurements (2026-10-30, RTX 4060 Laptop)

Recovered from the pilot session. Scripts and raw results: `~/Documents/ece759-plan-tests/` on Sparsh's laptop (to be committed under `baselines/pilot/`).

## 2. Measurements

Hardware for all of these: RTX 4060 Laptop GPU, 8 GB GDDR6, 128-bit bus, 24 SMs, 32 MB L2, theoretical 256 GB/s. CPU is an i7-13700HX (AVX2, AVX-VNNI, no AVX-512; 16 cores / 24 threads). CUDA 13.4 (nvcc 13.4.92), driver 615.78. torch 2.14.1+cu130, **transformers pinned to 4.53.3** (5.x breaks the NVlabs remote code), bf16. The enforced GPU power limit was **45 W** under the "balanced" platform profile, not 80 W.

Everything below is batch 1, greedy, single GPU. All numbers come from `transcript-1.md` and from `~/Documents/ece759-plan-tests/results/`.

### 2a. GPU microbenchmarks (`cuda/bw_launch.cu`, `cuda/gemm_tm.cu`, built with `nvcc -O3 -arch=sm_89`)

Two runs exist: one in-chat at about 14:35 and one saved to `results/gpu_microbench.txt` at 15:03.

| quantity | run 1 | run 2 (saved file) |
|---|---|---|
| read-only stream, 1 GiB | 218.5 GB/s | 250.3 GB/s |
| copy, 1 GiB | 201.7 | 231.4 |
| cudaMemcpy D2D, 1 GiB | 198.3 | 226.5 |
| launch cost (336 empty kernels, stream) | 2.20 µs/launch | 2.04 µs/launch |
| same kernels in a CUDA Graph | 0.49 µs/kernel | 0.49 µs/kernel |

So 336 launches cost about 0.7 ms in a stream and about 0.17 ms in a CUDA Graph.

`gemm_tm`: all 7 linears × 28 layers of Qwen2.5-1.5B (hidden 1536, inter 8960, kv_dim 256, vocab 151936), bf16 weights rotated through 2.6 GB to defeat the L2. Linear weights are 3.087 GB per step (layers 2.620 + lm_head 0.467), so the bandwidth floor is 12.06 ms/step at 256 GB/s.

cuBLAS step time T(M), M = tokens per step:

| M | run 1 (ms) | run 2 (ms) | run 2 × T(1) |
|---|---|---|---|
| 1 | 11.14 | 10.94 | 1.00 |
| 2 | 11.55 | 11.29 | 1.03 |
| 4 | 11.70 | 11.31 | 1.03 |
| 8 | 11.44 | 11.21 | 1.02 |
| 16 | 11.67 | 11.27 | 1.03 |
| 32 | 11.99 | 11.59 | 1.06 |
| 64 | 12.24 | 11.82 | 1.08 |
| 128 | 19.13 | 14.30 | 1.31 |
| 256 | 32.20 | 24.74 | 2.26 |
| 512 | 56.99 | 46.20 | 4.22 |

The lm_head alone takes about 1.87 ms at M=1 and 1.98 ms at M=32.

Naive custom weight-streaming kernel (one warp per output row), run 2 vs run 1:

| M | run 2 (ms) | run 2 (GB/s) | run 1 (ms) |
|---|---|---|---|
| 1 | 10.83 | 242 | 10.95 |
| 8 | 14.3 | 183 | 19.66 |
| 16 | 62.6 | 42 | 65.6 |
| 32 | 110.5 | 24 | 116.9 |

Conclusion drawn in chat: break-even on GPU is about 1.06 to 1.08 tokens/step, and a tensor-core `mma.sync` kernel is needed beyond M ≈ 8.

### 2b. CPU microbenchmark (`cpu/gemm_cpu.cpp`)

OpenMP over N with an AVX2+FMA inner loop. bf16 weights, fp32 activations. Run with `OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores`, and `g++ -O3 -march=native -fopenmp`.

| M | step ms (run 1, in-chat) | × T(1) (run 1) | step ms (saved file) | × T(1) (saved file) |
|---|---|---|---|---|
| 1 | 127.6 | 1.00 | 66.6 | 1.00 |
| 2 | 134.4 | 1.05 | 73.4 | 1.10 |
| 4 | 159.9 | 1.25 | 86.6 | 1.30 |
| 8 | 324.3 | 2.54 | 229.8 | 3.45 |
| 16 | 465.7 | 3.65 | 420.2 | 6.31 |
| 32 | 834.6 | 6.54 | 809.3 | 12.15 |
| 64 | 1736.6 | 13.61 | 1682.1 | 25.25 |

Measured DRAM stream was 53.5 GB/s in run 1 and 48.3 GB/s in the saved file. The saved-file M=1 time is faster by about 2× (run-to-run variance; the audit text quotes run 1). Chat conclusion: on CPU the dLLM needs at least 6.5 tokens/step to tie AR, delivers about 2, and loses by about 3×.

### 2c. llama.cpp (strong AR baseline)

- Built from a depth-1 clone: `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DGGML_NATIVE=ON -DLLAMA_CURL=OFF`, targets `llama-bench llama-cli llama-simple`. Build `792fcf4`.
- Model: Qwen2.5-1.5B-Instruct converted with `convert_hf_to_gguf.py --outtype bf16` and `--outtype f16` (both 3.09 GB). These were **not quantized**.
- Command: `llama-bench -ngl 99 -fa 1 -p 128 -n 128 -b 2048 -ub 512 -r 3`
- **bf16: tg128 = 64.62 ± 0.04 tok/s** (15.5 ms/token, about 199 GB/s effective, about 78% of theoretical). pp128 = 4764 ± 277 tok/s.
- f16: tg128 64.26 ± 0.08. pp128 5198 ± 432.
- CPU only (`-ngl 0 -t 16`): pp64 171.8 tok/s, **tg32 7.82 ± 0.06 tok/s** (about 128 ms/token).

### 2d. HF AR baseline (`bench/bench_ar.py`), Qwen2.5-1.5B-Instruct, bf16, transformers 4.53.3, 256 new tokens, greedy

Eager:

| prompt | tok/s | ms/token | prefill |
|---|---|---|---|
| gsm8k (98-token prompt) | 47.0 | 21.3 | 29.5 ms |
| intro (29-token prompt) | 49.3 | 20.3 | 20.3 ms |

`torch.compile` (`mode=reduce-overhead`, static cache) was unstable: 40.6 tok/s (24.6 ms/token) on the intro prompt, 2.5 tok/s (399.5 ms/token) on the gsm8k prompt because of recompiles. The author set it aside as untuned.

### 2e. Fast-dLLM v2 1.5B reference (`bench/bench_dllm.py`, NVlabs remote code)

- Model: `Efficient-Large-Model/Fast_dLLM_v2_1.5B`, loaded with `trust_remote_code=True`.
- 1.5437 B params, 3.088 GB in bf16. 28 layers, hidden 1536, 12 Q / 2 KV heads, MLP 8960, vocab 151936, tied embeddings, `bd_size=32`, mask id 151665.
- Generation call: `model.generate(ids, tokenizer=tok, max_new_tokens=256, small_block_size=sb, threshold=thr, use_block_cache=bc)`. `M` = the query-token count of each `forward` call (via a `ForwardCounter` wrapper).
- Two single prompts: "gsm8k" (Janet's ducks, with "####" instruction) and "intro" ("short introduction to large language models"). The chat template was the default system prompt. One warm-up run was done first.
- Columns: thr = confidence threshold τ, sb = sub-block size, bcache = `use_block_cache`, new = new tokens, fwd = forward calls, tok/fwd = tokens per step.

| prompt | thr | sb | bcache | new | fwd | tok/fwd | ms/fwd | tok/s | M histogram |
|---|---|---|---|---|---|---|---|---|---|
| gsm8k | 1.0 | 8 | F | 144 | 145 | 0.99 | 23.3 | 42.6 | {32:144, 96:1} |
| gsm8k | 0.95 | 8 | F | 144 | 88 | 1.64 | 22.0 | 74.5 | {32:87, 96:1} |
| gsm8k | **0.9** | 8 | F | 144 | 83 | **1.73** | 22.0 | **78.7** | {32:82, 96:1} |
| gsm8k | 0.7 | 8 | F | 150 | 67 | 2.24 | 22.6 | 99.0 | {32:66, 96:1} |
| gsm8k | 0.5 | 8 | F | 130 | 48 | 2.71 | 22.1 | 122.8 | {32:47, 96:1} |
| gsm8k | 1.0 | 8 | T | 144 | 145 | 0.99 | 22.3 | 44.6 | {8:115, 32:29, 96:1} |
| gsm8k | 0.95 | 8 | T | 144 | 90 | 1.60 | 22.2 | 72.2 | {8:65, 32:24, 96:1} |
| gsm8k | 0.9 | 8 | T | 144 | 83 | 1.73 | 22.4 | 77.6 | {8:58, 32:24, 96:1} |
| gsm8k | 0.7 | 8 | T | 183 | 86 | 2.13 | 22.6 | 94.2 | {8:56, 32:29, 96:1} |
| gsm8k | 0.5 | 8 | T | 126 | 42 | 3.00 | 22.7 | 132.3 | {8:22, 32:19, 96:1} |
| gsm8k | 0.9 | 4 | T | 157 | 96 | 1.64 | 23.4 | 69.9 | {4:48, 32:47, 96:1} |
| gsm8k | 0.9 | 16 | T | 154 | 80 | 1.93 | 23.2 | 82.9 | {16:65, 32:14, 96:1} |
| gsm8k | 0.9 | 32 | T | 154 | 74 | 2.08 | 22.0 | 94.5 | {32:73, 96:1} |
| intro | 1.0 | 8 | F | 228 | 228 | 1.00 | 22.6 | 44.3 | {32:228} |
| intro | 0.95 | 8 | F | 228 | 205 | 1.11 | 22.6 | 49.2 | {32:205} |
| intro | 0.9 | 8 | F | 228 | 181 | 1.26 | 22.4 | 56.3 | {32:181} |
| intro | 0.7 | 8 | F | 228 | 163 | 1.40 | 23.5 | 59.6 | {32:163} |
| intro | 0.5 | 8 | F | 211 | 118 | 1.79 | 22.6 | 79.2 | {32:118} |
| intro | 1.0 | 8 | T | 228 | 228 | 1.00 | 22.8 | 43.9 | {8:171, 32:57} |
| intro | 0.95 | 8 | T | 228 | 202 | 1.13 | 22.3 | 50.7 | {8:150, 32:52} |
| intro | 0.9 | 8 | T | 228 | 194 | 1.18 | 22.3 | 52.6 | {8:145, 32:49} |
| intro | 0.7 | 8 | T | 228 | 161 | 1.42 | 23.4 | 60.4 | {8:115, 32:46} |
| intro | 0.5 | 8 | T | 133 | 74 | 1.80 | 25.2 | 71.4 | {8:51, 32:23} |
| intro | 0.9 | 4 | T | 228 | 198 | 1.15 | 22.6 | 50.9 | {4:116, 32:82} |
| intro | 0.9 | 16 | T | 228 | 192 | 1.19 | 22.8 | 52.2 | {16:169, 32:23} |
| intro | 0.9 | 32 | T | 105 | 88 | 1.19 | 23.9 | 49.8 | {32:88} |

Per-run raw data (every run's text and timings): `recovered/ece759-plan-tests/results/dllm_timing.json`.

Observations stated in chat:
- **About 22 to 23 ms per forward, regardless of M.** The block cache gives no speedup at batch 1, which matches T(8) ≈ T(32).
- At τ = 0.5 the gsm8k text degrades ("egg egg egg $2", "9 * 9 = $18").
- Default `use_block_cache=False` means every denoising step is a full 32-token forward. With the cache on, the first step of each 8-token sub-block is M=32 and later steps are M=8. Each finished block costs one extra full-block forward, which also yields the first token of the next block. The prompt remainder (`len % 32`) shares the first block, so `max_new_tokens=64` produced only 34 tokens in the smoke test.
- The HF `generate()` default is `threshold=1` (no parallel decoding). The paper's Table 1 accuracies are at threshold 1.0.

### 2f. Kernel counts and GPU-busy time (`torch.profiler`, one forward, eager bf16; saved in `results/kernel_counts.txt`)

| forward | kernels | GPU-busy | wall time in generate loop |
|---|---|---|---|
| Fast-dLLM v2 forward, M=32, no prefix cache | 1364 | 15.8 ms | about 22.5 ms |
| Fast-dLLM v2 forward, M=32, 96-token prefix cache | 1420 | 16.2 ms | |
| Qwen2.5 HF AR decode, M=1, 96-token cache | 1196 | 14.4 ms | about 21 ms |

That leaves about 6 to 7 ms per step as CPU-side launch gaps. Under HF-eager decode the GPU also downclocked to P4, SM 465 MHz, **memory 6001 MHz** (not 8001), 34% utilization and 17 W.

### 2g. Quality probe (`bench/quality.py`, `python quality.py 30 20 both`, saved in `results/quality.log` and `quality.json`)

Settings:
- Dataset: BFCL v4 `simple_python`, first 30 items, with Hermes `<tool_call>` template tools, max 128 new tokens.
- Dataset: GSM8K test (parquet), first 20 items. Prompt: question + "Think step by step, then give the final numeric answer after '####'". Max 320 new tokens.
- dLLM settings: `small_block_size=8`, `use_block_cache=True`.
- Scoring: GSM8K exact match on the number after `####` (with a fallback to the last number). BFCL: JSON-valid `<tool_call>`, name match, args match against possible answers (simplified AST).

| model | task | n | result | s/item | tok/fwd |
|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct (HF, greedy) | BFCL-simple | 30 | JSON-valid 100%, name 100%, AST 77% | 0.67 | 1.00 |
| same | GSM8K | 20 | EM 55%, 273 tok/item | 6.05 | 1.00 |
| Fast-dLLM v2 τ=1.0 | BFCL | 30 | JSON-valid 3%, name 0%, AST 0% | 3.00 | 0.99 |
| same | GSM8K | 20 | EM 45%, 236 tok/item | 6.41 | 0.99 |
| Fast-dLLM v2 τ=0.9 | BFCL | 30 | 3% / 0% / 0% | 1.60 | 1.59 |
| same | GSM8K | 20 | EM 40%, 232 tok/item | 2.61 | 2.14 |
| Fast-dLLM v2 τ=0.7 | BFCL | 30 | 3% / 0% / 0% | 1.26 | 2.01 |
| same | GSM8K | 20 | EM 40%, 237 tok/item | 2.16 | 2.65 |

n = 20 has about ±11 pt error. The paper's full-set numbers quoted in chat: Qwen2.5-1.5B 57.0, Fast-dLLM v2 1.5B 62.0 (τ=1). The Fast-dLLM v2 model never emits tool calls (answers in prose or Python), at any τ.

### 2h. Other numbers from the chats

- **Paper (Fast-dLLM v2, arXiv 2509.26328):** 2.54× throughput over Qwen2.5-7B-Instruct on A100 (39.1 → 101.7 tok/s across τ = 1 → 0.9). About 1.5× on A100 and 1.8× on H100 at larger batch sizes. Needs about 1B fine-tuning tokens. The audit infers the AR baseline was about 40 tok/s ≈ HF-eager speed for 7B on A100.
- **Plan prediction in the audit** for a same-engine dLLM at τ=0.9: about 85 tok/s on prose and about 125 to 150 tok/s on GSM8K, versus about 73 tok/s for AR. Floor: 12.06 ms/step at 256 GB/s, 12.9 to 13.7 ms at achievable bandwidth, so an AR ceiling of about 73 to 78 tok/s.
- **NVML:** `nvmlDeviceGetTotalEnergyConsumption` works on this laptop GPU (34.7 J in 1 s under load).
- **Context from the Oct 15 research chats:**
  - llama.cpp's diffusion example has no KV cache and does CPU-side per-position sampling.
  - On an A100 the host sampling takes 18 to 35% of a step (discussion #22972 in llama.cpp).
  - The n-engine fork: block cache alone 1.02×, but feeding only the active chunk 5.93×.
  - DiffusionGemma on an RTX 5090: AR 2.3 to 3.5× faster than diffusion in llama.cpp. The AR twin ran at about 248 tok/s versus diffusion 72 to 108 tok/s.
  - The CPU diffusion discussion #20796 reported 7.4× vs 2.4× thread scaling on a 12-core EPYC.


### Why the reference dLLM code is slow (transcript-1)

- Memory-bound: at M ≤ 64, cuBLAS T(M)/T(1) is only 1.02 to 1.10, so a 32-token step costs about as much as a 1-token step. Diffusion wins on the GPU once tokens/step exceeds about 1.07.
- The reference implementation does not get anywhere near that floor.
  - It runs at 22 to 23 ms/step against a floor of 12 to 13.7 ms.
  - It launches about 1,364 to 1,420 kernels per step with only 15.8 to 16.2 ms GPU-busy, so about 6 to 7 ms is CPU launch gaps from Python and eager PyTorch.
  - The HF AR path has the same problem: 1,196 kernels and 14.4 ms busy against about 21 ms wall.
  - llama.cpp reaches 15.5 ms/token on AR, so there is at most about 1.15× left on the AR side, but about 1.6 to 1.7× left on the dLLM side.
- Launch cost per kernel (about 2 µs) is not itself the issue. The plan's "1 to 1.7 ms per step" was about 2× too high. The CPU gaps and the number of kernels are the issue. CUDA Graphs (0.49 µs/kernel) and fused kernels (about 350 launches estimated for about 12 kernels per layer) are what remove them.
- The block cache does nothing at batch 1 (22.3 ms vs 22.0 ms) because T(8) ≈ T(32).
- Actual tokens/step at τ=0.9 are modest: 1.2 on open prose, 1.7 to 2.1 on GSM8K. That is why the reference dLLM beats llama.cpp AR (64.6 tok/s) only on math (78 tok/s) and loses on prose (about 53 to 56).
- The paper's 2.5× was measured against a weak (HF-eager) AR baseline on a 7B model on an A100.
- A naive weight-streaming kernel ties cuBLAS at M=1 but collapses to 24 GB/s at M=32, so Tier 1 needs tensor-core tiling.
- The NVlabs code needs `transformers==4.53.x`, and `datasets` clashes with the pinned `huggingface_hub`.
- "70% of step time is sampling" (arXiv 2601.20706) is an FP64-softmax artifact of LLaDA's reference sampler (<10% with BF16).
- Qwen-style `<tool_call>` prompting works on Qwen2.5 but not on the Fast-dLLM v2 SFT, so BFCL is unusable for this model (the Bitter Lesson paper shows FdLLM-7B at 1.2 / 0 / 0). GSM8K should be primary and HumanEval/MBPP secondary.


## 6. Caveats

- All numbers are on a single laptop under the "balanced" 45 W profile and with downclocking under HF-eager. Run-to-run variance is visible: the GPU microbenchmarks ran 10 to 14% faster in the second run, and the CPU M=1 time was 128 ms in one run and 67 ms in the other.
- The quality probe is n=20 GSM8K and n=30 BFCL, so it is a sanity check only.
- The dLLM timing table uses one gsm8k prompt (about 144 to 183 generated tokens) and one prose prompt, not a dataset. It was not repeated.
- `transcript-*.md` are verbatim reconstructions. They include system prompt, reasoning blocks and large tool outputs, so they are big (about 0.1 to 0.5 MB each).
- Dates are CDT (UTC-5) from the in-chat timestamps. Cursor's `createdAt` is epoch ms in UTC.
