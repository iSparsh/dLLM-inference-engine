# Reference profile (2026-11-17)

Where NVIDIA's reference code spends its time per pass, and how fast the baselines really are, on the RTX 4060 Laptop on the **performance** power profile (P0, 2520 MHz SM, 8001 MHz memory, ~70 W). Reproduce with `tools/profile_reference.py`.

## End to end, no profiler

`uv run python tools/profile_reference.py --model dllm --prompts 3 --tau 0.9` (and `--model ar`), GSM8K-100 prompts, 64-token budget, warmed up. "Outside" is wall time per call minus CUDA-event time of the forward: the decode loop's own cost.

| Model | Prompt | New tokens | Forward calls | ms per call | Forward ms | Outside ms | tok/s |
|---|---|---:|---:|---:|---:|---:|---:|
| dLLM τ=0.9 | gsm8k-0005 | 54 | 45 | 17.52 | 17.26 | 0.26 | 68.5 |
| dLLM τ=0.9 | gsm8k-0017 | 52 | 30 | 17.75 | 17.44 | 0.31 | 97.7 |
| dLLM τ=0.9 | gsm8k-0021 | 64 | 32 | 17.71 | 17.41 | 0.30 | 113.0 |
| AR (HF) | gsm8k-0005 | 64 | 64 | 15.65 | 15.42 | 0.23 | 63.9 |
| AR (HF) | gsm8k-0017 | 64 | 64 | 15.96 | 15.67 | 0.28 | 62.7 |
| AR (HF) | gsm8k-0021 | 64 | 64 | 15.85 | 15.58 | 0.27 | 63.1 |

(These AR rows used the checkpoint's default repetition penalty of 1.1; we have since standardized on plain greedy, which doesn't change the per-token cost.)

## Inside one dLLM pass, with torch.profiler

Three consecutive post-prefill passes of gsm8k-0005 (context ≈ 130 tokens). The profiler itself inflates wall time, so use the table above for throughput.

| Category | ms per pass |
|---|---:|
| Matrix multiplies (cuBLAS/cutlass) | 13.76 |
| Norms, RoPE, SiLU, adds and other elementwise | 1.43 |
| KV-cache concatenation and other copies | 0.74 |
| Attention (memory-efficient SDPA) | 0.23 |
| Softmax/selection | 0.04 |
| **GPU busy (union)** | **16.21** |
| Kernels per pass | 1,447 |

The AR model's decode step: 14.39 ms GPU busy, 1,231 kernels, 12.72 ms of matrix-vector products.

## Baselines

| Measurement | Result |
|---|---|
| llama.cpp `llama-bench`, Qwen2.5-1.5B-Instruct bf16 GGUF, `-ngl 99 -fa 1 -p 128 -n 128 -r 3` | **tg128 76.04 ± 0.02 tok/s** (13.15 ms per token); pp128 6087 tok/s |
| cuBLAS, all matmuls of one pass incl. LM head (`baselines/pilot/cuda/gemm_tm.cu`) | M=1: 12.81 ms · M=32: 13.57 · M=64: 13.87 · M=128: 16.50 |
| Own layer loop, one 32-row pass, CUDA Graph vs eager vs HF (96-token prefix, capacity 160) | 16.11 ms graph · 17.68 own eager · 17.25 HF eager |

## Attention by context length

`bench/bench_op.py --op block_attention --m 32 --L <L> --graph` with the warm-up in `dllm.timing.benchmark`; ms per layer × 28 layers = ms per pass.

| Context L | HF path (cat + repeat_kv + masked SDPA) | SDPA on prepared inputs, no mask | Our PyTorch reference (full 4096-slot cache) |
|---:|---:|---:|---:|
| 128 | 0.88 | 0.32 | 9.05 |
| 1024 | 3.09 | 0.69 | 9.06 |
| 4096 | 10.58 | 1.95 | 9.09 |

The reference op always works over the whole 4096-slot cache, so it is a correctness reference, not something to run in the engine at that capacity.

## What this changed

- The October pilot's 22 ms per pass was measured on the "balanced" (45 W) profile. On "performance", the reference takes 17.6 ms.
- The decode loop outside the forward costs ~0.3 ms per pass, not the ~6.5 ms we had attributed to CPU gaps. The overhead is inside the forward: ~1,450 kernel launches and the gaps between them, small elementwise kernels, and KV copies.
- Matrix multiplies are already at the cuBLAS floor, so the headroom at short context is about 17.6 → 14.5 ms. At longer contexts the attention path grows quickly, which is where K2 pays off.
- Short microbenchmarks must warm the GPU up first: an idle laptop GPU runs at 210 MHz SM / 405 MHz memory, which made our first kernel timings 3–10× too slow.
