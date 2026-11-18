# Pilot baseline (2026-10-30)

The first measurements on the RTX 4060 Laptop, before the proposal. They give the numbers in
`docs/research/pilot-measurements.md` and the "before" picture for the project.

Setup: Python 3.12, `torch 2.14.1+cu130`, `transformers==4.53.3` (5.x breaks the NVIDIA remote
code), CUDA 13.4, bf16, batch size 1, greedy decoding. The laptop was on the "balanced" power
profile (45 W cap), so rerun on the performance profile before comparing.

| File | What it measures | How it was run |
|---|---|---|
| `cuda/bw_launch.cu` | DRAM bandwidth, kernel launch cost, launch cost inside a CUDA Graph | `nvcc -O3 -arch=sm_89 bw_launch.cu -o bw_launch && ./bw_launch` |
| `cuda/gemm_tm.cu` | Step time of all Qwen2.5-1.5B matmuls at M tokens per pass: cuBLAS vs a naive kernel | `nvcc -O3 -arch=sm_89 gemm_tm.cu -lcublas -o gemm_tm && ./gemm_tm` |
| `cpu/gemm_cpu.cpp` | The same on CPU (OpenMP + AVX2) | `g++ -O3 -march=native -fopenmp gemm_cpu.cpp -o gemm_cpu`, then `OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores ./gemm_cpu` |
| `bench/common.py` | Model loading and a forward-call counter | imported by the others |
| `bench/bench_ar.py` | HF eager AR baseline (Qwen2.5-1.5B-Instruct) | `python bench_ar.py` |
| `bench/bench_dllm.py` | NVIDIA reference Fast-dLLM v2 sweep over threshold, sub-block size and block cache | `python bench_dllm.py` |
| `bench/quality.py` | Small GSM8K (20) and BFCL (30) accuracy probe | `python quality.py 30 20 both` |
| `results/` | Raw outputs of all of the above, plus `llama-bench` output | |

`quality.py` reads datasets from `../data/` (GSM8K test parquet, BFCL v4 simple_python). They are
not committed; download them from Hugging Face (`openai/gsm8k`) and the BFCL repo.

The llama.cpp numbers used a CUDA build (`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89`) and
Qwen2.5-1.5B-Instruct converted to bf16 GGUF, run as
`llama-bench -ngl 99 -fa 1 -p 128 -n 128 -b 2048 -ub 512 -r 3`.

This is reference code. New work goes in the main source tree, not here.
