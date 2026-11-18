# dLLM Inference Engine

ECE 759 (Fall 2026) final project: fast inference for a diffusion language model (Fast-dLLM v2 1.5B) on edge GPUs, compared against its autoregressive twin (Qwen2.5-1.5B-Instruct) running on llama.cpp.

Team: Sparsh Mishra, Munasib Ilham, M Sadman Sakib, Ayesha Shafique.

## Start here

1. **[docs/GUIDE.md](docs/GUIDE.md)** is the plain-language walkthrough. Read this first if the project is new to you.
2. **[docs/explainer.html](docs/explainer.html)** is the visual companion. Download it and open it in a browser.
3. **[docs/PLAN.md](docs/PLAN.md)** is the detailed plan: research questions, what we build, what we measure, schedule.
4. **[docs/setup/ENVIRONMENT.md](docs/setup/ENVIRONMENT.md)** sets up your machine (Linux, Windows via WSL2, macOS) and Euler.
5. **[docs/specs/](docs/specs/README.md)** has the four work slices, written like assignments, and **[docs/INTERFACES.md](docs/INTERFACES.md)** the contracts they code against.

## Layout

| Path | What |
|---|---|
| `dllm/` | The Python package: models, prompts, reference decode loops, op registry, engine |
| `csrc/` | Our CUDA kernels (one file per op) |
| `tests/`, `bench/`, `tools/`, `eval/` | Tests, benchmarks, utilities, the GSM8K-100 problem list |
| `docs/` | Plan, guide, explainer, interfaces, specs, setup, research notes |
| `baselines/pilot/` | October pilot benchmarks (reference numbers; not the engine) |
| `scripts/euler/` | Euler cluster jobs; `scripts/check_macos.sh` checks a Mac setup |

Quick start (Linux or WSL2 with an NVIDIA GPU): `uv sync && uv run python tools/smoke_test.py && uv run pytest -q`. Details in ENVIRONMENT.md.
