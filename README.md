# dLLM Inference Engine

ECE 759 (Fall 2026) final project: fast inference for a diffusion language model (Fast-dLLM v2 1.5B) on edge GPUs, compared against its autoregressive twin (Qwen2.5-1.5B-Instruct) running on llama.cpp.

Team: Sparsh Mishra, Munasib Ilham, M Sadman Sakib, Ayesha Shafique.

## Start here

1. **[docs/GUIDE.md](docs/GUIDE.md)** is the plain-language walkthrough. Read this first if the project is new to you.
2. **[docs/explainer.html](docs/explainer.html)** is the visual companion. Download it and open it in a browser.
3. **[docs/PLAN.md](docs/PLAN.md)** is the detailed plan: research questions, what we build, what we measure, schedule.

## Layout

| Path | What |
|---|---|
| `docs/` | Plan, guide, explainer, research notes |
| `docs/research/` | Pilot measurements and the audit of the NVIDIA reference code |
| `baselines/pilot/` | October pilot benchmarks (reference numbers; not the engine) |
| `scripts/euler/` | Euler cluster scripts |
