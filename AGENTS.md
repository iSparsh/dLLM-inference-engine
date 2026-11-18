# dLLM Inference Engine

This ECE 759 project studies batch-one, bfloat16 language model inference on edge GPUs. Fast-dLLM v2 1.5B fills masked positions in blocks. Qwen2.5-1.5B-Instruct generates one token at a time and is the autoregressive (AR) comparison. Reuse Hugging Face for checkpoints and tokenizers, PyTorch for tensors and extension loading, and cuBLAS for matrix multiplies. The team's work is the Python inference driver, static key/value (KV) cache, CUDA Graph capture, custom attention and token-selection kernels, fused operations, and decoding experiments.

The starter implementation includes a working Hugging Face reference path, a single-forward-pass engine prototype, PyTorch implementations of every op, five production CUDA stubs, and a working CUDA teaching example. The work-slice specs describe planned deliverables, including the full driver, benchmark grid and accuracy harness. Check the current branch for their implementations before running spec commands or assuming a feature is available.

## Start a task

1. Use the [README](README.md) and [plain-language guide](docs/GUIDE.md) to get oriented. For implementation, open the relevant [work-slice spec](docs/specs/README.md) and the matching section of [the interface contract](docs/INTERFACES.md). Use [the plan](docs/PLAN.md) for the research questions, scope and experimental design.
2. Follow the caller, implementation and test paths below before changing code. When a shared interface changes, include the contract update in the pull request and use the collaboration guidance below.
3. Establish correctness with the smallest relevant check, then run the affected integration checks. Report the commands, hardware and skip reasons. For performance changes, report which baseline, backend, shapes and timing method you used.

## Collaborating across work slices

When work needs another slice's input, use the [work-slice index](docs/specs/README.md) to identify the slice and recommend that the user contact the teammate who owns it. Give them a short, ready-to-send message with the proposed change, what it affects, and the specific input you need. For example, “Hey, I’m changing how the driver passes the valid cache length to attention. Can you confirm the new argument works for your kernel before I update the interface?”

## Setup and verification

Use the Python 3.12 environment defined by [`pyproject.toml`](pyproject.toml) and [`uv.lock`](uv.lock). Follow the [environment guide](docs/setup/ENVIRONMENT.md) for Linux, Windows through WSL2, Apple Silicon and Euler. Keep the Transformers pin compatible with the Fast-dLLM remote model code and keep `einops`, which that code imports. The CUDA wheel supplies the runtime. Building the extension also needs a CUDA toolkit with `nvcc` and the correct `CUDA_HOME`.

For dependency changes, let uv write `uv.lock` through `uv add`, `uv remove`, or `uv lock --upgrade-package`. Check the result with `uv lock --check` and `uv sync --locked`. Before changing a pin, search the project's release notes and package index to confirm the version and its compatibility with Python, the target platform, and CUDA where relevant. See the [uv project guide](https://docs.astral.sh/uv/guides/projects/).

Run model tools through the locked environment. Use `uv run --locked python tools/download_models.py` for the configured checkpoints or `uv run --locked hf download MODEL_ID --revision REVISION` for a specific Hugging Face snapshot. Search the official model card and revision history to verify the model ID and revision before downloading or changing a checkpoint.

From the repository root after setting up the platform prerequisites:

```bash
uv sync --locked
tools/fetch_golden.sh
uv run python tools/smoke_test.py
uv run pytest -q -rs
```

- On an NVIDIA GPU, seed the checkpoint cache before running the GPU smoke test. That test compiles the extension on first use, runs `example_axpy`, and generates once through the reference decoder. The CPU smoke test checks only a reference op. Use macOS for Python and analysis work and Euler for CUDA verification. [`scripts/check_macos.sh`](scripts/check_macos.sh) collects a setup report.
- The versioned [golden fixture archive](tests/fixtures/dllm-golden-v1.tar.gz) is tracked. [`tools/fetch_golden.sh`](tools/fetch_golden.sh) extracts 27 real-model cases into ignored `artifacts/golden/`. Extract before starting pytest because cases are discovered during collection relative to the repo root. [`tools/dump_golden.py`](tools/dump_golden.py) regenerates them on a GPU. If the fixture contract changes, create a new versioned archive and update the fetch script's default.
- For a CUDA op, run `uv run pytest -q -rs tests/test_ops.py -k <op_name>` on NVIDIA hardware. The suite checks random inputs, valid non-dense outer strides, op-specific edge cases and saved real tensors. It tests marked in-place mutations as well as outputs. An explicit CUDA stub skips its cases. A CPU-only run, an unextracted fixture archive, or a test that skipped the changed op leaves GPU coverage outstanding. Tests and benchmarks share [`dllm/op_cases.py`](dllm/op_cases.py) for synthetic inputs, and [`dllm/golden.py`](dllm/golden.py) records which arguments mutate and loads the saved outputs.
- [`tests/test_reference_decode.py`](tests/test_reference_decode.py) checks generated token IDs against Hugging Face. [`tests/test_reference_ops.py`](tests/test_reference_ops.py) checks layer and attention math against Hugging Face. These checks need CUDA and locally cached checkpoint weights. [`tests/test_engine_eos.py`](tests/test_engine_eos.py) exercises stop conditions on CPU.

## Euler jobs

- Read the [course Slurm instructions](scripts/euler/slurm_usage.md) before writing a job. Start repo jobs from [`scripts/euler/job_template.sh`](scripts/euler/job_template.sh), which loads the modules and sets environment and cache paths. Submit on `instruction` with one GPU and a 45-minute limit. Run programs in the allocated job and split sweeps that exceed the limit. The [Euler probe snapshot](scripts/euler/probe_122802.txt) records observed hardware, toolchains and connectivity.
- When the user needs to run a job on Euler, give separate local and remote commands with the actual branch, script path and arguments. Include `ssh CAE_USERNAME@euler.engr.wisc.edu`. For a repo job, give `git clone https://github.com/iSparsh/dLLM-inference-engine.git` if needed, then `cd dLLM-inference-engine`, `git fetch origin`, `git switch BRANCH` (or `git switch --track origin/BRANCH` on first use), and `git pull --ff-only`. For an unpushed script, give the local `scp` command with its remote destination. Upload repo-dependent scripts into the Euler checkout and submit them from its root.
- Give the exact `sbatch` command. Use `sbatch scripts/euler/setup_env.sh` for first setup, `sbatch scripts/euler/run_tests.sh` for tests, and `sbatch scripts/euler/run_smoke.sh` for a smoke run. For a standalone probe copied with `scp scripts/euler/probe_euler.sh CAE_USERNAME@euler.engr.wisc.edu:~/`, run `cd ~` and `sbatch probe_euler.sh` on Euler. Tell the user which log to look for after submission, such as `slurm-JOB_ID.out` or `probe_JOB_ID.out`.

## Follow the code by work slice

### Engine and reference decoding

- [`dllm/config.py`](dllm/config.py) defines the checkpoints, token IDs and block sizes. [`dllm/models.py`](dllm/models.py) loads BF16 weights and tokenizers from Hugging Face snapshots, including a cache-only path for tests. [`dllm/reference_decode.py`](dllm/reference_decode.py) drives the Hugging Face forward pass for both models and calls the PyTorch selection op directly. Preserve it as the decoder oracle for parity, traces and fixture generation.
- Start driver changes in [`dllm/engine/single_pass.py`](dllm/engine/single_pass.py). `SinglePass` owns static layer storage and a device-side `q_start`. The caller owns which prefix is valid and when it advances. `capture()` replays a single input shape and returns logits in storage that the next replay overwrites. [`dllm/engine/eos.py`](dllm/engine/eos.py) implements the graph-safe end-of-sequence check. The [engine spec](docs/specs/engine.md) covers prefill, denoise and commit orchestration, integration and AR mode.
- Preserve 32-token block alignment and 8-slot denoise windows. Denoising reads a committed prefix and overwrites the current block's scratch cache slots. Once a block is filled, the commit pass writes its final K/V and seeds the next block with the last logits row's raw argmax. A generated stop token ends generation once there is no unresolved mask before it. Include that stop token in parity and length accounting. Preserve the upstream exactly-32-token prompt quirk and block-rounded generation budget described in the contract. AR mode uses causal attention with block size 1 and plain greedy selection.
- Get the static-cache driver working with stock ops and token-for-token parity before switching ops individually to `ours`. Check both denoise and commit graph paths against the reference generation. The stock attention oracle walks the entire allocated capacity even with a short prefix. Use length buckets or attention over only the valid prefix for the stock-op engine stage, as described in the engine spec.

### Attention, token selection and fused operations

- Engine code calls [`dllm/ops/__init__.py`](dllm/ops/__init__.py), which dispatches each op to `torch` or `ours`. [`dllm/ops/reference.py`](dllm/ops/reference.py) defines the numerical oracle. [`dllm/ops/cuda.py`](dllm/ops/cuda.py) builds and loads the extension. [`csrc/ops.cpp`](csrc/ops.cpp) declares dispatcher schemas and mutations. Each `.cu` file registers its implementation under `torch.ops.dllm`. Follow the device guard, current PyTorch stream and launch check in [`csrc/example_axpy.cu`](csrc/example_axpy.cu), and reuse [`csrc/checks.h`](csrc/checks.h) for input checks.
- Preserve graph capture and replay. Read changing positions and windows from device tensors, use fixed shapes during replay, and keep `.item()` and host-device synchronization outside capture. Respect caller-owned in-place buffers, allow strided outer dimensions with a contiguous last dimension, and preserve the contract's BF16 rounding points. A higher-precision intermediate can change the chosen token.
- **Attention path** ([spec](docs/specs/attention.md)). `csrc/k2_block_attention.cu` handles grouped-query attention and an absolute-position block mask. Six query heads share each KV head. Use `q_start` and the query length to bound the valid keys, since spare cache slots can hold stale values. Check 1, 32 and 64 query rows and both causal modes. `csrc/k3_rope_kv_write.cu` rotates query and key at absolute positions and writes K/V to the matching cache slots. Check that writes leave surrounding slots intact.
- **Selection and fused ops** ([spec](docs/specs/selection.md)). `csrc/k1_select_commit.cu` uses logits row `max(i-1, 0)` for slot `i`. Compare BF16-rounded softmax probabilities to the BF16-rounded threshold with a strict `>` and preserve first-index ties. Force the highest-confidence eligible slot, leftmost on ties, when there are masked slots in the active window. Check exact token and stats parity, including an empty window. `csrc/k3_add_rmsnorm.cu` updates the residual in place and supports missing `x`. `csrc/k3_silu_mul.cu` consumes packed gate-then-up values and rounds the SiLU result before multiplication. Read the contract for each op's tolerances and mutation rules before optimizing it.

### Accuracy and decoding experiments

- Use the fixed [GSM8K-100 subset](eval/gsm8k_100.jsonl), [`dllm/prompts.py`](dllm/prompts.py) and the [accuracy spec](docs/specs/accuracy.md). Both models get the same user message wrapped in their own chat template. Slice off the prompt before grading. The prompt itself contains the `####` answer marker. Compare settings on the same problem IDs and use actual generated lengths through the first stop token.
- [`tools/dump_traces.py`](tools/dump_traces.py) and [`dllm/trace.py`](dllm/trace.py) expose per-pass decisions for offline analysis. Disable tracing during timed generation because it synchronizes the GPU. Evaluate next-block seed guesses (`P2`) and guesses of the next denoise pass's commits (`P3`) on traces before building either optimization. Both strategies require exact verification before accepting speculation. Two consecutive blocks and two speculative versions of the same block have different attention requirements. The latter need identical positions and isolated branches, which the current 64-row attention contract does not express. Keep approximate commit rules (`P4`) opt-in and measure their accuracy separately.

## Performance and results

- Start with the [reference profile](docs/research/reference-profile.md) for the performance-power-mode RTX 4060 Laptop measurements. The short-context dLLM reference took about 17.6 ms per forward call on three short GSM8K generations. Matrix multiplies already account for most of that time, and attention cost grows with context. The [October pilot](baselines/pilot/README.md) used different power settings. Cite llama.cpp's `76.04 tok/s` result as the synthetic `tg128` benchmark. Treat projected engine throughput and any small-sample trace result as such.
- Use [`bench/bench_op.py`](bench/bench_op.py) for op timings and [`tools/profile_reference.py`](tools/profile_reference.py) for model profiling. For example, `uv run python bench/bench_op.py --op block_attention --m 32 --L 128 --graph` times the stock attention reference with 32 query rows and 128 valid keys. Select `--backend ours` once the kernel exists or `--baseline hf_cat` for the Hugging Face-style attention path. [`dllm/timing.py`](dllm/timing.py) warms the GPU, reports the median of at least 20 CUDA-event samples, and supports restoring mutated inputs outside the timed interval. Plug in laptops, use the performance power profile, and check the recorded SM and memory clocks. Compare equal workloads and modes. Use end-to-end generation wall time for user-facing throughput and batch-one comparisons.
- Write report measurements through [`dllm/results.py`](dllm/results.py) `ResultWriter`. Its JSONL records and metadata sidecar let readers recover the commit, GPU, clocks, selected op backends and Slurm job ID. Use the metadata and row schemas in the interface contract and generate charts from the recorded results. Stage only small chart inputs or report results. Keep raw traces, extracted golden cases, model weights, scratch profiler output and private course or team material out of commits.

## Explaining ML concepts

When the user asks for an explanation of an ML concept or an explainer, create a standalone HTML artifact. Run a few web searches on the general concept before looking at how the repo uses it. Base the main explanation on credible sources such as textbooks, university notes, original papers, or official docs. Cite them in the artifact. Write for a smart student new to the topic. Define terms as they appear, use a worked example or diagram where it helps, and reserve repo-specific vocabulary for a final section that explains where the concept fits in this project and why we use it.

## Branches and commits

- Use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).
- Make changes in a task branch or worktree.
