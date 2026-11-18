# Environment Setup

Everyone sets up two things: **a local environment** (for writing code, running tests and analysis) and **Euler** (for GPU runs that everyone can reproduce). Pick your local section by operating system, then do the Euler section.

| Your machine | What you can do locally | What goes to Euler |
|---|---|---|
| Linux + NVIDIA GPU | everything | reproducible runs, long sweeps |
| Windows + NVIDIA GPU (through WSL2) | everything | reproducible runs, long sweeps |
| macOS (Apple Silicon, macOS 14 or newer) | all Python code, tests on CPU, analysis, charts | anything that needs CUDA: kernels, model runs, benchmarks |

Everything below was tested on 2026-11-17 on Linux (RTX 4060 Laptop, driver 620, CUDA 13.4) with a fresh clone. The macOS dependency set was resolved but not run on a Mac; the Euler scripts were syntax-checked but not yet run on Euler. If a step fails for you, post the error in the team chat and we'll fix this file.

**Pinned versions** (in `pyproject.toml` and `uv.lock`; don't change them without telling the team): Python 3.12, `torch==2.14.1` (the CUDA 13.0 build on Linux), `transformers==4.53.3` (5.x breaks the dLLM's model code), `huggingface_hub==0.36.2`, `einops==0.8.2` (the dLLM's model code needs it).

---

## Linux with an NVIDIA GPU

You need:
- An NVIDIA driver that supports CUDA 13.0 (driver 580 or newer). Check with `nvidia-smi`: the top right shows the highest CUDA version the driver supports.
- The CUDA toolkit, version 13.x, for `nvcc` (it compiles our kernels). 13.0 matches Euler exactly; any 13.x works, and a newer minor version only prints a harmless version-mismatch warning. Check with `nvcc --version`.
- `git` and a C++ compiler (`gcc`).

Then:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh        # installs uv (the Python package manager); open a new shell after
git clone https://github.com/iSparsh/dLLM-inference-engine.git
cd dLLM-inference-engine
uv sync                                                 # creates .venv with the pinned versions (~3 GB, mostly torch)
export CUDA_HOME=/usr/local/cuda                        # the folder that contains bin/nvcc; put this in your ~/.bashrc
uv run python tools/download_models.py                  # both models, ~3.1 GB each, into ~/.cache/huggingface
uv run python tools/smoke_test.py                       # builds our CUDA ops (1-2 min the first time) and runs the dLLM once
uv run pytest -q                                        # most tests skip until the kernels are written; none should fail
```

`smoke_test.py` should end with a line like `example_axpy passed; generated tokens=..., passes=..., ms/pass=...`.

## Windows: use WSL2

Our kernels build with Linux tools, so on Windows we work inside WSL2 (a real Linux running on Windows, with direct access to your NVIDIA GPU).

1. **Windows side.** Install the latest NVIDIA driver for your GPU from nvidia.com (Game Ready or Studio; it must support CUDA 13.0, i.e. version 580 or newer). This one driver serves both Windows and WSL2. **Do not install any NVIDIA driver inside Linux.**

2. **Install WSL2 with Ubuntu.** In PowerShell as administrator:
   ```powershell
   wsl --install -d Ubuntu-24.04
   ```
   Reboot if asked, open "Ubuntu" from the Start menu, and create your Linux user.

3. **Check the GPU from Ubuntu:** `nvidia-smi` should list your GPU. (It comes from the Windows driver.)

4. **Install the CUDA toolkit for WSL** (toolkit only; this package does not include a driver, which is what you want):
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt-get update
   sudo apt-get -y install cuda-toolkit-13-0 build-essential git
   echo 'export CUDA_HOME=/usr/local/cuda-13.0; export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
   source ~/.bashrc
   nvcc --version            # should say release 13.0
   ```
   This toolkit also installs the profilers `ncu` (Nsight Compute) and `nsys` (Nsight Systems).

5. **Clone inside the Linux file system** (your home folder, `~`), not under `/mnt/c/...`: builds and model loading are many times slower across the Windows boundary. Then follow the Linux steps above from `curl -LsSf ...` onward, with `export CUDA_HOME=/usr/local/cuda-13.0`.

6. **To profile with Nsight Compute,** allow access to the GPU performance counters once on Windows: NVIDIA Control Panel → Desktop menu → Enable Developer Settings → Developer → Manage GPU Performance Counters → "Allow access to the GPU performance counters to all users". Without this, `ncu` reports a permission error. If it still doesn't work, profile on Euler.

Official references: [CUDA on WSL user guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html), [CUDA toolkit for WSL-Ubuntu](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_network), [Nsight Compute requirements](https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html).

## macOS (Apple Silicon)

There is no CUDA on a Mac, so locally you work on everything that is plain Python: the reference ops, the GSM8K harness and grading, trace analysis, statistics and charts. Anything that needs an NVIDIA GPU runs on Euler. PyTorch 2.14 needs **macOS 14 (Sonoma) or newer** on an Apple Silicon Mac.

```bash
xcode-select --install                                 # compilers and git, if you don't have them
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/iSparsh/dLLM-inference-engine.git
cd dLLM-inference-engine
uv sync
uv run python tools/smoke_test.py                      # runs the CPU-only checks and says so
uv run pytest -q                                       # GPU tests skip; nothing should fail
```

**First time on a Mac, run the check script and send its report to the team:**
```bash
bash scripts/check_macos.sh            # add --skip-model to skip the 3 GB model download
```
It checks your macOS version and chip, installs uv if needed, sets up the environment, runs the reference ops on the CPU and on the Mac's GPU (MPS), runs the tests, and generates a short answer with the dLLM. Everything goes into one `mac-env-report-<time>.txt`. We haven't run this setup on a Mac yet, so the report tells us what to fix.

Optional: with the model downloaded, you can run the harness on the CPU for quick checks (`--limit 2`). It is slow, so do real runs on Euler.

## Euler (everyone)

Euler is UW's Slurm cluster. Our course partition (`instruction`) has **NVIDIA RTX 4000 SFF Ada** GPUs: the same architecture as the RTX 4060 Laptop (sm_89), 20 GB of memory, about 280 GB/s. Rules that matter: **never run programs on the login node** (CAE suspends accounts for it); compiling there is allowed. One GPU per job, 45 minutes maximum.

**One-time setup** (one job; the first run downloads about 9 GB, so allow most of the 45 minutes):

```bash
ssh <cae-username>@euler.engr.wisc.edu
git clone https://github.com/iSparsh/dLLM-inference-engine.git
cd dLLM-inference-engine
sbatch scripts/euler/setup_env.sh        # installs uv, builds ~/dllm-venv, downloads both models to ~/hf_cache,
                                         # builds our CUDA ops, runs the smoke test and pytest
squeue -u $USER                          # wait until the job is gone
cat slurm-<jobid>.out                    # check the smoke test line and that pytest has no failures
```

**Day to day:**

```bash
cd dLLM-inference-engine && git pull
sbatch scripts/euler/run_tests.sh        # the test suite on a GPU
sbatch scripts/euler/run_smoke.sh        # the smoke test
```

**Your own jobs:** copy `scripts/euler/job_template.sh` (it sets the partition, GPU, time limit, CUDA module, `~/dllm-venv` and `HF_HOME=~/hf_cache`), add your commands at the end, and `sbatch` it from the repo root. Keep each job under 45 minutes; split long sweeps into several jobs.

Euler facts (from our probe job, 2026-11-17): CUDA modules up to `nvidia/cuda/13.0.0`, driver 580, compute nodes have internet access (Hugging Face and PyPI work), home directories are on a shared file system, and each node has a local `/scratch` disk.

---

## Benchmarking hygiene (laptops)

A laptop GPU's speed depends on its power mode. Our pilot ran on Linux's "balanced" profile (45 W cap) and measured 22 ms per pass; the same code on the "performance" profile takes 17.6 ms. For every timing run:
- plug in the charger;
- Linux: `powerprofilesctl set performance` (or your vendor tool); Windows: Power mode "Best performance" and the vendor app's performance mode;
- let `dllm.timing.gpu_state()` record clocks (the benchmark scripts do this) and check the memory clock is at its maximum.

## When something goes wrong

| Symptom | Fix |
|---|---|
| `The ours backend needs an NVIDIA GPU and CUDA toolkit` | Set `CUDA_HOME` to the folder that contains `bin/nvcc`, or use the `torch` backend. |
| Warning about CUDA 13.x vs 13.0 mismatch while building | Harmless; any CUDA 13 toolkit works with our torch build. |
| `ModuleNotFoundError: einops`, or errors inside the dLLM's `modeling.py` | Run `uv sync`; don't install packages by hand. `transformers` must stay at 4.53.3. |
| Tests say "Download the model checkpoint..." | Run `uv run python tools/download_models.py`. |
| First run of an `ours` op hangs for a minute or two | That's the one-time compile. Later runs reuse the build. |
| Out of GPU memory on an 8 GB card | Load one model at a time; close other GPU programs. |
