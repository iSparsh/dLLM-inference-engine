#!/usr/bin/env bash
#SBATCH --partition=instruction
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=00:45:00
# Submit from the checkout root: sbatch scripts/euler/run_smoke.sh
set -euo pipefail
: "${SLURM_JOB_ID:?Submit the job with sbatch from the repo root}"
cd "${SLURM_SUBMIT_DIR:?}"
[[ -f pyproject.toml ]] || { echo "Submit from the checkout root" >&2; exit 1; }
module load nvidia/cuda/13.0.0
export UV_PROJECT_ENVIRONMENT="$HOME/dllm-venv" HF_HOME="$HOME/hf_cache"
export PATH="$UV_PROJECT_ENVIRONMENT/bin:$HOME/.local/bin:$PATH"
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-1}"
