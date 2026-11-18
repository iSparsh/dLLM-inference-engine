#!/usr/bin/env bash
#SBATCH --partition=instruction
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=00:45:00
source "${SLURM_SUBMIT_DIR:?Submit from the repo root}/scripts/euler/job_template.sh"
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" sh
fi
uv sync --locked --python 3.12
python tools/download_models.py
python -c "from dllm.ops.cuda import load; load()"   # build our CUDA ops once
tools/fetch_golden.sh
python tools/smoke_test.py
pytest -q
