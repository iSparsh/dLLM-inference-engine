#!/usr/bin/env bash
#SBATCH --partition=instruction
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=00:45:00
source "${SLURM_SUBMIT_DIR:?Submit from the repo root}/scripts/euler/job_template.sh"
pytest -q
