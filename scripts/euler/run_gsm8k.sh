#!/usr/bin/env bash
#SBATCH --partition=instruction
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=00:45:00
# One job runs one setting: sbatch scripts/euler/run_gsm8k.sh --model dllm --tau 0.9
source "${SLURM_SUBMIT_DIR:?Submit from the repo root}/scripts/euler/job_template.sh"
python eval/gsm8k.py "$@"
