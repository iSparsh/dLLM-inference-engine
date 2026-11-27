#!/usr/bin/env bash
#SBATCH --partition=instruction
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=00:45:00
source "${SLURM_SUBMIT_DIR:?Submit from the repo root}/scripts/euler/job_template.sh"
python tools/dump_traces.py --n 30 --tau 0.9 0.7 --out traces/n30.jsonl
