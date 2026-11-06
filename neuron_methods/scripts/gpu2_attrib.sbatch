#!/bin/bash
#SBATCH -J wcn-gpu2
#SBATCH -p gpu-he
#SBATCH --constraint=b200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=160G
#SBATCH -t 01:00:00
#SBATCH -o gpu2_%j.out
#
# GPU job 2 (one B200, under an hour): gate G2, attribution.
#   1. Runs the repo's own attribution (code/attrib.py: attribution plus its verify ablations, all 46 forks) in the
#      attribution env. The repo's files are not edited: a wrapper only points the hard-coded model path at our download.
#      Writes results/attrib/<fork>.npz and .verify.json, the files the repo's README lists as not included
#      (results/attrib/ is in the repo's .gitignore). These full scores are also what methods 1 to 3 need.
#   2. Compares with the original numbers (gpu2_attrib_check.py): M0 and V0 at the 45 forks the original logs record, and the
#      behavior sets rebuilt with the repo's sets.py logic against results/ablation_sets.json.
# Submit from the logs folder:  cd ~/scratch/wcn/logs && sbatch ~/scratch/wcn/setup/gpu2_attrib.sbatch
set -Eeuo pipefail
source "$HOME/scratch/wcn/setup/env.sh"
SETUP="$HOME/scratch/wcn/setup"
trap 'echo "GPU JOB 2 FAILED at line $LINENO: $BASH_COMMAND"' ERR
echo "GPU job 2: Slurm job $SLURM_JOB_ID on $(hostname), $(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
export HF_HUB_OFFLINE=1

echo; echo "=== 1. The repo's attribution, all 46 forks (the original log format: fork, M0, V0, times, V after top-k)"
"$WCN_ATTRIB_PY" "$SETUP/gpu2_attrib_run.py"
nvidia-smi --query-gpu=memory.used --format=csv

echo; echo "=== 2. Gate G2: compare with the original numbers"
"$WCN_CLIENT_PY" "$SETUP/gpu2_attrib_check.py"
echo "GPU JOB 2 DONE $(date)"
