#!/bin/bash
#SBATCH -J wcn-gpu4b
#SBATCH -p gpu-he
#SBATCH --constraint=b200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=160G
#SBATCH -t 01:30:00
#SBATCH -o gpu4b_%j.out
#
# GPU job 4b (one B200, under an hour): Step 3 for method M4, RelP.
#   Patches the model's backward pass (relp.py), checks the patches on the real model, computes RelP attribution at
#   all 46 forks, runs the M4 check set before the run (top 10 must halve V at 35 or more of 46 forks), and builds
#   M4's sets, its matched random set at 100 and V at each set's own forks.
#   Reads the repo's results and GPU job 2's results/attrib/ files, read only. Writes only ~/scratch/wcn/methods/:
#   m4_sets.json, m4_summary.json and relp/ (46 attribution files, about 4 MB each), moved into methods/ only if
#   every technical check passes. An M4 check FAIL is a result, not an error: M4 is then dropped.
# Submit from the logs folder:  cd ~/scratch/wcn/logs && sbatch ~/scratch/wcn/setup/gpu4b.sbatch
set -Eeuo pipefail
source "$HOME/scratch/wcn/setup/env.sh"
SETUP="$HOME/scratch/wcn/setup"
trap 'echo "GPU JOB 4b FAILED at line $LINENO: $BASH_COMMAND"' ERR
echo "GPU job 4b: Slurm job $SLURM_JOB_ID on $(hostname), $(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
export HF_HUB_OFFLINE=1
"$WCN_ATTRIB_PY" "$SETUP/gpu4b_m4.py"
nvidia-smi --query-gpu=memory.used --format=csv
echo "GPU JOB 4b DONE $(date)"
