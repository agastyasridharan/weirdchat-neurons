#!/bin/bash
#SBATCH -J wcn-gpu4a
#SBATCH -p gpu-he
#SBATCH --constraint=b200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=160G
#SBATCH -t 01:30:00
#SBATCH -o gpu4a_%j.out
#
# GPU job 4a (one B200, under an hour): Step 3 for methods M1, M2 and M3.
#   Builds each method's sets of 3, 5, 10, 20, 50 and 100 neurons for the four ladder behaviors from the 46 discovery
#   forks, a matched random set at 100 for each, and V at each set's own forks. M3 runs 145 gradient passes.
#   Reads the repo's results (forks, prompt token ids, ablation_sets.json, upstream_sets.json) and GPU job 2's results/attrib/ files, read only.
#   Writes only ~/scratch/wcn/methods/: m123_sets.json, m123_summary.json, m3_steps.json (in a job folder first;
#   moved into methods/ only if every check passes).
# Submit from the logs folder:  cd ~/scratch/wcn/logs && sbatch ~/scratch/wcn/setup/gpu4a.sbatch
set -Eeuo pipefail
source "$HOME/scratch/wcn/setup/env.sh"
SETUP="$HOME/scratch/wcn/setup"
trap 'echo "GPU JOB 4a FAILED at line $LINENO: $BASH_COMMAND"' ERR
echo "GPU job 4a: Slurm job $SLURM_JOB_ID on $(hostname), $(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
export HF_HUB_OFFLINE=1
"$WCN_ATTRIB_PY" "$SETUP/gpu4a_m123.py"
nvidia-smi --query-gpu=memory.used --format=csv
echo "GPU JOB 4a DONE $(date)"
