#!/bin/bash
#SBATCH -J wcn-gpu7
#SBATCH -p gpu-he
#SBATCH --constraint=b200
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 01:30:00
#SBATCH -o gpu7_%j.out
#SBATCH --dependency=singleton
#
# GPU job 7 (one B200, about 20 to 30 minutes): Step 6, part 1, the harm checks that need no judge.
#   One Qwen server with the ablation extension, started exactly as in GPU job 5 (same flags, so it reuses the
#   compile cache that job filled). No judge.
#   Then gpu7_bench.py: the unedited model first and last, and every method's sets at sizes 10 and 100, each on 40
#   yes/no facts, 400 MMLU questions and 250 GSM8K problems (greedy, as in the repo's sidefx.py), with damage for each.
# It can run at the same time as GPU job 6 (3 GPUs): together they use 4 GPUs.
# Writes only ~/scratch/wcn/runs/bench/ (and a server log in ~/scratch/wcn/logs/ and its servers file in ~/scratch/wcn/tmp/).
# Rerunning continues where it stopped. --dependency=singleton: a second copy submitted by mistake waits until the
# first has ended, so two never write to the same files at once.
# Submit from the logs folder:  cd ~/scratch/wcn/logs && sbatch ~/scratch/wcn/setup/gpu7_bench.sbatch
set -Eeuo pipefail
source "$HOME/scratch/wcn/setup/env.sh"
SETUP="$HOME/scratch/wcn/setup"
trap 'echo "GPU JOB 7 FAILED at line $LINENO: $BASH_COMMAND"' ERR
echo "GPU job 7: Slurm job $SLURM_JOB_ID on $(hostname), $(date); GPUs $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
export HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
IFS=, read -r GPU_Q REST <<< "$CUDA_VISIBLE_DEVICES"
[ -n "$GPU_Q" ] && [ -z "${REST:-}" ] || { echo "STOP: expected one GPU, got '$CUDA_VISIBLE_DEVICES'"; exit 1; }
PATHS="$WCN_BASE/paths.json"
QWEN=$("$WCN_CLIENT_PY" -c "import json; print(json.load(open('$PATHS'))['qwen'])")
free_port() { "$WCN_CLIENT_PY" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()'; }
PQ=$(free_port)
QLOG="$WCN_BASE/logs/gpu7_serve_qwen_${SLURM_JOB_ID}.log"
echo "qwen log: $QLOG"

# Qwen with the ablation extension: exactly as in GPU job 5, with PYTHONDONTWRITEBYTECODE=1 so that importing the repo's
# ablate_ext.py doesn't write a __pycache__ folder into the repo
CUDA_VISIBLE_DEVICES="$GPU_Q" VLLM_SERVER_DEV_MODE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$WCN_ROOT/code" \
  "$WCN_BASE/envs/gpu/bin/vllm" serve "$QWEN" --served-model-name qwen --dtype bfloat16 --max-model-len 4096 \
  --max-num-seqs 512 --no-enable-prefix-caching --worker-extension-cls ablate_ext.AblateExt \
  --port "$PQ" --host 127.0.0.1 --gpu-memory-utilization 0.92 --limit-mm-per-prompt '{"image":0,"video":0}' \
  --attention-backend TRITON_ATTN > "$QLOG" 2>&1 &
QPID=$!
trap 'kill $QPID 2>/dev/null || true' EXIT

wait_up() {   # name, port, pid, log
  for i in $(seq 1 300); do
    if curl -sf "http://127.0.0.1:$2/health" > /dev/null; then echo "$1 server up after about $((i * 5)) s"; return 0; fi
    if ! kill -0 "$3" 2>/dev/null; then break; fi
    sleep 5
  done
  echo "STOP: the $1 server did not start. Last lines of its log:"; tail -n 80 "$4"; return 1
}
wait_up qwen "$PQ" "$QPID" "$QLOG"

SERVERS="$WCN_BASE/tmp/servers_gpu7_${SLURM_JOB_ID}.json"
echo "{\"http://127.0.0.1:$PQ\": \"qwen_abl\"}" > "$SERVERS"
WCN_SERVERS="$SERVERS" PYTHONPATH="$WCN_ROOT/code" "$WCN_CLIENT_PY" "$SETUP/gpu7_bench.py"

echo; echo "--- server log lines about errors:"
grep -iE "error|traceback" "$QLOG" | head -n 20 || true
echo "GPU JOB 7 DONE $(date)"
