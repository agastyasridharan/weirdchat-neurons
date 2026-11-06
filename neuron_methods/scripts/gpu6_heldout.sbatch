#!/bin/bash
#SBATCH -J wcn-gpu6
#SBATCH -p gpu-he
#SBATCH --constraint=b200
#SBATCH --gres=gpu:3
#SBATCH -c 24
#SBATCH --mem=256G
#SBATCH -t 02:30:00
#SBATCH -o gpu6_%j.out
#SBATCH --dependency=singleton
#
# GPU job 6 (three B200s, about 1.5 to 2 hours): Step 5, the held-out run.
# The same layout, servers and flags as GPU job 5, which ran on these nodes:
#   GPU 1: the Qwen server with the ablation extension (as in GPU jobs 1, 3 and 5).
#   GPUs 2 and 3: two Gemma judge servers (code/serve.sh "gemma" role plus our B200 settings, as in GPU jobs 3
#     and 5). The second judge starts after the first is up, so the two never fill the compile cache at the same time.
#   Then gpu6_heldout.py: every method that passed validation at sizes 3 to 100, the matched random sets, and the
#   unedited model first and last, on the 38 held-out test prompts, 16 replies each, judged by the production judge,
#   with damage for every condition; then the pre-set statistics.
# Writes only ~/scratch/wcn/runs/heldout/ (and server logs in ~/scratch/wcn/logs/). Rerunning continues where it stopped.
# --dependency=singleton: a second copy submitted by mistake waits until the first has ended, so two never write
# to the same files at once.
# Submit from the logs folder:  cd ~/scratch/wcn/logs && sbatch ~/scratch/wcn/setup/gpu6_heldout.sbatch
set -Eeuo pipefail
source "$HOME/scratch/wcn/setup/env.sh"
SETUP="$HOME/scratch/wcn/setup"
trap 'echo "GPU JOB 6 FAILED at line $LINENO: $BASH_COMMAND"' ERR
echo "GPU job 6: Slurm job $SLURM_JOB_ID on $(hostname), $(date); GPUs $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
export HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
IFS=, read -r GPU_Q GPU_J1 GPU_J2 <<< "$CUDA_VISIBLE_DEVICES"
[ -n "$GPU_Q" ] && [ -n "$GPU_J1" ] && [ -n "$GPU_J2" ] || { echo "STOP: expected three GPUs, got '$CUDA_VISIBLE_DEVICES'"; exit 1; }
PATHS="$WCN_BASE/paths.json"
GEMMA=$("$WCN_CLIENT_PY" -c "import json; print(json.load(open('$PATHS'))['gemma'])")
QWEN=$("$WCN_CLIENT_PY" -c "import json; print(json.load(open('$PATHS'))['qwen'])")
free_port() { "$WCN_CLIENT_PY" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()'; }
PQ=$(free_port); PJ1=$(free_port); PJ2=$(free_port)
QLOG="$WCN_BASE/logs/gpu6_serve_qwen_${SLURM_JOB_ID}.log"
JLOG1="$WCN_BASE/logs/gpu6_serve_gemma1_${SLURM_JOB_ID}.log"
JLOG2="$WCN_BASE/logs/gpu6_serve_gemma2_${SLURM_JOB_ID}.log"
echo "qwen log:   $QLOG"; echo "judge logs: $JLOG1"; echo "            $JLOG2"

# Judge: the repo's flags (code/serve.sh:12, 26) plus Triton attention and the CUTLASS FP8 kernels, exactly as in GPU jobs 3 and 5.
start_judge() {  # GPU, port, log
  CUDA_VISIBLE_DEVICES="$1" "$WCN_BASE/envs/gpu/bin/vllm" serve "$GEMMA" --served-model-name gemma \
    --quantization fp8 --max-model-len 20480 --max-num-seqs 256 --enable-prefix-caching --reasoning-parser gemma4 \
    --port "$2" --host 127.0.0.1 --gpu-memory-utilization 0.92 --limit-mm-per-prompt '{"image":0,"video":0}' \
    --attention-backend TRITON_ATTN --linear-backend cutlass > "$3" 2>&1 &
}
# Qwen with the ablation extension: exactly as in GPU job 5, with PYTHONDONTWRITEBYTECODE=1 so that importing the repo's
# ablate_ext.py doesn't write a __pycache__ folder into the repo
CUDA_VISIBLE_DEVICES="$GPU_Q" VLLM_SERVER_DEV_MODE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$WCN_ROOT/code" \
  "$WCN_BASE/envs/gpu/bin/vllm" serve "$QWEN" --served-model-name qwen --dtype bfloat16 --max-model-len 4096 \
  --max-num-seqs 512 --no-enable-prefix-caching --worker-extension-cls ablate_ext.AblateExt \
  --port "$PQ" --host 127.0.0.1 --gpu-memory-utilization 0.92 --limit-mm-per-prompt '{"image":0,"video":0}' \
  --attention-backend TRITON_ATTN > "$QLOG" 2>&1 &
QPID=$!
start_judge "$GPU_J1" "$PJ1" "$JLOG1"
JPID1=$!
JPID2=""
trap 'kill $QPID $JPID1 $JPID2 2>/dev/null || true' EXIT

wait_up() {   # name, port, pid, log
  for i in $(seq 1 300); do
    if curl -sf "http://127.0.0.1:$2/health" > /dev/null; then echo "$1 server up after about $((i * 5)) s"; return 0; fi
    if ! kill -0 "$3" 2>/dev/null; then break; fi
    sleep 5
  done
  echo "STOP: the $1 server did not start. Last lines of its log:"; tail -n 80 "$4"; return 1
}
wait_up "judge 1" "$PJ1" "$JPID1" "$JLOG1"
start_judge "$GPU_J2" "$PJ2" "$JLOG2"
JPID2=$!
wait_up qwen "$PQ" "$QPID" "$QLOG"
wait_up "judge 2" "$PJ2" "$JPID2" "$JLOG2"

SERVERS="$WCN_BASE/tmp/servers_gpu6_${SLURM_JOB_ID}.json"
echo "{\"http://127.0.0.1:$PQ\": \"qwen_abl\", \"http://127.0.0.1:$PJ1\": \"gemma\", \"http://127.0.0.1:$PJ2\": \"gemma\"}" > "$SERVERS"
WCN_SERVERS="$SERVERS" PYTHONPATH="$WCN_ROOT/code" "$WCN_CLIENT_PY" "$SETUP/gpu6_heldout.py"

echo; echo "--- server log lines about errors:"
grep -iE "error|traceback" "$QLOG" "$JLOG1" "$JLOG2" | head -n 20 || true
echo "GPU JOB 6 DONE $(date)"
