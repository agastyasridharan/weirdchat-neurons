#!/bin/bash
#SBATCH -J wcn-gpu3
#SBATCH -p gpu-he
#SBATCH --constraint=b200
#SBATCH --gres=gpu:2
#SBATCH -c 16
#SBATCH --mem=192G
#SBATCH -t 01:30:00
#SBATCH -o gpu3_%j.out
#
# GPU job 3 (two B200s, about an hour): gates G3 and G4.
#   GPU 1: the Gemma judge server (code/serve.sh "gemma" role, line 26) plus our B200 settings.
#   GPU 2: the Qwen server with the ablation extension (as in GPU job 1).
#   G3: the judge re-judges 300 of the stored unedited replies; it must agree with the original verdicts (kappa >= 0.75).
#   G4: the 12 code-execution test prompts, 16 replies each, unedited and with the original top-100 set zeroed.
# Replies and verdicts are saved in ~/scratch/wcn/gates/, not in results/.
# Submit from the logs folder:  cd ~/scratch/wcn/logs && sbatch ~/scratch/wcn/setup/gpu3_judge.sbatch
set -Eeuo pipefail
source "$HOME/scratch/wcn/setup/env.sh"
SETUP="$HOME/scratch/wcn/setup"
trap 'echo "GPU JOB 3 FAILED at line $LINENO: $BASH_COMMAND"' ERR
echo "GPU job 3: Slurm job $SLURM_JOB_ID on $(hostname), $(date); GPUs $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
export HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
IFS=, read -r GPU_J GPU_Q <<< "$CUDA_VISIBLE_DEVICES"
[ -n "$GPU_J" ] && [ -n "$GPU_Q" ] || { echo "STOP: expected two GPUs, got '$CUDA_VISIBLE_DEVICES'"; exit 1; }
PATHS="$WCN_BASE/paths.json"
GEMMA=$("$WCN_CLIENT_PY" -c "import json; print(json.load(open('$PATHS'))['gemma'])")
QWEN=$("$WCN_CLIENT_PY" -c "import json; print(json.load(open('$PATHS'))['qwen'])")
free_port() { "$WCN_CLIENT_PY" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()'; }
PJ=$(free_port); PQ=$(free_port)
JLOG="$WCN_BASE/logs/gpu3_serve_gemma_${SLURM_JOB_ID}.log"; QLOG="$WCN_BASE/logs/gpu3_serve_qwen_${SLURM_JOB_ID}.log"
echo "judge log: $JLOG"; echo "qwen log:  $QLOG"

# Judge: the repo's flags (code/serve.sh:12, 26), plus Triton attention and the CUTLASS FP8 kernels. On B200s vLLM would
# otherwise pick FlashInfer for both, and FlashInfer needs nvcc or the internet to get its kernels. On the original H200s,
# vLLM 0.30.0 already uses CUTLASS for FP8, because its FlashInfer FP8 kernel needs compute capability 100.
CUDA_VISIBLE_DEVICES="$GPU_J" "$WCN_BASE/envs/gpu/bin/vllm" serve "$GEMMA" --served-model-name gemma \
  --quantization fp8 --max-model-len 20480 --max-num-seqs 256 --enable-prefix-caching --reasoning-parser gemma4 \
  --port "$PJ" --host 127.0.0.1 --gpu-memory-utilization 0.92 --limit-mm-per-prompt '{"image":0,"video":0}' \
  --attention-backend TRITON_ATTN --linear-backend cutlass > "$JLOG" 2>&1 &
JPID=$!
# Qwen with the ablation extension: exactly as in GPU job 1
CUDA_VISIBLE_DEVICES="$GPU_Q" VLLM_SERVER_DEV_MODE=1 PYTHONPATH="$WCN_ROOT/code" \
  "$WCN_BASE/envs/gpu/bin/vllm" serve "$QWEN" --served-model-name qwen --dtype bfloat16 --max-model-len 4096 \
  --max-num-seqs 512 --no-enable-prefix-caching --worker-extension-cls ablate_ext.AblateExt \
  --port "$PQ" --host 127.0.0.1 --gpu-memory-utilization 0.92 --limit-mm-per-prompt '{"image":0,"video":0}' \
  --attention-backend TRITON_ATTN > "$QLOG" 2>&1 &
QPID=$!
trap 'kill $JPID $QPID 2>/dev/null || true' EXIT

wait_up() {   # name, port, pid, log
  for i in $(seq 1 300); do
    if curl -sf "http://127.0.0.1:$2/health" > /dev/null; then echo "$1 server up after about $((i * 5)) s"; return 0; fi
    if ! kill -0 "$3" 2>/dev/null; then break; fi
    sleep 5
  done
  echo "STOP: the $1 server did not start. Last lines of its log:"; tail -n 80 "$4"; return 1
}
wait_up judge "$PJ" "$JPID" "$JLOG"
wait_up qwen "$PQ" "$QPID" "$QLOG"

SERVERS="$WCN_BASE/tmp/servers_gpu3_${SLURM_JOB_ID}.json"
echo "{\"http://127.0.0.1:$PJ\": \"gemma\", \"http://127.0.0.1:$PQ\": \"qwen_abl\"}" > "$SERVERS"
WCN_SERVERS="$SERVERS" PYTHONPATH="$WCN_ROOT/code" "$WCN_CLIENT_PY" "$SETUP/gpu3_judge_check.py" "http://127.0.0.1:$PQ"

echo; echo "--- judge server log lines about kernels, backends and errors:"
grep -iE "backend|Selected .*Kernel|flashinfer|cutlass|error" "$JLOG" | head -n 25 || true
echo "GPU JOB 3 DONE $(date)"
