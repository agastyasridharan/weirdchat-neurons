#!/bin/bash
#SBATCH -J wcn-gpu1
#SBATCH -p gpu-he
#SBATCH --constraint=b200
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 01:00:00
#SBATCH -o gpu1_%j.out
#
# GPU job 1 (one B200, under an hour). It only reads and measures; it changes no files outside
# ~/scratch/wcn (logs and compile caches there). Three parts:
#   1. PyTorch and Triton work on a B200 with our CUDA 12.9 setup (serving env).
#   2. The attribution env really uses flash-linear-attention, and its kernels run forward and backward.
#   3. Gates G0 and G1: Qwen served by vLLM with the ablation extension reproduces the original numbers,
#      measured with the repo's client code (code/rate.py, code/common.py).
# Submit from the logs folder so this job's output file lands there:
#     cd ~/scratch/wcn/logs && sbatch ~/scratch/wcn/setup/gpu1_check.sbatch
set -Eeuo pipefail
source "$HOME/scratch/wcn/setup/env.sh"
SETUP="$HOME/scratch/wcn/setup"
trap 'echo "GPU JOB 1 FAILED at line $LINENO: $BASH_COMMAND"' ERR
echo "GPU job 1: Slurm job $SLURM_JOB_ID on $(hostname), $(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
export HF_HUB_OFFLINE=1     # compute nodes have no internet; everything is already in scratch (code/serve.sh:9 does the same)

# Parts 1 and 2 report and carry on if they fail, so one job tells us as much as possible.
echo; echo "=== 1. PyTorch and Triton on this GPU (serving env)"
"$WCN_GPU_PY" "$SETUP/gpu1_triton_check.py" || echo "TRITON CHECK FAILED (continuing to part 2)"

echo; echo "=== 2. Attribution env: transformers uses fla, and fla's kernels run"
"$WCN_ATTRIB_PY" "$SETUP/gpu1_fla_check.py" || echo "FLA CHECK FAILED (continuing to part 3)"

echo; echo "=== 3. vLLM server with the ablation extension, then gates G0 and G1"
QWEN=$("$WCN_CLIENT_PY" -c 'import json, os; print(json.load(open(os.environ["WCN_BASE"] + "/paths.json"))["qwen"])')
PORT=$("$WCN_CLIENT_PY" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')
SERVE_LOG="$WCN_BASE/logs/gpu1_serve_${SLURM_JOB_ID}.log"
echo "model: $QWEN"
echo "server log: $SERVE_LOG"
# The flags are those of the "qwen_abl" role in code/serve.sh (lines 12, 19-20), plus our two B200 settings:
# FlashInfer's sampler off (it would compile kernels with nvcc, which we don't
# have) and Triton attention (vLLM's B200 default is FlashInfer).
VLLM_SERVER_DEV_MODE=1 PYTHONPATH="$WCN_ROOT/code" VLLM_USE_FLASHINFER_SAMPLER=0 \
  "$WCN_BASE/envs/gpu/bin/vllm" serve "$QWEN" --served-model-name qwen --dtype bfloat16 --max-model-len 4096 \
  --max-num-seqs 512 --no-enable-prefix-caching --worker-extension-cls ablate_ext.AblateExt \
  --port "$PORT" --host 127.0.0.1 --gpu-memory-utilization 0.92 --limit-mm-per-prompt '{"image":0,"video":0}' \
  --attention-backend TRITON_ATTN > "$SERVE_LOG" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null || true' EXIT
UP=0
for i in $(seq 1 300); do                      # wait up to 25 minutes for the first start (it compiles)
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then UP=1; echo "server up after about $((i * 5)) s"; break; fi
  if ! kill -0 "$SPID" 2>/dev/null; then break; fi
  sleep 5
done
if [ "$UP" != 1 ]; then
  echo "STOP: the vLLM server did not start. Last lines of its log:"
  tail -n 80 "$SERVE_LOG"
  exit 1
fi
SERVERS="$WCN_BASE/tmp/servers_gpu1_${SLURM_JOB_ID}.json"
echo "{\"http://127.0.0.1:$PORT\": \"qwen_abl\"}" > "$SERVERS"
WCN_SERVERS="$SERVERS" PYTHONPATH="$WCN_ROOT/code" "$WCN_CLIENT_PY" "$SETUP/gpu1_vllm_check.py" "http://127.0.0.1:$PORT"

echo; echo "--- server log lines about the attention backend, FlashInfer and errors:"
grep -iE "backend|flashinfer|error|warning.*(ptxas|cuda)" "$SERVE_LOG" | head -n 25 || true
echo "GPU JOB 1 DONE $(date)"
