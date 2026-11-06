#!/bin/bash
#SBATCH -J wcn-gpu8
#SBATCH -p gpu-he
#SBATCH --constraint=b200
#SBATCH --gres=gpu:3
#SBATCH -c 24
#SBATCH --mem=256G
#SBATCH -t 02:00:00
#SBATCH -o gpu8_%j.out
#SBATCH --dependency=singleton
#
# GPU job 8 (three B200s, about an hour): Step 6, part 2, the judged harm checks on the final sets.
# The same layout, servers and flags as GPU jobs 5 and 6:
#   GPU 1: the Qwen server with the ablation extension.
#   GPUs 2 and 3: two Gemma judge servers; the second starts after the first is up.
#   Then gpu8_final.py: the unedited model and the final sets picked in GPU job 6 (up to 8), each on the repo's
#   legitimate-use and self prompts, the other behaviors' held-out test prompts, and the benchmark items, with damage.
#   It imports gpu6_heldout.py and gpu7_bench.py from the same folder.
# Writes only ~/scratch/wcn/runs/final/ (and server logs in ~/scratch/wcn/logs/). Rerunning continues where it stopped.
# --dependency=singleton: a second copy submitted by mistake waits until the first has ended, so two never write
# to the same files at once.
# Submit from the logs folder:  cd ~/scratch/wcn/logs && sbatch ~/scratch/wcn/setup/gpu8_final.sbatch
set -Eeuo pipefail
source "$HOME/scratch/wcn/setup/env.sh"
SETUP="$HOME/scratch/wcn/setup"
trap 'echo "GPU JOB 8 FAILED at line $LINENO: $BASH_COMMAND"' ERR
echo "GPU job 8: Slurm job $SLURM_JOB_ID on $(hostname), $(date); GPUs $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
export HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
IFS=, read -r GPU_Q GPU_J1 GPU_J2 <<< "$CUDA_VISIBLE_DEVICES"
[ -n "$GPU_Q" ] && [ -n "$GPU_J1" ] && [ -n "$GPU_J2" ] || { echo "STOP: expected three GPUs, got '$CUDA_VISIBLE_DEVICES'"; exit 1; }
# Before any server starts: GPU job 6 must have finished with every check passed, since its summary names the sets
"$WCN_CLIENT_PY" - "$WCN_BASE/runs/heldout/heldout_summary.json" <<'PY' || { echo "STOP: GPU job 6 has not finished with every check passed. Submit this job after it has."; exit 1; }
import json, sys
h = json.load(open(sys.argv[1]))
assert h.get("checks") and all(c["pass"] for c in h["checks"]), "a held-out check failed"
assert len(h.get("final_sets", {})) == 4, "no final sets"
print("GPU job 6 summary found: every check passed; final sets for", ", ".join(sorted(h["final_sets"])))
PY
PATHS="$WCN_BASE/paths.json"
GEMMA=$("$WCN_CLIENT_PY" -c "import json; print(json.load(open('$PATHS'))['gemma'])")
QWEN=$("$WCN_CLIENT_PY" -c "import json; print(json.load(open('$PATHS'))['qwen'])")
free_port() { "$WCN_CLIENT_PY" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()'; }
PQ=$(free_port); PJ1=$(free_port); PJ2=$(free_port)
QLOG="$WCN_BASE/logs/gpu8_serve_qwen_${SLURM_JOB_ID}.log"
JLOG1="$WCN_BASE/logs/gpu8_serve_gemma1_${SLURM_JOB_ID}.log"
JLOG2="$WCN_BASE/logs/gpu8_serve_gemma2_${SLURM_JOB_ID}.log"
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

SERVERS="$WCN_BASE/tmp/servers_gpu8_${SLURM_JOB_ID}.json"
echo "{\"http://127.0.0.1:$PQ\": \"qwen_abl\", \"http://127.0.0.1:$PJ1\": \"gemma\", \"http://127.0.0.1:$PJ2\": \"gemma\"}" > "$SERVERS"
WCN_SERVERS="$SERVERS" PYTHONPATH="$WCN_ROOT/code" "$WCN_CLIENT_PY" "$SETUP/gpu8_final.py"

echo; echo "--- server log lines about errors:"
grep -iE "error|traceback" "$QLOG" "$JLOG1" "$JLOG2" | head -n 20 || true
echo "GPU JOB 8 DONE $(date)"
