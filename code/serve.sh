#!/bin/bash
# usage: serve.sh {qwen|gemma} GPU PORT
# One vLLM server per GPU (TP=1, data parallel across servers). Logs in logs/serve_<role>_<gpu>.log
ROLE=$1; GPU=$2; PORT=$3
P=/data/agastyas/weirdchat-neurons
source /data/agastyas/latekey/env.sh
V=/data/agastyas/latekey/vllm-env
export CUDA_HOME=$V/lib/python3.12/site-packages/nvidia/cu13 PATH=$V/lib/python3.12/site-packages/nvidia/cu13/bin:$V/bin:$PATH
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU VLLM_CACHE_ROOT=$P/tmp/vllm_cache TMPDIR=$P/tmp FLASHINFER_WORKSPACE_BASE=$P/tmp
QWEN=/data/hf_cache/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
GEMMA=/data/hf_cache/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475
COMMON="--port $PORT --host 127.0.0.1 --gpu-memory-utilization 0.92 --limit-mm-per-prompt {\"image\":0,\"video\":0}"
if [ "$ROLE" = qwen ]; then
  # subject: bf16 (the precision we will instrument), sampling defaults from generation_config
  exec vllm serve $QWEN --served-model-name qwen --dtype bfloat16 --max-model-len 4096 --max-num-seqs 512 --enable-prefix-caching $COMMON
elif [ "$ROLE" = qwen_abl ]; then
  # subject with neuron ablation via /collective_rpc set_ablation (ablate_ext.py); no prefix cache, so no stale
  # KV/state from before an ablation change can be reused
  export VLLM_SERVER_DEV_MODE=1 PYTHONPATH=$P/code
  exec vllm serve $QWEN --served-model-name qwen --dtype bfloat16 --max-model-len 4096 --max-num-seqs 512 --no-enable-prefix-caching --worker-extension-cls ablate_ext.AblateExt $COMMON
elif [ "$ROLE" = gemma_fp8kv ]; then
  # judge with FP8 KV cache (as in WeirdChat production); FA4 fp8-KV kernels need sm100, so FlashInfer on H200
  exec vllm serve $GEMMA --served-model-name gemma --quantization fp8 --kv-cache-dtype fp8 --attention-backend FLASHINFER --max-model-len 20480 --max-num-seqs 512 --enable-prefix-caching --reasoning-parser gemma4 $COMMON
else
  # judge: FP8 weights as in WeirdChat production; thinking toggled per request
  exec vllm serve $GEMMA --served-model-name gemma --quantization fp8 --max-model-len 20480 --max-num-seqs 256 --enable-prefix-caching --reasoning-parser gemma4 $COMMON
fi
