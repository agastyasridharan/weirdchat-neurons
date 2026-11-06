#!/bin/bash
# env.sh: settings for the WeirdChat x neurons work on Oscar.
# The shell setup scripts and the GPU jobs source it inside their own process. Steps 3 and 4 need it
# sourced first: do that in a separate shell and close it afterwards, because anything started from a
# shell that sourced it, including sbatch jobs for other projects, would inherit these cache locations.
# It only sets variables and creates folders inside ~/scratch/wcn. Nothing goes to your home folder.

export WCN_BASE="$HOME/scratch/wcn"                  # everything for this project lives here
export WCN_ROOT="$WCN_BASE/weirdchat-neurons"        # the repo's code reads WCN_ROOT (code/common.py:21)

# Keep other projects' Python settings out, and keep Python from reading packages in ~/.local
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1

# Python environments (made by step1_install.sh)
export WCN_GPU_PY="$WCN_BASE/envs/gpu/bin/python"         # vLLM 0.30.0 (CUDA 12.9 build): serving
export WCN_ATTRIB_PY="$WCN_BASE/envs/attrib/bin/python"   # torch + transformers 5.14.1 + fla (Python 3.13): attribution
export WCN_CLIENT_PY="$WCN_BASE/envs/client/bin/python"   # drivers, judge parsing, downloads, checks

# uv, the installer: its cache, its Python builds and its links all stay in scratch;
# any personal uv config file is ignored, so every run installs the same way
export PATH="$WCN_BASE/bin:$PATH"
export UV_CACHE_DIR="$WCN_BASE/cache/uv"
export UV_PYTHON_INSTALL_DIR="$WCN_BASE/python"
export UV_PYTHON_BIN_DIR="$WCN_BASE/bin"
export UV_PYTHON_INSTALL_BIN=0
export UV_TOOL_DIR="$WCN_BASE/uv-tools"
export UV_TOOL_BIN_DIR="$WCN_BASE/bin"
export UV_NO_CONFIG=1
export UV_PYTHON_PREFERENCE=only-managed               # only the Pythons uv installed here, never conda or system
export UV_CONCURRENT_DOWNLOADS=8
export UV_CONCURRENT_INSTALLS=4
export PIP_CACHE_DIR="$WCN_BASE/cache/pip"

# Hugging Face: models and datasets go to scratch; no telemetry. A token, if used, comes only from
# the HF_TOKEN variable in your shell; nothing here stores it.
export HF_HOME="$WCN_BASE/hf"
export HF_HUB_DISABLE_TELEMETRY=1

# Caches and settings folders that default to the home folder
export XDG_CACHE_HOME="$WCN_BASE/cache/xdg"
export XDG_CONFIG_HOME="$WCN_BASE/xdg/config"
export XDG_DATA_HOME="$WCN_BASE/xdg/data"
export XDG_STATE_HOME="$WCN_BASE/xdg/state"
export TORCH_HOME="$WCN_BASE/cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$WCN_BASE/cache/inductor"
export TRITON_HOME="$WCN_BASE/cache/triton-home"
export TRITON_CACHE_DIR="$WCN_BASE/cache/triton"
export CUDA_CACHE_PATH="$WCN_BASE/cache/nv"
export VLLM_CACHE_ROOT="$WCN_BASE/cache/vllm"          # code/serve.sh:9 sets this too
export VLLM_CONFIG_ROOT="$WCN_BASE/cache/vllm-config"
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export FLASHINFER_WORKSPACE_BASE="$WCN_BASE/cache/flashinfer"   # code/serve.sh:9 sets this too
export MPLCONFIGDIR="$WCN_BASE/cache/matplotlib"
export TMPDIR="$WCN_BASE/tmp"

# B200 GPUs only: Triton compiles for B200 with its own CUDA 13.1 ptxas, which the B200 driver
# (575.57.08, CUDA 12.9) can't be expected to run. Point it at the CUDA 12.9 ptxas that step 1
# installs. H100s use Triton's other ptxas (CUDA 12.8), which is left alone.
_wcn_ptxas="$WCN_BASE/envs/gpu/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin/ptxas"
if [ -x "$_wcn_ptxas" ]; then export TRITON_PTXAS_BLACKWELL_PATH="$_wcn_ptxas"; fi
unset _wcn_ptxas

mkdir -p "$WCN_BASE/bin" "$WCN_BASE/logs" "$WCN_BASE/envs" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" \
         "$UV_TOOL_DIR" "$PIP_CACHE_DIR" "$HF_HOME" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" \
         "$XDG_STATE_HOME" "$TORCH_HOME" "$TORCHINDUCTOR_CACHE_DIR" \
         "$TRITON_HOME" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$VLLM_CACHE_ROOT" "$VLLM_CONFIG_ROOT" \
         "$FLASHINFER_WORKSPACE_BASE" "$MPLCONFIGDIR" "$TMPDIR"
