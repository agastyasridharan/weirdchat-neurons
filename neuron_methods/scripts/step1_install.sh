#!/bin/bash
# STEP 1: install uv, Python 3.12 and 3.13, and the three Python environments, all inside ~/scratch/wcn.
#
# Where: a LOGIN node (it needs internet; compute nodes have none). It only downloads and unpacks
# packages, at low priority, and writes nothing outside ~/scratch/wcn. About 20 GB, 15 to 30 minutes.
# Run it so it survives a dropped connection, then follow the log:
#     nohup bash ~/scratch/wcn/setup/step1_install.sh > /dev/null 2>&1 &
#     tail -f ~/scratch/wcn/logs/step1_install_*.log        (Ctrl-C stops watching, not the install)
# It is safe to run again: finished parts are skipped or left unchanged.
#
# Why these versions:
# - vLLM 0.30.0 is the repo's version (README.md:70). Its default PyPI build needs CUDA 13, but the
#   B200 driver is 575.57.08 (CUDA 12.9), so we use the CUDA 12.9 build from vLLM's GitHub release,
#   checked by its SHA-256 below, with PyTorch 2.13.0 built for CUDA 12.9.
# - transformers 5.14.1 for attribution: the newest version whose Qwen3.5 code prints the
#   "fast path is not available" warning seen in the original attribution logs (logs/attrib_0.log:1).
#   Python 3.13, as in those logs (envs/ml/lib/python3.13).
# - flash-linear-attention: the repo's README.md:88 and PHASE1.md:78 ("with flash-linear-attention kernels,
#   Triton 3.7.1 in genv") say the original attribution environment had it. Since 5.15.0 (Aug 10) replaced the
#   code behind that warning, the original environment dates from mid-July to Aug 10 at the latest; fla was
#   0.5.1 or 0.5.2 (Jul 27) then. We use 0.5.2; gate G2 shows whether it matters. causal-conv1d
#   is NOT installed, matching the warning.
# - docent-python 0.1.75: the version in WeirdChat's own lock file. Its transcript rendering is
#   identical to the newest version's.
set -Eeuo pipefail
SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SETUP_DIR/env.sh"
LOG="$WCN_BASE/logs/step1_install_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
trap 'echo "STEP 1 FAILED at line $LINENO: $BASH_COMMAND"' ERR
echo "STEP 1 start $(date) on $(hostname). Log: $LOG"

# 0. Refuse to run where it can't work
if [ -n "${SLURM_JOB_ID:-}" ]; then echo "STOP: this is inside a Slurm job (compute node, no internet). Run it on a login node."; exit 1; fi
curl -sS -o /dev/null --max-time 20 https://pypi.org/simple/pip/ || { echo "STOP: no internet here. Run it on a login node."; exit 1; }
echo "glibc: $(ldd --version 2>&1 | sed -n 1p)   (vLLM's packages need 2.31 or newer; RHEL 9 has 2.34)"
export UV_HTTP_TIMEOUT=300

VLLM_WHEEL_NAME="vllm-0.30.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
VLLM_WHEEL_URL="https://github.com/vllm-project/vllm/releases/download/v0.30.0/vllm-0.30.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
VLLM_WHEEL_SHA256="e98cb69659bfcfc849cf11ce0781a7161d40b02b51a6c3636924a5909f2aabcc"

# 1. uv 0.13.6, installed with Oscar's own python3 into a small folder; only the uv command is linked
if [ ! -x "$WCN_BASE/tools/bin/uv" ]; then
  python3 -m venv "$WCN_BASE/tools"
  nice -n 10 "$WCN_BASE/tools/bin/pip" install --disable-pip-version-check "uv==0.13.6"
fi
ln -sf "$WCN_BASE/tools/bin/uv" "$WCN_BASE/bin/uv"
uv --version

# 2. Python 3.12, the version of the original vLLM environment (code/serve.sh:7-8), and 3.13, the version of
#    the original attribution environment (logs/attrib_0.log:2)
nice -n 10 uv python install 3.12 3.13

# 3. The CUDA 12.9 build of vLLM 0.30.0, checked against a pinned SHA-256 of the release asset
mkdir -p "$WCN_BASE/wheels"
W="$WCN_BASE/wheels/$VLLM_WHEEL_NAME"
if [ ! -f "$W" ]; then
  curl -fL --retry 3 --max-time 1800 -o "$W.part" "$VLLM_WHEEL_URL"
  mv "$W.part" "$W"
fi
echo "$VLLM_WHEEL_SHA256  $W" | sha256sum -c - || { mv -f "$W" "$W.bad"; echo "STOP: the vLLM wheel's SHA-256 does not match (kept as $W.bad; a re-run downloads it again)."; exit 1; }

# 4. Serving environment: vLLM + PyTorch 2.13.0 for CUDA 12.9. nvidia-cuda-nvcc-cu12 supplies the
#    CUDA 12.9 ptxas that env.sh gives Triton on B200s. transformers pinned to the version vLLM
#    0.30.0's release notes name.
[ -x "$WCN_GPU_PY" ] || uv venv --python 3.12 "$WCN_BASE/envs/gpu"
nice -n 10 uv pip install --python "$WCN_GPU_PY" --torch-backend=cu129 \
    "$W" "transformers==5.16.1" "nvidia-cuda-nvcc-cu12==12.9.86"

# 5. Attribution environment (Python 3.13): PyTorch 2.13.0 for CUDA 12.9, transformers 5.14.1, accelerate
#    (needed for device_map="cuda" in code/attrib.py:42), flash-linear-attention, and what the repo's
#    code/common.py imports at the top (pandas, httpx). On B200s its Triton uses the serving env's
#    CUDA 12.9 ptxas, through env.sh.
[ -x "$WCN_ATTRIB_PY" ] || uv venv --python 3.13 "$WCN_BASE/envs/attrib"
nice -n 10 uv pip install --python "$WCN_ATTRIB_PY" --torch-backend=cu129 \
    "torch==2.13.0" "transformers==5.14.1" "accelerate==1.15.0" "flash-linear-attention==0.5.2" \
    numpy pandas pyarrow httpx safetensors

# 6. Client environment: the repo's drivers and the judge code (README.md:87-88; code/wcref/judge.py:15-17,216;
#    code/wcref/types.py:16; code/compare_judges.py:8), plus huggingface_hub for downloads. No torch.
[ -x "$WCN_CLIENT_PY" ] || uv venv --python 3.12 "$WCN_BASE/envs/client"
nice -n 10 uv pip install --python "$WCN_CLIENT_PY" \
    numpy pandas pyarrow httpx scipy openai pydantic anyio jinja2 huggingface_hub \
    "docent-python==0.1.75" "transformers==5.16.1"

# 7. Checks. No GPU here, so these check versions and files only; the GPU checks come in the
#    first GPU job. Any failure stops the script with STEP 1 FAILED in the log.
source "$SETUP_DIR/env.sh"     # again, now that the CUDA 12.9 ptxas exists
echo "=== checks"
"$WCN_GPU_PY" - <<'PY'
import importlib.metadata as md, pathlib, torch
v = {p: md.version(p) for p in ["vllm", "torch", "transformers", "triton", "flashinfer-python", "nvidia-cuda-nvcc-cu12"]}
print("gpu env:", v, "| torch CUDA build:", torch.version.cuda)
assert v["vllm"] == "0.30.0+cu129", "expected the CUDA 12.9 build of vLLM 0.30.0"
assert v["torch"] == "2.13.0+cu129" and torch.version.cuda == "12.9", "expected torch 2.13.0 for CUDA 12.9"
assert v["transformers"] == "5.16.1"
so = pathlib.Path(md.distribution("vllm").locate_file("vllm/_C_stable_libtorch.abi3.so"))
b = so.read_bytes()
assert b"libcudart.so.12" in b and b"libcudart.so.13" not in b, "vLLM binary is not the CUDA 12 build"
print("gpu env: vLLM binary links the CUDA 12 runtime: OK")
PY
"$WCN_GPU_PY" - <<'PY'
import os
from triton import knobs
t = knobs.nvidia.ptxas_blackwell
print("Triton ptxas for B200:", t.path, "version", t.version)
assert t.version == "12.9" and t.path == os.environ.get("TRITON_PTXAS_BLACKWELL_PATH"), "Triton is not using the CUDA 12.9 ptxas"
PY
"$WCN_ATTRIB_PY" - <<'PY'
import importlib.metadata as md, importlib.util, sys, torch
v = {p: md.version(p) for p in ["torch", "transformers", "accelerate", "flash-linear-attention", "fla-core", "triton"]}
print("attrib env: Python", sys.version.split()[0], v, "| torch CUDA build:", torch.version.cuda)
assert sys.version_info[:2] == (3, 13)
assert v["torch"] == "2.13.0+cu129" and torch.version.cuda == "12.9"
assert v["transformers"] == "5.14.1" and v["fla-core"] == "0.5.2"
assert v["triton"] == "3.7.1", "the repo's PHASE1.md:78 says Triton 3.7.1"
assert importlib.util.find_spec("causal_conv1d") is None, "causal-conv1d should not be installed (the original logs show it absent)"
print("attrib env: causal-conv1d absent, as in the original logs: OK")
print("NOTE: whether transformers actually uses fla can only be checked on a GPU node (first GPU job)")
PY
"$WCN_CLIENT_PY" - <<'PY'
import importlib.metadata as md
from docent.data_models.chat import AssistantMessage, ChatMessage, SystemMessage, UserMessage   # as code/wcref/judge.py:15
from docent.data_models.transcript import Transcript                                         # as code/wcref/judge.py:16
from openai import AsyncOpenAI                                                                # as code/wcref/judge.py:17
v = {p: md.version(p) for p in ["transformers", "huggingface_hub", "pandas", "pyarrow", "docent-python", "docent", "openai", "pydantic"]}
print("client env:", v)
assert v["docent-python"] == "0.1.75" and v["transformers"] == "5.16.1"
print("client env: judge imports OK")
PY

# 8. Record exactly what got installed, for the write-up
for e in gpu attrib client; do uv pip freeze --python "$WCN_BASE/envs/$e/bin/python" > "$WCN_BASE/logs/freeze_$e.txt"; done
echo "package lists saved: $WCN_BASE/logs/freeze_{gpu,attrib,client}.txt"
du -sh "$WCN_BASE/envs" "$WCN_BASE/python" "$UV_CACHE_DIR" "$WCN_BASE/wheels" 2>/dev/null || true
echo "STEP 1 DONE $(date)"
