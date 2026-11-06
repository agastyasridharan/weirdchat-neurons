"""GPU job 2, part 1 (attribution env): run the repo's attribution exactly as code/attrib.py does (main(0, 1): every
fork, attribution then its verify ablations), without editing the repo's files. The only change is at run time: the model
path is hard-coded at code/common.py:24, so it is pointed at our download before code/attrib.py imports it
(code/attrib.py:21 copies it at import). Rerun-safe: the repo's main skips forks whose verify file exists."""
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.environ["WCN_ROOT"], "code"))
import common  # noqa: E402  (code/common.py)

common.QWEN_SNAP = json.load(open(os.path.join(os.environ["WCN_BASE"], "paths.json")))["qwen"]
import attrib  # noqa: E402  (code/attrib.py; reads QWEN_SNAP from common now)

assert attrib.QWEN_SNAP == common.QWEN_SNAP, "model path was not passed to the repo's attrib.py"
print("model path for the repo's attrib.py:", attrib.QWEN_SNAP, flush=True)
attrib.main(0, 1)
n = len(list(attrib.OUT.glob("*.verify.json")))
print(f"attribution files present for {n} of 46 forks", flush=True)
sys.exit(0 if n == 46 else 1)
