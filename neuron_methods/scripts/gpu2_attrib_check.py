"""GPU job 2, part 2 (client env): gate G2. Reads our results/attrib/ files and the original logs and set file; writes nothing.
 (a) M0 and V0 at the 45 forks the original logs record (logs/attrib_*.log and attrib2_*.log, rounded to 2 and 3 decimals there).
     Pass: median |V0 difference| <= 0.005 and none above 0.02 (set before the run).
 (b) The behavior sets, rebuilt from our files with code/sets.py logic (lines 33-46). The repo's sets.py itself is not
     run, because it would overwrite results/ablation_sets.json. Pass: each of the 5 behaviors' top-100 shares at
     least 95 neurons with results/ablation_sets.json. Top-10 and top-1000 overlaps are shown too.
 (c) Information: V after zeroing each fork's own top-k neurons (our verify files vs the original log lines).
"""
import glob
import json
import os
import re
from pathlib import Path

import numpy as np

P = Path(os.environ["WCN_ROOT"])
RES = P / "results"
NL, NI = 64, 17408                                   # code/attrib.py:24
B5 = ["claims-called-911", "denying-ai-identity", "fabricated-code-execution", "false-physical-embodiment",
      "laser-at-aircraft"]                           # code/sets.py:20-21
OK = []


def check(name, ok, detail=""):
    OK.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  [{detail}]", flush=True)


forks = [json.loads(line) for line in (RES / "phase2_forks.jsonl").read_text().splitlines() if line.strip()]
ours = {}
for i in range(len(forks)):
    z = np.load(RES / "attrib" / f"{i}.npz")
    assert str(z["key"]) == forks[i]["key"], f"fork {i}: file belongs to another fork"
    ours[i] = {"M0": float(z["M0"]), "V0": float(z["V0"]),
               "verify": json.loads((RES / "attrib" / f"{i}.verify.json").read_text())}

pat = re.compile(r"fork (\d+) (\S+) len (\d+): M0 (-?[\d.]+) V0 ([\d.]+); attrib \d+s verify \d+s; V after top-k (.*)$")
orig = {}
for fn in sorted(glob.glob(str(P / "logs" / "attrib_*.log")) + glob.glob(str(P / "logs" / "attrib2_*.log"))):
    for line in open(fn):
        m = pat.search(line.strip())
        if m:
            orig[int(m.group(1))] = {"M0": float(m.group(4)), "V0": float(m.group(5)),
                                    "topk": {int(k): float(v.split("/")[0]) for k, v in
                                             (t.split(":") for t in m.group(6).split())}}

# (a) M0 and V0
idx = sorted(orig)
dv = np.array([abs(ours[i]["V0"] - orig[i]["V0"]) for i in idx])
dm = np.array([abs(ours[i]["M0"] - orig[i]["M0"]) for i in idx])
worst = idx[int(dv.argmax())]
check(f"G2: V0 at the {len(idx)} logged forks: median |difference| <= 0.005 and none above 0.02",
      np.median(dv) <= 0.005 and dv.max() <= 0.02,
      f"median {np.median(dv):.4f}, largest {dv.max():.4f} (fork {worst}: ours {ours[worst]['V0']:.3f}, original {orig[worst]['V0']:.3f})")
print(f"info  M0 |difference|: median {np.median(dm):.3f}, largest {dm.max():.3f} (the original M0 is rounded to 0.01)")

# (b) behavior sets, code/sets.py:33-46 without writing anything
sets = json.loads((RES / "ablation_sets.json").read_text())


def flat(spec):
    return {int(l) * NI + int(i) for l, ns in spec.items() for i in ns}


for b in B5:
    ix = [i for i, f in enumerate(forks) if f["behavior_id"] == b]
    score = np.zeros(NL * NI)
    for i in ix:
        A = np.load(RES / "attrib" / f"{i}.npz")["A"].ravel().astype(np.float64)
        score += A / np.sort(A)[::-1][:100].sum()
    order = np.argsort(-score)
    ov = {k: len(set(order[:k].tolist()) & flat(sets[f"{b}|top{k}"]["spec"])) for k in (10, 100, 1000)}
    check(f"G2: {b} top-100 shares >= 95 neurons with the original", ov[100] >= 95,
          f"top-10 {ov[10]}/10, top-100 {ov[100]}/100, top-1000 {ov[1000]}/1000; {len(ix)} forks")

# (c) V after zeroing each fork's top-k (information)
diffs = []
for i in idx:
    ov_ = {s["k"]: s["V"] for s in ours[i]["verify"]["sets"]}
    diffs += [abs(ov_[k] - v) for k, v in orig[i]["topk"].items() if k in ov_]
diffs = np.array(diffs)
print(f"info  V after zeroing top-k (k = 1 to 10,000) vs the original log: median |difference| {np.median(diffs):.3f}, "
      f"95th percentile {np.percentile(diffs, 95):.3f}, largest {diffs.max():.3f} (the original values are rounded to 0.01)")

print(f"\n{sum(OK)} of {len(OK)} checks passed.")
print("GATE G2 PASS" if all(OK) else "GATE G2 FAIL.")
raise SystemExit(0 if all(OK) else 1)
