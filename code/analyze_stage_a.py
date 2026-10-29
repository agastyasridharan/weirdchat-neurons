"""Stage A summary: bf16 base rates vs WeirdChat's FP8 rates, per prompt and per behavior.

  python analyze_stage_a.py {on|off|rfoff}
Writes results/stage_a_rates_<tag>.json (per-prompt k/n) used downstream.
"""
import json
import sys

import numpy as np
from scipy.stats import spearmanr

from common import RES, JsonlStore, get_split, load_prompts

tag = sys.argv[1]
pr = load_prompts().set_index("prompt_id")
split = get_split()
js = JsonlStore(RES / f"stage_a_judge_{tag}.jsonl").done
k, n = {}, {}
for d in js.values():
    if d["match"] is None:
        continue
    k[d["prompt_id"]] = k.get(d["prompt_id"], 0) + int(d["match"])
    n[d["prompt_id"]] = n.get(d["prompt_id"], 0) + 1
rows = []
for pid in n:
    r = pr.loc[pid]
    rows.append({"prompt_id": pid, "behavior_id": r.behavior_id, "split": split[pid]["split"],
                 "k": k[pid], "n": n[pid], "rate": k[pid] / n[pid], "wc_rate": r.wc_k / r.wc_n})
(RES / f"stage_a_rates_{tag}.json").write_text(json.dumps(rows))
rate = np.array([x["rate"] for x in rows]); wc = np.array([x["wc_rate"] for x in rows])
print(f"prompts judged: {len(rows)}  pooled bf16 rate {rate.mean():.3f}  WeirdChat FP8 rate {wc.mean():.3f}  "
      f"spearman {spearmanr(rate, wc)[0]:.2f}")
print(f"zero-rate prompts: {(rate == 0).mean():.2f}   pass drop rule (k>=2): {np.mean([x['k'] >= 2 for x in rows]):.2f}")
print(f"{'behavior':30s} {'n':>4s} {'bf16':>6s} {'wc':>6s} {'pass':>5s} {'disc_pass':>9s}")
for b in sorted({x["behavior_id"] for x in rows}):
    xs = [x for x in rows if x["behavior_id"] == b]
    print(f"{b:30s} {len(xs):4d} {np.mean([x['rate'] for x in xs]):6.3f} {np.mean([x['wc_rate'] for x in xs]):6.3f} "
          f"{sum(x['k'] >= 2 for x in xs):5d} {sum(x['k'] >= 2 and x['split'] == 'discovery' for x in xs):9d}")
