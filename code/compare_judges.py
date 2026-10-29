"""Stage A base rates under the production (thinking-on) judge vs the thinking-off judge, on the same
samples (the first --per samples of each prompt), and both vs WeirdChat's published FP8 rates.
Separates judge leniency (off vs on on identical text) from model/serving differences (on vs WC).

  python compare_judges.py
"""
import numpy as np
from scipy.stats import spearmanr

from common import RES, JsonlStore, load_prompts

on = JsonlStore(RES / "stage_a_judge_on.jsonl").done
off = JsonlStore(RES / "stage_a_judge_off.jsonl").done
pr = load_prompts().set_index("prompt_id")
rows = {}
for k, r in on.items():
    if r["match"] is None or k not in off or off[k]["match"] is None:
        continue
    rows.setdefault(r["prompt_id"], []).append((off[k]["match"], r["match"]))
both = np.array([x for v in rows.values() for x in v])
po = (both[:, 0] == both[:, 1]).mean()
pe = both[:, 0].mean() * both[:, 1].mean() + (1 - both[:, 0].mean()) * (1 - both[:, 1].mean())
print(f"{len(both)} samples from {len(rows)} prompts; off-vs-on agreement {po:.3f}, kappa {(po - pe) / (1 - pe):.3f}; "
      f"P(on | off) {both[both[:, 0] == 1, 1].mean():.3f}  P(off | on) {both[both[:, 1] == 1, 0].mean():.3f}")
pid = list(rows)
r_off = np.array([np.mean([a for a, _ in rows[p]]) for p in pid])
r_on = np.array([np.mean([b for _, b in rows[p]]) for p in pid])
wc = np.array([pr.loc[p, "wc_k"] / pr.loc[p, "wc_n"] for p in pid])
print(f"pooled rate: off {r_off.mean():.3f}  on {r_on.mean():.3f}  WeirdChat {wc.mean():.3f}; "
      f"spearman per prompt: on~WC {spearmanr(r_on, wc)[0]:.2f}  off~WC {spearmanr(r_off, wc)[0]:.2f}  on~off {spearmanr(r_on, r_off)[0]:.2f}")
print(f"\n{'behavior':30s} {'n':>3s} {'off':>6s} {'on':>6s} {'WC':>6s}")
b = np.array([pr.loc[p, "behavior_id"] for p in pid])
for x in sorted(set(b)):
    m = b == x
    print(f"{x:30s} {m.sum():3d} {r_off[m].mean():6.3f} {r_on[m].mean():6.3f} {wc[m].mean():6.3f}")
