"""Summaries of Phase 2a (attrib.py) outputs. Run with genv python (needs numpy).

  python analyze_attrib.py
"""
import json
from collections import Counter, defaultdict

import numpy as np

from common import RES

NI = 17408
D = RES / "attrib"
F = [json.loads(l) for l in (RES / "phase2_forks.jsonl").read_text().splitlines()]


def load(i):
    z = np.load(D / f"{i}.npz")
    return {k: z[k] for k in ("A", "A_last", "absa")}, json.loads((D / f"{i}.verify.json").read_text())


def main():
    rows, tops, by_b = [], {}, defaultdict(list)
    pred, act = [], []
    layer_hist = Counter()
    for i, f in enumerate(F):
        a, v = load(i)
        A = a["A"].ravel()
        floor = max(0.02, sum(min(max(x["q"], .02), .98) * x["p"] for x in f["next_tokens"] if x["tok"] != f["fork_token"])
                    / max(1e-9, sum(x["p"] for x in f["next_tokens"] if x["tok"] != f["fork_token"])))
        half = v["V0"] / 2
        k_half = next((s["k"] for s in v["sets"] if s["V"] <= half), None)
        k_half_r = next((s["k"] for s in v["sets"] if max(s["V_rand"]) <= half), None)
        pos = np.sort(A[A > 0])[::-1]
        rows.append(dict(i=i, b=f["behavior_id"], tok=f["fork_token_str"], V0=v["V0"], k_half=k_half, k_half_r=k_half_r,
                         V10=v["sets"][2]["V"], V10r=np.mean(v["sets"][2]["V_rand"]), floor=floor,
                         top10_share=pos[:10].sum() / pos.sum(), last_share=a["A_last"].clip(0).sum() / a["A"].clip(0).sum()))
        for s in v["single"]:
            pred.append(-s["A"]); act.append(s["dM"])
        top = np.argsort(-A)[:100]
        tops[i] = set(top.tolist())
        by_b[f["behavior_id"]].append(i)
        layer_hist.update((top // NI).tolist())
    print(f"single-neuron zero ablation: corr(predicted dM, actual dM) = {np.corrcoef(pred, act)[0, 1]:.3f} "
          f"(n={len(pred)}); median actual/predicted = {np.median(np.array(act) / np.array(pred)):.2f}")
    print("\nper fork: V0, k (top-attribution neurons zeroed) to halve V, same for matched random (None = never <= 10000)")
    for r in sorted(rows, key=lambda r: (r["b"], r["i"])):
        print(f"  {r['i']:2d} {r['b'][:26]:26s} {r['tok']!r:12s} V0 {r['V0']:.2f} floor {r['floor']:.2f}  k_half {str(r['k_half']):>5s}  "
              f"rand {str(r['k_half_r']):>5s}  V@10 {r['V10']:.2f} (rand {r['V10r']:.2f})  top10 share {r['top10_share']:.2f}  "
              f"last-pos share {r['last_share']:.2f}")
    kh = [r["k_half"] for r in rows]
    print(f"\nk_half distribution: {Counter(kh)}; random never halves at k<=300 in "
          f"{sum(r['k_half_r'] is None or r['k_half_r'] > 300 for r in rows)}/{len(rows)}")
    print(f"median share of positive attribution at the last position: {np.median([r['last_share'] for r in rows]):.2f}")
    print("\nlayers of top-100 neurons (all forks):", " ".join(f"{l}:{n}" for l, n in sorted(layer_hist.items())))
    print("\noverlap of top-100 sets: mean Jaccard within behavior vs across behaviors; neurons in top-100 of >= half the forks")
    for b, ix in sorted(by_b.items()):
        if len(ix) < 2:
            continue
        jw = [len(tops[x] & tops[y]) / len(tops[x] | tops[y]) for n, x in enumerate(ix) for y in ix[n + 1:]]
        jx = [len(tops[x] & tops[y]) / len(tops[x] | tops[y]) for x in ix for y in tops if F[y]["behavior_id"] != b]
        c = Counter(n for x in ix for n in tops[x])
        common = [divmod(n, NI) for n, m in c.most_common() if m >= len(ix) / 2]
        print(f"  {b[:28]:28s} n {len(ix):2d}  within {np.mean(jw):.3f}  across {np.mean(jx):.3f}  shared {len(common)} "
              f"{common[:8]}")


if __name__ == "__main__":
    main()
