"""Phase 2d inputs: upstream neuron and head sets per behavior (genv; reads results/attrib/ and results/upstream/).

For behavior b with forks F_b, each score sums per-fork attributions scaled so every fork counts equally (as sets.py):
  M-score of a neuron (attrib A), T-score of a neuron (upstream AT: acts through the late writers), H-score of a head.
Sets (layers below LATE = 48 unless noted):
  late       the top-100 set's neurons in layers >= 48 (the writers)
  early      the top-100 set's neurons below 48
  mid100     top 100 by M-score below 48 (all paths to M)
  med100     top 100 by T-score below 48 (paths through the writers)
  randmed100 random neurons matched to med100 on layer and sum|a| rank
  head10     top 10 attention heads by H-score, any layer (columns of o_proj / out_proj)
  randhead10 one random other head in the same layer for each

  python upsets.py   -> results/upstream_sets.json
"""
import json
from itertools import combinations

import numpy as np

from attrib import NI, NL, matched
from common import RES
from sets import B5, spec
from upstream import LATE, NH


def norm(x, k):
    return x / np.sort(x.ravel())[::-1][:k].sum()


def main():
    F = [json.loads(l) for l in (RES / "phase2_forks.jsonl").read_text().splitlines()]
    old = json.loads((RES / "ablation_sets.json").read_text())
    sets, rng = {}, np.random.default_rng(3)
    for b in B5:
        ix = [i for i, f in enumerate(F) if f["behavior_id"] == b]
        sM, sT, sH, absa = np.zeros((NL, NI)), np.zeros((NL, NI)), np.zeros((NL, NH)), np.zeros((NL, NI))
        for i in ix:
            a, u = np.load(RES / "attrib" / f"{i}.npz"), np.load(RES / "upstream" / f"{i}.npz")
            sM += norm(a["A"].astype(np.float64), 100)
            sT += norm(u["AT"].astype(np.float64), 100)
            sH += norm(np.nan_to_num(u["H"].astype(np.float64), nan=-np.inf), 10)
            absa += a["absa"] / len(ix)
        sM[LATE:], sT[LATE:] = -np.inf, -np.inf
        sH[~np.isfinite(sH)] = -np.inf
        top = old[f"{b}|top100"]["spec"]
        late = {l: v for l, v in top.items() if int(l) >= LATE}
        early = {l: v for l, v in top.items() if int(l) < LATE}
        mid = np.argsort(-sM.ravel())[:100]
        med = np.argsort(-sT.ravel())[:100]
        heads = np.argsort(-sH.ravel())[:10]
        rh = {}
        for h in heads:
            l, j = divmod(int(h), NH)
            nh = int(np.isfinite(sH[l]).sum())
            pool = [x for x in range(nh) if x != j and x not in rh.get(l, []) and l * NH + x not in set(heads.tolist())]
            rh.setdefault(l, []).append(int(rng.choice(pool)))
        hs = {}
        for h in heads:
            hs.setdefault(str(int(h) // NH), []).append(int(h) % NH)
        base = {"b": b, "forks": ix}
        sets[f"{b}|late"] = base | {"kind": "late", "spec": late}
        sets[f"{b}|early"] = base | {"kind": "early", "spec": early}
        sets[f"{b}|mid100"] = base | {"kind": "mid100", "spec": spec(mid)}
        sets[f"{b}|med100"] = base | {"kind": "med100", "spec": spec(med)}
        sets[f"{b}|randmed100"] = base | {"kind": "randmed100", "spec": spec(matched(med, absa, rng))}
        sets[f"{b}|head10"] = base | {"kind": "head10", "heads": hs}
        sets[f"{b}|randhead10"] = base | {"kind": "randhead10", "heads": {str(l): v for l, v in rh.items()}}
        lay = lambda s: np.bincount([int(l) // 8 for l, v in s.items() for _ in v], minlength=8).tolist()
        n = lambda s: sum(len(v) for v in s.values())
        fl = lambda s: {int(l) * NI + i for l, v in s.items() for i in v}
        print(f"{b:28s} forks {len(ix):2d}  late {n(late)} early {n(early)}; |mid100 & med100| {len(set(mid) & set(med))}, "
              f"|early & med100| {len(fl(early) & set(med.tolist()))}; med100 layers/8 {lay(spec(med))}")
        print("   top heads (layer.head, score, type): " + ", ".join(
            f"{int(h) // NH}.{int(h) % NH} {sH.ravel()[h]:.2f}{'F' if (int(h) // NH) % 4 == 3 else 'L'}" for h in heads))
    for kind in ["med100", "mid100", "head10"]:
        key = "heads" if kind == "head10" else "spec"
        S = {b: {(int(l), i) for l, v in sets[f"{b}|{kind}"][key].items() for i in v} for b in B5}
        print(f"{kind} cross-behavior overlap: " + ", ".join(f"{a[:6]}/{c[:6]} {len(S[a] & S[c])}" for a, c in combinations(B5, 2)))
    (RES / "upstream_sets.json").write_text(json.dumps(sets))
    print(f"{len(sets)} sets")


if __name__ == "__main__":
    main()
