"""Phase 3b: neurons shared across behaviors, from the reply-level contrasts of contrast.py (genv).

Per prompt p: C_p = mean G(behavior replies) - mean G(other replies) (contrast.py), scaled by s_p = the sum of its 100
largest entries so every prompt counts equally. Per behavior b: S_b = mean over its prompts of C_p / s_p, then
z_b = S_b / (sum of the 100 largest entries of S_b) so every behavior counts equally. Behaviors with >= MINP prompts
enter the shared score:
  shared(n) = median over those behaviors of z_b(n)   (> 0 only if the unit pushes at least half of them)
Nulls and controls:
  perm    the same pipeline with behavior labels shuffled within each prompt (NPERM times): how much sharing chance
          (and units that move every reply) produce
  rare    a contrast with no behavior in it: the less likely half of a prompt's non-behavior replies minus the more
          likely half. Units shared across behaviors that also score here make unlikely text likelier in general
          (temperature-like), not any behavior in particular.
Sets (100 neurons each, matched random controls on layer and sum|a| rank):
  shared100 / sharedmid100 (layers < 48) / rare100 / loo100|b (shared score without behavior b, for b in the score)
  loomid100|b: the same below layer 48
  selfpair100: top 100 by min(z_denying-ai-identity, z_false-physical-embodiment), the one pair whose maps overlap
               beyond the permutation null (the self-model link Phase 2d found)

  python sets3.py   -> results/shared_sets.json, results/shared_stats.json, results/shared_z.npz (z_b per behavior,
                       shared, rare and pair scores over all 64 x 17408 neurons, flattened)
"""
import json
from collections import defaultdict
from itertools import combinations

import numpy as np

from attrib import NI, NL, matched
from common import RES, log
from contrast import OUT, unbits
from sets import spec
from upstream import LATE, NH

MINP, NPERM, K = 8, 10, 100


def topsum(x, k=K):
    return np.partition(x.ravel(), -k)[-k:].sum()


def main():
    rng = np.random.default_rng(7)
    S, Sperm, Srare, SH = defaultdict(lambda: 0), defaultdict(lambda: 0), defaultdict(lambda: 0), defaultdict(lambda: 0)
    npr, absa, Lgap = defaultdict(int), defaultdict(lambda: 0), defaultdict(list)
    for f in sorted(OUT.glob("*.npz")):
        z = np.load(f)
        b, lab, L = str(z["b"]), z["lab"], z["L"]
        G = unbits(z["G"]).reshape(len(lab), -1)
        C = G[lab == 1].mean(0) - G[lab == 0].mean(0)
        s = topsum(C)
        S[b] = S[b] + C / s
        H = z["GH"].reshape(len(lab), -1)
        SH[b] = SH[b] + (H[lab == 1].mean(0) - H[lab == 0].mean(0)) / s
        perms = []
        for _ in range(NPERM):
            pl = rng.permutation(lab)
            while (pl == lab).all():
                pl = rng.permutation(lab)
            Cp = G[pl == 1].mean(0) - G[pl == 0].mean(0)
            perms.append(Cp / topsum(Cp))
        Sperm[b] = Sperm[b] + np.stack(perms)
        neg = np.where(lab == 0)[0]
        neg = neg[np.argsort(L[neg])]
        h = len(neg) // 2
        R = G[neg[:h]].mean(0) - G[neg[-h:]].mean(0)
        Srare[b] = Srare[b] + R / topsum(R)
        absa[b] = absa[b] + unbits(z["absa"]).reshape(NL, NI)
        npr[b] += 1
        Lgap[b].append(float(L[lab == 1].mean() - L[lab == 0].mean()))
    B = sorted(npr)
    E = [b for b in B if npr[b] >= MINP]
    log("prompts per behavior: " + ", ".join(f"{b} {npr[b]}" for b in B) + f"; in the shared score: {len(E)}")
    z = {b: S[b] / npr[b] for b in B}
    z = {b: v / topsum(v) for b, v in z.items()}
    zp = {b: Sperm[b] / npr[b] for b in B}
    zp = {b: v / np.partition(v, -K, axis=1)[:, -K:].sum(1, keepdims=True) for b, v in zp.items()}
    rare = sum(Srare[b] for b in E) / sum(npr[b] for b in E)
    rare = rare / topsum(rare)
    zh = {b: SH[b] / npr[b] for b in B}
    ab = sum(absa[b] for b in B) / sum(npr.values())

    stats = {"n_prompts": dict(npr), "eligible": E, "logp_gap": {b: float(np.mean(v)) for b, v in Lgap.items()}}
    # 1. similarity of behaviors' attribution maps (all neurons, and below LATE), vs the permutation null
    cos = lambda a, c: float(a @ c / np.linalg.norm(a) / np.linalg.norm(c))
    mid = np.arange(NL * NI) < LATE * NI
    stats["cos"] = {f"{a}|{c}": [cos(z[a], z[c]), cos(z[a][mid], z[c][mid]),
                                 float(np.mean([cos(zp[a][k], zp[c][k]) for k in range(NPERM)]))]
                    for a, c in combinations(E, 2)}
    stats["cos_rare"] = {b: cos(z[b], rare) for b in B}
    # 2. how many neurons are among the top 1000 of >= m behaviors, real vs permuted labels
    def counts(Z):
        hit = np.zeros(NL * NI, np.int16)
        for v in Z:
            hit[np.argpartition(v, -1000)[-1000:]] += 1
        return [int((hit >= m).sum()) for m in range(1, len(Z) + 1)]
    stats["top1000_shared"] = counts([z[b] for b in E])
    stats["top1000_shared_perm"] = np.mean([counts([zp[b][k] for b in E]) for k in range(NPERM)], 0).tolist()
    stats["top1000_shared_rare_removed"] = counts([np.where(rare >= np.sort(rare)[-1000], -np.inf, z[b]) for b in E])

    shared = np.median(np.stack([z[b] for b in E]), 0)
    sharedp = [np.median(np.stack([zp[b][k] for b in E]), 0) for k in range(NPERM)]
    stats["shared_top100_sum"] = float(np.sort(shared)[-K:].sum())
    stats["shared_top100_sum_perm"] = [float(np.sort(x)[-K:].sum()) for x in sharedp]
    sharedmid = np.where(mid, shared, -np.inf)
    sets = {}

    def add(name, order, **kw):
        sel = np.asarray(order[:K])
        sets[name] = {"kind": name.split("|")[0], "spec": spec(sel), **kw}
        sets[f"rand_{name}"] = {"kind": "rand_" + name.split("|")[0], "spec": spec(matched(sel, ab, rng)), **kw}
        return sel
    s100 = add("shared100", np.argsort(-shared))
    smid = add("sharedmid100", np.argsort(-sharedmid))
    r100 = add("rare100", np.argsort(-rare))
    for b in E:
        add(f"loo100|{b}", np.argsort(-np.median(np.stack([z[c] for c in E if c != b]), 0)), b=b)
    pair = np.minimum(z["denying-ai-identity"], z["false-physical-embodiment"])
    sp100 = add("selfpair100", np.argsort(-pair))
    stats["selfpair_support"] = {b: float(z[b][sp100].sum()) for b in B}
    print("selfpair100 (min of denying-ai-identity and false-physical-embodiment z) support:",
          {b[:12]: round(v, 3) for b, v in stats["selfpair_support"].items()}, "layers/8",
          np.bincount(sp100 // NI // 8, minlength=8).tolist())
    for b in E:  # appended after the other sets so their random controls stay as first drawn
        add(f"loomid100|{b}", np.argsort(-np.where(mid, np.median(np.stack([z[c] for c in E if c != b]), 0), -np.inf)), b=b)
    lay = lambda s: np.bincount(s // NI // 8, minlength=8).tolist()
    stats["layers_per8"] = {"shared100": lay(s100), "sharedmid100": lay(smid), "rare100": lay(r100)}
    stats["overlap"] = {"shared100&rare100": len(set(s100) & set(r100)), "sharedmid100&rare100": len(set(smid) & set(r100)),
                        **{f"shared100&loo|{b}": len(set(s100) & {int(l) * NI + i for l, v in sets[f"loo100|{b}"]["spec"].items() for i in v}) for b in E}}
    # per-behavior support of the shared set: share of the set's summed z_b each behavior gives, and its rank there
    stats["support"] = {b: {"z_sum": float(z[b][s100].sum()), "z_sum_mid": float(z[b][smid].sum()),
                            "median_rank": int(np.median(np.argsort(np.argsort(-z[b]))[s100]))} for b in B}
    stats["top_neurons"] = [{"n": [int(x) // NI, int(x) % NI], "shared": float(shared[x]), "rare": float(rare[x]),
                             "z": {b: float(z[b][x]) for b in B}} for x in s100[:30]]
    hs = np.median(np.stack([zh[b] for b in E]), 0)
    stats["top_heads"] = [{"h": [int(x) // NH, int(x) % NH], "shared": float(hs[x]),
                           "z": {b: float(zh[b][x]) for b in B}} for x in np.argsort(-hs)[:15]]
    np.savez(RES / "shared_z.npz", shared=shared.astype(np.float32), rare=rare.astype(np.float32), pair=pair.astype(np.float32),
             **{f"z|{b}": z[b].astype(np.float32) for b in B})
    (RES / "shared_sets.json").write_text(json.dumps(sets))
    (RES / "shared_stats.json").write_text(json.dumps(stats))
    print("log p gap (behavior - other replies):", {b[:12]: round(v, 3) for b, v in stats["logp_gap"].items()})
    print("cosine all / below 48 / permuted:")
    for k, v in stats["cos"].items():
        print(f"   {k.split('|')[0][:22]:22s} {k.split('|')[1][:22]:22s} {v[0]:+.3f} {v[1]:+.3f} {v[2]:+.3f}")
    print("cosine with the rareness contrast:", {b[:12]: round(v, 3) for b, v in stats["cos_rare"].items()})
    print("neurons in the top 1000 of >= m behaviors (m = 1..):", stats["top1000_shared"])
    print("   permuted labels:                               ", [round(x) for x in stats["top1000_shared_perm"]])
    print("   rareness top-1000 removed:                     ", stats["top1000_shared_rare_removed"])
    print(f"shared top-100 median-z sum {stats['shared_top100_sum']:.3f}; permuted {np.round(stats['shared_top100_sum_perm'], 3)}")
    print("layers/8:", stats["layers_per8"], "overlap:", stats["overlap"])
    print("support of shared100 (sum z_b, sum z_b for sharedmid100, median rank in z_b):")
    for b, v in stats["support"].items():
        print(f"   {b:30s} {v['z_sum']:+.3f} {v['z_sum_mid']:+.3f} {v['median_rank']}")
    print("top shared neurons:", [(t["n"], round(t["shared"], 4), round(t["rare"], 4)) for t in stats["top_neurons"][:12]])
    print("top shared heads:", [(f"{t['h'][0]}.{t['h'][1]}", round(t["shared"], 3)) for t in stats["top_heads"][:10]])
    print(len(sets), "sets")


if __name__ == "__main__":
    main()
