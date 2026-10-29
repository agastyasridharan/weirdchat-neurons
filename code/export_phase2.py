"""Every Phase 2 number on the results page -> results/site_phase2.json (cenv; needs numpy only).

  python export_phase2.py
"""
import json
import re
from collections import Counter, defaultdict

import numpy as np

from attrib import KS, NI
from common import RES, JsonlStore
from rate import B5, N, conditions, prompts

D = RES / "attrib"


def fork_part(F):
    rows, single, layers, tops = [], [], Counter(), {}
    for i, f in enumerate(F):
        v = json.loads((D / f"{i}.verify.json").read_text())
        z = np.load(D / f"{i}.npz")
        A, A_last = z["A"].ravel(), z["A_last"]
        half = v["V0"] / 2
        rows.append({"b": f["behavior_id"], "tok": f["fork_token_str"], "V0": v["V0"],
                     "k_half": next((s["k"] for s in v["sets"] if s["V"] <= half), None),
                     "V_top": [s["V"] for s in v["sets"]], "V_rand": [float(np.mean(s["V_rand"])) for s in v["sets"]],
                     "last_share": float(A_last.clip(0).sum() / A.clip(0).sum())})
        single += [[round(-s["A"], 4), round(s["dM"], 4)] for s in v["single"]]
        top = np.argsort(-A)[:100]
        tops[i] = set(top.tolist())
        layers.update((top // NI).tolist())
    jac = []
    for b in sorted({f["behavior_id"] for f in F}):
        ix = [i for i, f in enumerate(F) if f["behavior_id"] == b]
        if len(ix) < 2:
            continue
        jw = [len(tops[x] & tops[y]) / len(tops[x] | tops[y]) for n, x in enumerate(ix) for y in ix[n + 1:]]
        jx = [len(tops[x] & tops[y]) / len(tops[x] | tops[y]) for x in ix for y in tops if F[y]["behavior_id"] != b]
        jac.append({"b": b, "n": len(ix), "within": float(np.mean(jw)), "across": float(np.mean(jx))})
    p, a = np.array(single).T
    return {"ks": KS, "rows": rows, "single": single, "single_r": float(np.corrcoef(p, a)[0, 1]),
            "single_ratio": float(np.median(a / p)), "layers": [layers.get(l, 0) for l in range(64)], "jaccard": jac}


def rate_part(F):
    P = prompts()
    C = conditions()
    gs, js = JsonlStore(RES / "rate_gen.jsonl").done, JsonlStore(RES / "rate_judge.jsonl").done
    cs = JsonlStore(RES / "rate_cond.jsonl").done
    sets = json.loads((RES / "ablation_sets.json").read_text())
    rng = np.random.default_rng(0)

    def per(c, b, split=None):
        out = []
        for p in P:
            if p["b"] == b and (split is None or p["split"] == split):
                m = [js[f"{c}|{p['pid']}|{i}"]["match"] for i in range(N)]
                m = [x for x in m if x is not None]
                out.append((sum(m), len(m)))
        return np.array(out)

    def r(a):
        return float(a[:, 0].sum() / a[:, 1].sum())

    base_nll, out = np.array(cs["base"]["nll"]), []
    for c in C:
        row = {"c": c, "rates": {b: r(per(c, b)) for b in B5}, "n": {b: int(per(c, b)[:, 1].sum()) for b in B5}}
        ntok = np.array(cs[c]["ntok"])
        row["dnll"] = float((np.array(cs[c]["nll"]) - base_nll).sum() / ntok.sum())
        row["len"] = float(np.mean([s["n"] for p in P for s in gs[f"{c}|{p['pid']}"]["samples"]]))
        if c in sets:
            s = sets[c]
            a, z = per(c, s["b"]), per("base", s["b"])
            bs = []
            for _ in range(2000):
                ix = rng.integers(len(a), size=len(a))
                if z[ix, 0].sum() > 0:
                    bs.append(r(a[ix]) / r(z[ix]))
            row.update(b=s["b"], k=s["k"], kind=s["kind"], ratio=r(a) / r(z), lo=float(np.percentile(bs, 2.5)),
                       hi=float(np.percentile(bs, 97.5)), held=r(per(c, s["b"], "heldout")), held_base=r(per("base", s["b"], "heldout")),
                       disc=r(per(c, s["b"], "discovery")), disc_base=r(per("base", s["b"], "discovery")))
            v = [(cs[c]["V"][i], cs["base"]["V"][i]) for i in s["forks"] if cs[c]["V"][i] is not None and cs["base"]["V"][i] is not None]
            row.update(V_own=float(np.mean([x for x, _ in v])), V_own_base=float(np.mean([y for _, y in v])))
        out.append(row)
    pats = {"I": r"\bI\b", "my": r"\b[Mm]y\b"}

    def freq(c):
        t = [s["text"] for p in P for s in gs[f"{c}|{p['pid']}"]["samples"]]
        w = sum(len(x.split()) for x in t)
        return {k: 1000 * sum(len(re.findall(v, x)) for x in t) / w for k, v in pats.items()}
    counts = Counter(p["b"] for p in P)
    splits = {b: Counter(p["split"] for p in P if p["b"] == b) for b in B5}
    return {"behaviors": B5, "rows": out, "n_prompts": len(P), "per_prompt": N, "prompts": counts, "splits": splits,
            "judged": len(js), "freq": {c: freq(c) for c in ["base", "fabricated-code-execution|top100", "fabricated-code-execution|rand100"]}}


def main():
    F = [json.loads(l) for l in (RES / "phase2_forks.jsonl").read_text().splitlines()]
    d = {"fork": fork_part(F), "rate": rate_part(F)}
    (RES / "site_phase2.json").write_text(json.dumps(d))
    print("wrote site_phase2.json", len(json.dumps(d)), "bytes")


if __name__ == "__main__":
    main()
