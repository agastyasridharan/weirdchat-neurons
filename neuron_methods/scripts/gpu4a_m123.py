"""GPU job 4a (attribution env, one B200): Step 3 for methods M1, M2 and M3, with matched random controls.

All sets come from the 46 discovery forks only; the size ladder uses the four behaviors with
29 forks (code execution 13, laser 8, false embodiment 5, 911 3). Sizes 3, 5, 10, 20, 50, 100, nested.
  M1  the repo's score (sets.py:37-43): per fork A / (sum of its 100 largest A), summed over the behavior's forks; top k.
  M2  the same score, top k among layers 0-47 only.
  M3  start from M1's top 3; then 5 times: zero the set, recompute A at every fork of the behavior with the set
      zeroed (the model hook, attrib.py:50-60), divide by each fork's normalizer from before any zeroing, and add the
      best neurons not in the set (explicit exclusion): +2, +5, +10, +30, +50, giving 5, 10, 20, 50, 100.
      M3's 3-neuron set is M1's by construction and is not listed again.
  C   one matched random set per method and behavior at size 100: the repo's attrib.matched with the behavior's mean
      sum|a| and np.random.default_rng(1) (sets.py:42-48).
Then V is measured at each set's own forks with the set zeroed (no gradient), a cheap check on the training data.

Checks, in order (each prints PASS or FAIL; outputs reach methods/ only if all pass):
  inputs: the 46 forks and GPU job 2's attribution files;
  M1 rebuilt from those files matches results/ablation_sets.json as G2 found (top 100: 95 or more shared);
  M2 uses layers 0-47 only, and its top 100 shares 95 or more with the repo's mid100 (code/upsets.py:42-49, the same
    score and layer cut, in results/upstream_sets.json); every set is nested and has distinct neurons;
  model: 64 layers of 17,408 neurons, 48 DeltaNet and 16 attention;
  the M3 driver with nothing zeroed reproduces GPU job 2's attribution at each of the 29 forks: V within 0.01 and
    95 or more of the top 100 shared (same code and GPU type, so the only expected difference is run-to-run noise;
    the limits are loose on purpose and catch only a broken driver: wrong prefix, tokens or ablation);
  M3 steps: A is exactly 0 at every zeroed neuron, all 64 layers are seen, values are finite, new neurons are new;
  controls: same layers one for one, disjoint from the set, distinct;
  bug check: M1's 100-neuron set lowers mean V at its own forks for every behavior (the original top-100 sets cut V at most
    forks, PHASE2.md:16-23).
"""
import sys
import time

sys.dont_write_bytecode = True
import gpu4_lib as G  # noqa: E402
import numpy as np  # noqa: E402

OUT_NAMES = ["m123_sets.json", "m123_summary.json", "m3_steps.json"]
M3_STEPS = [2, 5, 10, 30, 50]


def own_fork_V(M, F, ids, ix, flat):
    return {int(i): G.V_at(M, F[i], ids[i], flat) for i in ix}


def main():
    t_start = time.time()
    out = G.Outputs("4a", OUT_NAMES)
    G.stage("Inputs")
    F, ids, by_b = G.load_forks()
    Z = G.load_attrib(F)
    ladder_ix = {b: by_b[b] for b in G.LADDER}
    norms = {i: G.normalizer(Z[i]["A"]) for b in G.LADDER for i in ladder_ix[b]}
    G.check("inputs: every ladder fork's normalizer (sum of its 100 largest A) is positive",
            all(v > 0 for v in norms.values()), f"smallest {min(norms.values()):.4g}")

    G.stage("M1 and M2 from GPU job 2's attribution files")
    orig = __import__("json").loads((G.RES / "ablation_sets.json").read_text())
    # The repo's mid100 (code/upsets.py:42-49: the same score with layers 48 and up set to -inf, top 100) is our M2 at 100.
    orig_up = __import__("json").loads((G.RES / "upstream_sets.json").read_text())
    order = {}
    for b in G.LADDER:
        ix = ladder_ix[b]
        score = G.behavior_score([Z[i]["A"] for i in ix], [norms[i] for i in ix])
        order[("M1", b)] = G.top_order(score, 100)
        order[("M2", b)] = G.top_order(score, 100, layer_cut=G.LAYER_CUT)
        o10 = len(set(order[("M1", b)][:10]) & G.flat_of(orig[f"{b}|top10"]["spec"]))
        o100 = len(set(order[("M1", b)]) & G.flat_of(orig[f"{b}|top100"]["spec"]))
        G.check(f"M1 {b}: rebuilt top 100 shares 95 or more with the repo's ablation_sets.json (G2 found 97 to 99)",
                o100 >= 95, f"top 10: {o10}/10, top 100: {o100}/100")
        hi = sum(x // G.NI >= G.LAYER_CUT for x in order[("M1", b)])
        G.check(f"M2 {b}: all 100 neurons in layers 0-47", all(x // G.NI < G.LAYER_CUT for x in order[("M2", b)]),
                f"layers {min(x // G.NI for x in order[('M2', b)])} to {max(x // G.NI for x in order[('M2', b)])}; "
                f"M1's top 100 has {hi} in layers 48-63; M1 and M2 share {len(set(order[('M1', b)]) & set(order[('M2', b)]))}")
        m2 = len(set(order[("M2", b)]) & G.flat_of(orig_up[f"{b}|mid100"]["spec"]))
        G.check(f"M2 {b}: shares 95 or more with the repo's mid100 (results/upstream_sets.json), the same set built by the repo's code",
                m2 >= 95, f"{m2}/100")

    G.stage("Model")
    M = G.attrib.Model()
    G.check_model(M)

    G.stage("M3 driver check: nothing zeroed must reproduce GPU job 2 at all 29 ladder forks (longest first)")
    all_ix = sorted((i for b in G.LADDER for i in ladder_ix[b]), key=lambda i: -len(ids[i]))
    drv = {}
    for i in all_ix:
        t0 = time.time()
        A, m, V, nl = G.attrib_at(M, F[i], ids[i], [])
        top_new = set(np.argsort(-A.ravel())[:100].tolist())
        top_old = set(np.argsort(-Z[i]["A"].ravel())[:100].tolist())
        drv[i] = dict(dV=abs(V - float(Z[i]["V0"])), overlap=len(top_new & top_old), layers=nl,
                      identical=bool(np.array_equal(A, Z[i]["A"])), V=V, s=round(time.time() - t0, 1))
        print(f"  fork {i:2d} {F[i]['behavior_id'][:26]:26s} len {len(ids[i]):4d}: V {V:.4f} (job 2: "
              f"{float(Z[i]['V0']):.4f}), top-100 shared {drv[i]['overlap']}, identical {drv[i]['identical']}, "
              f"{drv[i]['s']} s", flush=True)
    G.check("M3 driver: V within 0.01 of GPU job 2 at all 29 forks", all(d["dV"] <= 0.01 for d in drv.values()),
            f"largest |dV| {np.nanmax([d['dV'] for d in drv.values()]):.2e}")
    G.check("M3 driver: 95 or more of the top 100 shared at all 29 forks", min(d["overlap"] for d in drv.values()) >= 95,
            f"smallest {min(d['overlap'] for d in drv.values())}; identical A at "
            f"{sum(d['identical'] for d in drv.values())} of 29 forks")
    G.check("M3 driver: all 64 layers seen at every fork", all(d["layers"] == G.NL for d in drv.values()))

    G.stage("M3: iterative re-ranking, 5 steps x 29 forks = 145 gradient passes")
    m3_steps = {}
    for b in G.LADDER:
        ix = ladder_ix[b]
        S = list(order[("M1", b)][:3])
        steps = []
        for add in M3_STEPS:
            score = np.zeros(G.NL * G.NI)
            Vs, zero_ok, layers_ok, finite = {}, True, True, True
            for i in ix:
                A, m, V, nl = G.attrib_at(M, F[i], ids[i], S)
                zero_ok &= bool((A.ravel()[S] == 0).all())
                layers_ok &= nl == G.NL
                finite &= bool(np.isfinite(A).all() and np.isfinite(V))
                score += A.ravel().astype(np.float64) / norms[i]
                Vs[int(i)] = V
            new = G.top_order(score, add, exclude=S)
            ok = G.check(f"M3 {b}, {len(S)} zeroed: A exactly 0 there, 64 layers, finite, {add} new neurons",
                         zero_ok and layers_ok and finite and len(set(new)) == add and not set(new) & set(S),
                         f"mean V/V0 {np.mean([Vs[i] / float(Z[i]['V0']) for i in ix]):.3f}; added layers "
                         f"{sorted(x // G.NI for x in new)[:12]}{' ...' if add > 12 else ''}")
            steps.append({"zeroed": len(S), "V_grad_pass": Vs, "added": new,
                          "added_score": [float(score[x]) for x in new]})
            S += new
            if not ok:
                break
        order[("M3", b)] = S
        m3_steps[b] = steps
        G.info(f"M3 {b}: shares {len(set(S) & set(order[('M1', b)]))} of 100 with M1; at 10: "
             f"{len(set(S[:10]) & set(order[('M1', b)][:10]))}; at 20: {len(set(S[:20]) & set(order[('M1', b)][:20]))}")

    G.stage("Sets, nesting and matched random controls at size 100")
    sets = {}
    for (meth, b), o in sorted(order.items()):
        G.check(f"{meth} {b}: 100 distinct neurons, so every size is the first k (nested)",
                len(o) == 100 and len(set(o)) == 100 and (meth != "M3" or o[:3] == order[("M1", b)][:3]))
        for k in G.SIZES:
            if meth == "M3" and k == 3:
                continue  # M1's set by construction
            sets[f"{meth}|{b}|{k}"] = {"method": meth, "b": b, "k": k, "kind": "top", "flat": o[:k],
                                       "spec": G.spec(o[:k]), "forks": ladder_ix[b]}
        ctrl = G.control(o, G.mean_absa(Z, ladder_ix[b]))
        G.check_control(f"{meth} {b}", o, ctrl)
        sets[f"rand_{meth}|{b}|100"] = {"method": meth, "b": b, "k": 100, "kind": "rand", "flat": ctrl,
                                        "spec": G.spec(ctrl), "forks": ladder_ix[b], "control_of": f"{meth}|{b}|100"}

    G.stage("V at each set's own forks (set zeroed at all positions, no gradient)")
    V0 = {b: own_fork_V(M, F, ids, ladder_ix[b], []) for b in G.LADDER}
    table = {}
    for key, s in sets.items():
        Vs = own_fork_V(M, F, ids, s["forks"], s["flat"])
        v0 = V0[s["b"]]
        table[key] = {"V": Vs, "mean_V": float(np.mean(list(Vs.values()))),
                      "mean_V_over_V0": float(np.mean([Vs[i] / v0[i] for i in Vs])),
                      "halved": int(sum(Vs[i] <= v0[i] / 2 for i in Vs)), "n": len(Vs)}
    for b in G.LADDER:
        G.check(f"bug check, {b}: M1's 100-neuron set lowers mean V at its own forks",
                table[f"M1|{b}|100"]["mean_V"] < np.mean(list(V0[b].values())),
                f"mean V {np.mean(list(V0[b].values())):.3f} -> {table[f'M1|{b}|100']['mean_V']:.3f}")
    print("\nmean V / V0 at own forks (forks halved):", flush=True)
    for b in G.LADDER:
        print(f"  {b} ({len(ladder_ix[b])} forks; mean V0 {np.mean(list(V0[b].values())):.3f})")
        for meth in ("M1", "M2", "M3"):
            cells = []
            for k in G.SIZES:
                t = table.get(f"{meth}|{b}|{k}") or table[f"M1|{b}|{k}"]
                cells.append(f"{k}: {t['mean_V_over_V0']:.2f} ({t['halved']})")
            t = table[f"rand_{meth}|{b}|100"]
            print(f"    {meth}  " + "  ".join(cells) + f"   random 100: {t['mean_V_over_V0']:.2f} ({t['halved']})", flush=True)
    for b in G.LADDER:  # grad-pass V during M3 against the no-gradient V of the same sets (info)
        d = [abs(st["V_grad_pass"][i] - (table.get(f"M3|{b}|{st['zeroed']}") or table[f"M1|{b}|{st['zeroed']}"])["V"][i])
             for st in m3_steps[b] for i in st["V_grad_pass"]]
        G.info(f"M3 {b}: V in the gradient pass vs without gradient, same set: largest |diff| {max(d):.2e}")

    n_ok, n = G.summary_lines()
    all_ok = n_ok == n
    out.write_json("m123_sets.json", sets)
    out.write_json("m3_steps.json", m3_steps)
    out.write_json("m123_summary.json", {"checks": G.RESULTS, "driver_check": drv, "V0_own_forks": V0,
                                         "V_own_forks": table, "minutes": round((time.time() - t_start) / 60, 1)})
    print(f"\n{n_ok} of {n} checks passed in {(time.time() - t_start) / 60:.0f} minutes.")
    out.finish(all_ok)
    print("STEP 3 JOB 4a PASS" if all_ok else "STEP 3 JOB 4a FAIL.", flush=True)
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
