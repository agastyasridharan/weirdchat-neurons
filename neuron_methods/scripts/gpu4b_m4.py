"""GPU job 4b (attribution env, one B200): Step 3 for method M4 (RelP), its pre-set check, its sets and control.

The rules are in relp.py (frozen RMSNorm 1/rms, SiLU sigmoids, softmax
weights, output gates, DeltaNet decay, beta, query and key; the half rule at the MLP product). The attribution is the
repo's (A = a * dM/da at the down_proj input, summed over positions, attrib.py:82-97); only the backward pass differs.

Technical checks (each prints PASS or FAIL; outputs reach methods/ only if all pass):
  model and patches: 64 layers; transformers 5.14.1 and the exact modeling file; every rule on;
  exact zeros (one short fork): with the rules on, the attention q_proj and k_proj outputs and the DeltaNet query,
    key, z, b and a projections get exactly zero gradient, while v_proj and the DeltaNet value channels get some.
    With plain gradients, all of them get some. Exact zeros do not depend on rounding;
  at every fork: each patched module ran exactly once in the RelP pass (and none with the rules off); V in the RelP
    pass equals V in a plain gradient pass of this job within 0.01 (the forward values are the same by construction;
    a tiny-model test during development, not included here,
    found them identical); the DeltaNet gated norm's output matches the frozen formula within a
    relative error of 0.02 (bfloat16 rounding is about 0.004; a wrong formula gives errors near 1).
Reported, not gated: conservation (with every rule on, sum(emb * grad) should equal sum(logits * dM/dlogits); exact in
  float32 in the tiny-model test, but bfloat16 rounding is large here) and plain gradients for contrast; this job's
  plain attribution against GPU job 2's.

The M4 check (set before the run): zeroing each fork's own top 10 RelP neurons must halve V
(V <= V0 / 2) at 35 or more of the 46 forks. Top 1 and 3 are reported too. M1 is measured the same way in this job, on
the same forks with the same V0, from GPU job 2's attribution files. For reference, the original table counts the first k at
which V halves (code/analyze_attrib.py:31-32): 23, 37 and 39 forks by top 1, 3 and 10 (PHASE2.md:16-23); both counts
are printed for M4 and M1, and the original count is also rebuilt from our GPU job 2 verify files. If M4 fails, it is dropped.

Then M4's sets for the four ladder behaviors (scores combined as in sets.py), sizes 3 to 100, its matched random set
at 100, and V at each set's own forks.
"""
import json
import sys
import time

sys.dont_write_bytecode = True
import gpu4_lib as G  # noqa: E402
import numpy as np  # noqa: E402
import relp  # noqa: E402
import torch  # noqa: E402

OUT_NAMES = ["m4_sets.json", "m4_summary.json", "relp"]
KS_CHECK = [1, 3, 10]
NEED = 35  # forks out of 46 at k = 10
FLA_NORM_SHA256 = "d8654c43cbea15799a9863ba0758f46b86c411285adf34c3c96735ef984d88d5"  # fla 0.5.2 fused_norm_gate.py


class Keep:
    """Keeps the embeddings and the logits of the last forward pass, for the conservation numbers."""

    def __init__(self, M):
        self.t = {}
        M.m.model.embed_tokens.register_forward_hook(self._hook("emb"))
        M.m.lm_head.register_forward_hook(self._hook("logits"))

    def _hook(self, name):
        def hook(mod, inp, out):
            if out.requires_grad:
                out.retain_grad()  # the logits; the embeddings become a leaf with a grad at attrib.py:66
            self.t[name] = out
        return hook

    def relevance(self, name):
        t = self.t.get(name)
        if t is None or t.grad is None:
            return float("nan")
        return float((t.detach().double() * t.grad.double()).sum())


def zero_grad_test(M, f, ids_, rules):
    """Max |grad| over the parts the rules freeze, and the smallest per-layer max |grad| over the value paths."""
    cfg = M.m.config.get_text_config()
    kd = cfg.linear_key_head_dim * cfg.linear_num_key_heads
    keep, hs = {}, []

    def mk(name):
        def hook(mod, inp, out):
            if out.requires_grad:
                out.retain_grad()
            keep[name] = out
        return hook

    for l, L in enumerate(M.m.model.layers):
        if L.block_type == "full_attention":
            for nm in ("q_proj", "k_proj", "v_proj"):
                hs.append(getattr(L.self_attn, nm).register_forward_hook(mk(f"{nm}_{l}")))
        else:
            for nm in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"):
                hs.append(getattr(L.linear_attn, nm).register_forward_hook(mk(f"{nm}_{l}")))
    try:
        if rules:
            with relp.on():
                G.attrib_at(M, f, ids_, [])
        else:
            G.attrib_at(M, f, ids_, [])
        mx = lambda t: 0.0 if t is None else float(t.abs().max())
        frozen, value = 0.0, []
        for l, L in enumerate(M.m.model.layers):
            g = lambda nm: keep[f"{nm}_{l}"].grad
            if L.block_type == "full_attention":
                frozen = max(frozen, mx(g("q_proj")), mx(g("k_proj")))
                value.append(mx(g("v_proj")))
            else:
                qkv = g("in_proj_qkv")
                frozen = max(frozen, mx(None if qkv is None else qkv[..., :2 * kd]), mx(g("in_proj_z")),
                             mx(g("in_proj_b")), mx(g("in_proj_a")))
                value.append(mx(None if qkv is None else qkv[..., 2 * kd:]))
    finally:
        for h in hs:
            h.remove()
        keep.clear()
    return frozen, min(value)


def halving(V0, Vk):
    lit = {k: int(sum(Vk[i][k] <= V0[i] / 2 for i in V0)) for k in KS_CHECK}
    first = {k: int(sum(any(Vk[i][j] <= V0[i] / 2 for j in KS_CHECK if j <= k) for i in V0)) for k in KS_CHECK}
    return lit, first


def main():
    t_start = time.time()
    out = G.Outputs("4b", OUT_NAMES)
    G.stage("Inputs")
    F, ids, by_b = G.load_forks()
    Z = G.load_attrib(F)
    ladder_ix = {b: by_b[b] for b in G.LADDER}

    G.stage("Model and RelP patches")
    M = G.attrib.Model()
    G.check_model(M)
    expected = relp.install(M.m)
    G.check("RelP: every rule on", all(relp.FREEZE.values()), f"{relp.FREEZE}")
    norms_cls = {f"{type(L.linear_attn.norm).__module__}.{type(L.linear_attn.norm).__name__}"
                 for L in M.m.model.layers if L.block_type == "linear_attention"}
    info(f"patched calls expected per forward pass: {expected}")
    info(f"DeltaNet gated norm class: {norms_cls}")
    try:
        import importlib.util
        import fla
        import fla.modules.fused_norm_gate as fng
        info(f"fla {fla.__version__}; fused_norm_gate.py sha256 matches the 0.5.2 wheel: "
             f"{relp.file_sha256(fng.__file__) == FLA_NORM_SHA256}; 'kernels' package installed: "
             f"{importlib.util.find_spec('kernels') is not None}")
    except ImportError as e:
        info(f"fla not importable: {e}")
    keep = Keep(M)

    G.stage("Exact zeros: what the rules freeze gets no gradient (shortest fork)")
    i_short = min(range(len(F)), key=lambda i: len(ids[i]))
    fr, val = zero_grad_test(M, F[i_short], ids[i_short], rules=True)
    G.check("RelP pass: frozen parts get exactly zero gradient", fr == 0.0, f"max |grad| {fr:.3g}")
    G.check("RelP pass: value paths get gradient in every attention and DeltaNet layer", val > 0,
            f"smallest per-layer max |grad| {val:.3g}")
    fr0, val0 = zero_grad_test(M, F[i_short], ids[i_short], rules=False)
    G.check("plain pass (sanity): the same parts do get gradient", fr0 > 0 and val0 > 0, f"max |grad| {fr0:.3g}")
    torch.cuda.empty_cache()

    G.stage("RelP and plain attribution at all 46 forks (longest first)")
    relp_dir = out.path("relp")
    relp_dir.mkdir()
    rows, A_r = {}, {}
    for i in sorted(range(len(F)), key=lambda i: -len(ids[i])):
        t0 = time.time()
        relp.COUNTS.clear()
        A_n, m_n, V_n, nl_n = G.attrib_at(M, F[i], ids[i], [])
        c_plain = dict(relp.COUNTS)
        cons_n = (keep.relevance("emb"), keep.relevance("logits"))
        relp.COUNTS.clear()
        del relp.NORM_ERR[:]
        with relp.on():
            A, m_r, V_r, nl_r = G.attrib_at(M, F[i], ids[i], [])
        c_relp = dict(relp.COUNTS)
        cons_r = (keep.relevance("emb"), keep.relevance("logits"))
        A_r[i] = A
        np.savez_compressed(relp_dir / f"{i}.npz", A=A, M=m_r, V=V_r, key=F[i]["key"])
        top_n = set(np.argsort(-A_n.ravel())[:100].tolist())
        rows[i] = dict(b=F[i]["behavior_id"], len=len(ids[i]), V_plain=V_n, V_relp=V_r, V_job2=float(Z[i]["V0"]),
                       counts_ok=c_relp == expected and c_plain == {}, layers=(nl_n, nl_r),
                       norm_err=max(relp.NORM_ERR) if relp.NORM_ERR else float("nan"),
                       finite=bool(np.isfinite(A).all() and np.isfinite(V_r)),
                       cons_relp=cons_r, cons_plain=cons_n,
                       plain_vs_job2_top100=len(top_n & set(np.argsort(-Z[i]["A"].ravel())[:100].tolist())),
                       relp_vs_plain_top10=len(set(np.argsort(-A.ravel())[:10].tolist())
                                              & set(np.argsort(-Z[i]["A"].ravel())[:10].tolist())),
                       s=round(time.time() - t0, 1))
        r = rows[i]
        print(f"  fork {i:2d} {r['b'][:26]:26s} len {r['len']:4d}: V plain {V_n:.4f} RelP {V_r:.4f} (job 2 "
              f"{r['V_job2']:.4f}); calls ok {r['counts_ok']}; norm err {r['norm_err']:.1e}; conservation RelP "
              f"{cons_r[0]:.3g} vs {cons_r[1]:.3g}, plain {cons_n[0]:.3g}; top-10 shared with M1 "
              f"{r['relp_vs_plain_top10']}; {r['s']} s", flush=True)
        torch.cuda.empty_cache()
    G.check("every fork: each patched module ran once in the RelP pass, none in the plain pass",
            all(r["counts_ok"] for r in rows.values()), f"{sum(r['counts_ok'] for r in rows.values())} of 46")
    G.check("every fork: all 64 layers seen in both passes, finite values",
            all(r["layers"] == (G.NL, G.NL) and r["finite"] for r in rows.values()))
    dV = [abs(r["V_relp"] - r["V_plain"]) for r in rows.values()]
    G.check("every fork: V in the RelP pass equals V in the plain pass within 0.01", all(x <= 0.01 for x in dV),
            f"largest |dV| {np.nanmax(dV):.2e}; identical at {sum(x == 0 for x in dV)} of 46")
    ne = [r["norm_err"] for r in rows.values()]
    G.check("every fork: DeltaNet gated norm output matches the frozen formula (relative error 0.02 or less)",
            all(np.isfinite(x) and x <= 0.02 for x in ne), f"largest {np.nanmax(ne):.2e}")
    ratio = lambda c: c[0] / c[1] if c[1] else float("nan")
    rr = [ratio(r["cons_relp"]) for r in rows.values()]
    rp = [ratio(r["cons_plain"]) for r in rows.values()]
    info(f"conservation, sum(emb*grad) / sum(logits*dM/dlogits): RelP median {np.median(rr):.3f} "
         f"(range {min(rr):.3f} to {max(rr):.3f}); plain median {np.median(rp):.3f} (range {min(rp):.3g} to {max(rp):.3g})")
    info(f"this job's plain attribution vs GPU job 2: V within {max(abs(r['V_plain'] - r['V_job2']) for r in rows.values()):.2e}; "
         f"top 100 shared, smallest {min(r['plain_vs_job2_top100'] for r in rows.values())}")
    info(f"RelP top 10 shared with M1's top 10 at the same fork: median "
         f"{np.median([r['relp_vs_plain_top10'] for r in rows.values()]):.0f} of 10")

    G.stage("M4 check: zero each fork's own top 1, 3 and 10, with no gradient")
    V0, V_m4, V_m1 = {}, {}, {}
    for i in range(len(F)):
        V0[i] = G.V_at(M, F[i], ids[i], [])
        o4 = [int(x) for x in np.argsort(-A_r[i].ravel())[:10]]
        o1 = [int(x) for x in np.argsort(-Z[i]["A"].ravel())[:10]]
        V_m4[i] = {k: G.V_at(M, F[i], ids[i], o4[:k]) for k in KS_CHECK}
        V_m1[i] = {k: G.V_at(M, F[i], ids[i], o1[:k]) for k in KS_CHECK}
        print(f"  fork {i:2d} {F[i]['behavior_id'][:26]:26s} V0 {V0[i]:.3f}; M4 " +
              " ".join(f"{k}:{V_m4[i][k]:.3f}" for k in KS_CHECK) + "; M1 " +
              " ".join(f"{k}:{V_m1[i][k]:.3f}" for k in KS_CHECK), flush=True)
    lit4, first4 = halving(V0, V_m4)
    lit1, first1 = halving(V0, V_m1)
    orig = {}
    for k in KS_CHECK:  # the repo's definition on our GPU job 2 verify files (code/analyze_attrib.py:31-32)
        n = 0
        for i in range(len(F)):
            v = json.loads((G.ATTRIB / f"{i}.verify.json").read_text())
            kh = next((s["k"] for s in v["sets"] if s["V"] <= v["V0"] / 2), None)
            n += kh is not None and kh <= k
        orig[k] = n
    print("\nforks where V halves (of 46):          top 1   top 3   top 10")
    print(f"  M4, at that k                          {lit4[1]:5d}   {lit4[3]:5d}   {lit4[10]:6d}")
    print(f"  M4, by that k (first hit)              {first4[1]:5d}   {first4[3]:5d}   {first4[10]:6d}")
    print(f"  M1, at that k (this job)               {lit1[1]:5d}   {lit1[3]:5d}   {lit1[10]:6d}")
    print(f"  M1, by that k (this job)               {first1[1]:5d}   {first1[3]:5d}   {first1[10]:6d}")
    print(f"  M1, repo definition, GPU job 2 files   {orig[1]:5d}   {orig[3]:5d}   {orig[10]:6d}")
    print("  M1, original table (PHASE2.md:16-23)      23      37       39", flush=True)
    m4_pass = lit4[10] >= NEED
    print(f"\nM4 CHECK {'PASS' if m4_pass else 'FAIL'}: top 10 halves V at {lit4[10]} of 46 forks (needs {NEED})", flush=True)

    G.stage("M4 sets for the four ladder behaviors, matched random set at 100, V at own forks")
    sets, table = {}, {}
    for b in G.LADDER:
        ix = ladder_ix[b]
        nrm = [G.normalizer(A_r[i]) for i in ix]
        G.check(f"M4 {b}: every fork's normalizer is positive", all(x > 0 for x in nrm), f"smallest {min(nrm):.4g}")
        o = G.top_order(G.behavior_score([A_r[i] for i in ix], nrm), 100)
        G.check(f"M4 {b}: 100 distinct neurons, so every size is the first k (nested)", len(set(o)) == 100)
        for k in G.SIZES:
            sets[f"M4|{b}|{k}"] = {"method": "M4", "b": b, "k": k, "kind": "top", "flat": o[:k], "spec": G.spec(o[:k]),
                                   "forks": ix, "m4_check": "PASS" if m4_pass else "FAIL"}
        ctrl = G.control(o, G.mean_absa(Z, ix))
        G.check_control(f"M4 {b}", o, ctrl)
        sets[f"rand_M4|{b}|100"] = {"method": "M4", "b": b, "k": 100, "kind": "rand", "flat": ctrl, "spec": G.spec(ctrl),
                                    "forks": ix, "control_of": f"M4|{b}|100", "m4_check": "PASS" if m4_pass else "FAIL"}
        M1o = G.top_order(G.behavior_score([Z[i]["A"] for i in ix], [G.normalizer(Z[i]["A"]) for i in ix]), 100)
        info(f"M4 {b}: shares with M1 " + ", ".join(f"{k}: {len(set(o[:k]) & set(M1o[:k]))}" for k in G.SIZES))
    for key, s in sets.items():
        Vs = {int(i): G.V_at(M, F[i], ids[i], s["flat"]) for i in s["forks"]}
        table[key] = {"V": Vs, "mean_V": float(np.mean(list(Vs.values()))),
                      "mean_V_over_V0": float(np.mean([Vs[i] / V0[i] for i in Vs])),
                      "halved": int(sum(Vs[i] <= V0[i] / 2 for i in Vs)), "n": len(Vs)}
    print("\nmean V / V0 at own forks (forks halved):", flush=True)
    for b in G.LADDER:
        cells = [f"{k}: {table[f'M4|{b}|{k}']['mean_V_over_V0']:.2f} ({table[f'M4|{b}|{k}']['halved']})" for k in G.SIZES]
        t = table[f"rand_M4|{b}|100"]
        print(f"  {b}: M4 " + "  ".join(cells) + f"   random 100: {t['mean_V_over_V0']:.2f} ({t['halved']})", flush=True)

    n_ok, n = G.summary_lines()
    all_ok = n_ok == n
    out.write_json("m4_sets.json", sets)
    out.write_json("m4_summary.json", {
        "checks": G.RESULTS, "m4_check": {"pass": m4_pass, "need": NEED, "at_k": lit4, "by_k": first4,
                                          "M1_at_k": lit1, "M1_by_k": first1, "M1_orig_definition_job2": orig},
        "V0": V0, "V_M4_topk": V_m4, "V_M1_topk": V_m1, "forks": rows, "V_own_forks": table,
        "expected_calls": expected, "minutes": round((time.time() - t_start) / 60, 1)})
    print(f"\n{n_ok} of {n} checks passed in {(time.time() - t_start) / 60:.0f} minutes.")
    out.finish(all_ok)
    if all_ok:
        print(f"STEP 3 JOB 4b PASS; M4 CHECK {'PASS' if m4_pass else 'FAIL: M4 is dropped'}", flush=True)
    else:
        print("STEP 3 JOB 4b FAIL.", flush=True)
    return all_ok


def info(msg):
    G.info(msg)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
