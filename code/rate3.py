"""Phase 3c: behavior rates for all 11 behaviors when the shared sets (sets3.py) are zeroed or amplified (cenv;
qwen_abl + gemma servers).

Prompts: HELD-OUT split only (the attribution used discovery prompts), up to NP per behavior: "hi" = Stage A
thinking-on rate >= 0.1, "lo" = below 0.1 (for amplification). N replies per prompt, production judge.
Conditions:
  base                                      hi + lo, all behaviors
  <set> zeroed                              hi, all behaviors: shared100, sharedmid100, rare100, selfpair100 and their matched
                                            random sets; denying-ai-identity|med100 (Phase 2d's "I am an AI" set)
  <set>|x2                                  lo, all behaviors: shared100, sharedmid100, rare100, selfpair100 and
                                            the random sets of all but rare100; dai_med100|x2 on hi (does more "I am an AI" lower
                                            the others?)
  loo100|<b>, loomid100|<b> zeroed          hi of b only: shared set built without b (all layers / below 48), and
                                            their matched random sets
  <c>#2                                     a second batch of N replies for the REP conditions, pooled with <c>
Each condition also gets damage (NLL on the 200 reference replies).

RATE3_MODE=single: base plus the single neurons picked from screen1.py (results/single_sets.json), zeroed on the
high-rate prompts of the behaviors each was picked for (or all), 16 replies per prompt, stores sn_*.jsonl. REP_SN get
a second batch of 16.

  python rate3.py run | summary | export
"""
import asyncio
import json
import os
import sys
from collections import defaultdict

import numpy as np

from common import MAX_RESP, RES, JsonlStore, Pool, generate, get_split, judge, load_prompts, log
from rate import One, nll

MODE = os.environ.get("RATE3_MODE", "shared")  # "single": the single neurons from screen1.py (results/single_sets.json)
ST = {"shared": "sh", "single": "sn"}[MODE]
N, NP, MIN_RATE = (8 if MODE == "shared" else 16), 12, 0.1
ZERO = ["shared100", "rand_shared100", "sharedmid100", "rand_sharedmid100", "rare100", "rand_rare100", "selfpair100",
        "rand_selfpair100"]
AMP = ["shared100", "rand_shared100", "sharedmid100", "rand_sharedmid100", "rare100", "selfpair100", "rand_selfpair100"]
DAI = "denying-ai-identity|med100"
# second batch of N replies per prompt for the conditions the first pass singled out (keys "<c>#2", pooled with <c>)
REP = ["base", "sharedmid100", "rand_sharedmid100", "selfpair100", "rand_selfpair100", "dai_med100", "sharedmid100|x2",
       "rand_sharedmid100|x2", "selfpair100|x2", "rand_selfpair100|x2"]
# single mode: second batch for the neurons whose first batch moved the rate (either way), their controls and base
REP_SN = ["base"] + [f"{r}{c}" for c in ["chemtrails-assertion|L24n8304", "recommends-drunk-driving|L38n16525",
                                         "claims-called-911|L38n2788"] for r in ("", "rand_")]
CTRL = {"shared100": "rand_shared100", "sharedmid100": "rand_sharedmid100", "rare100": "rand_rare100",
        "selfpair100": "rand_selfpair100"}


def prompts():
    J = JsonlStore(RES / "stage_a_judge_on.jsonl").done
    pr = load_prompts().set_index("prompt_id")
    sp = get_split()
    m = defaultdict(list)
    for j in J.values():
        if j["match"] is not None:
            m[j["prompt_id"]].append(j["match"])
    byb = defaultdict(lambda: {"hi": [], "lo": []})
    for pid, v in sorted(m.items()):
        if sp[pid]["split"] == "discovery":
            continue
        b = pr.loc[pid, "behavior_id"]
        byb[b]["hi" if np.mean(v) >= MIN_RATE else "lo"].append(
            {"pid": pid, "b": b, "user": pr.loc[pid, "user_text"], "stage_a": float(np.mean(v))})
    rng = np.random.default_rng(11)
    out = {"hi": [], "lo": []}
    for b in sorted(byb):
        for mode in out:
            ps = byb[b][mode]
            out[mode] += [ps[k] for k in sorted(rng.permutation(len(ps))[:NP].tolist())]
    return out["hi"], out["lo"]


def conditions():
    if MODE == "single":
        S = json.loads((RES / "single_sets.json").read_text())
        C = {"base": {"spec": {}, "hi": "all", "lo": None}} | {
            k: {"spec": {"spec": s["spec"]}, "hi": s["hi"], "lo": None, "kind": s["kind"], "b": s.get("b")} for k, s in S.items()}
        for c in REP_SN:
            C[f"{c}#2"] = C[c] | {"rep": c}
        return C
    S = json.loads((RES / "shared_sets.json").read_text())
    U = json.loads((RES / "upstream_sets.json").read_text())
    C = {"base": {"spec": {}, "hi": "all", "lo": "all"}}
    for k in ZERO:
        C[k] = {"spec": {"spec": S[k]["spec"]}, "hi": "all", "lo": None, "kind": k}
    C["dai_med100"] = {"spec": {"spec": U[DAI]["spec"]}, "hi": "all", "lo": None, "kind": "dai_med100"}
    for k in AMP:
        C[f"{k}|x2"] = {"spec": {"scale": 2, "spec": S[k]["spec"]}, "hi": None, "lo": "all", "kind": k}
    C["dai_med100|x2"] = {"spec": {"scale": 2, "spec": U[DAI]["spec"]}, "hi": "all", "lo": None, "kind": "dai_med100"}
    for k, s in S.items():
        if s["kind"] in ("loo100", "rand_loo100", "loomid100", "rand_loomid100"):
            C[k] = {"spec": {"spec": s["spec"]}, "hi": s["b"], "lo": None, "kind": s["kind"], "b": s["b"]}
    for c in REP:
        C[f"{c}#2"] = C[c] | {"rep": c}
    return C


def pick(P, who):
    return [] if who is None else P if who == "all" else [p for p in P if p["b"] == who]


async def run():
    C = conditions()
    HI, LO = prompts()
    gen = JsonlStore(RES / "stage_a_gen.jsonl").done
    refs = json.loads((RES / "ref_texts.json").read_text())
    gs, js, cs = (JsonlStore(RES / f"{ST}_{n}.jsonl") for n in ("gen", "judge", "cond"))
    jpool = Pool("gemma", 96)
    qurls = [u for u, r in json.loads(open(os.environ["WCN_SERVERS"]).read()).items() if r == "qwen_abl"]
    work = {c: [("hi", p) for p in pick(HI, C[c]["hi"])] + [("lo", p) for p in pick(LO, C[c]["lo"])] for c in C}
    log(f"{len(C)} conditions, {sum(map(len, work.values())) * N} replies; held-out high {len(HI)} low {len(LO)} prompts; "
        f"qwen {qurls}")
    jtasks, queued = [], set()

    async def judge_one(k, p, text):
        r = await judge(jpool, p["b"], p["user"], text, thinking=True)
        js.add({"key": k, "match": r["match"], "explanation": r["explanation"]})

    def judge_all(c):
        for mode, p in work[c]:
            for i, s in enumerate(gs.done.get(f"{c}|{mode}|{p['pid']}", {}).get("samples", [])):
                k = f"{c}|{mode}|{p['pid']}|{i}"
                if k not in js.done and k not in queued:
                    queued.add(k)
                    jtasks.append(asyncio.create_task(judge_one(k, p, s)))

    queue = asyncio.Queue()
    for c in C:
        judge_all(c)
        if c not in cs.done or any(f"{c}|{m}|{p['pid']}" not in gs.done for m, p in work[c]):
            queue.put_nowait(c)

    async def worker(url):
        q = One(url, 256)
        while not queue.empty():
            c = queue.get_nowait()
            sp = C[c]["spec"]
            r = (await q.post("/collective_rpc", {"method": "set_ablation", "args": [json.dumps(sp)]}))["results"][0]
            assert r["zeroed"] == sum(len(v) for v in sp.get("spec", {}).values()), (c, r)
            if c not in cs.done:
                ns = await asyncio.gather(*[nll(q, x) for x in refs])
                cs.add({"key": c, "check": r, "nll": [n for n, _ in ns], "ntok": [t for _, t in ns]})

            async def one(mode, p):
                key = f"{c}|{mode}|{p['pid']}"
                if key not in gs.done:
                    out = await generate(q, gen[p["pid"]]["prompt_ids"], N, MAX_RESP)
                    gs.add({"key": key, "samples": [o["text"] for o in out], "n": [len(o["ids"] or []) for o in out]})
            await asyncio.gather(*[one(m, p) for m, p in work[c]])
            judge_all(c)
            log(f"{url[-4:]} {c}: generated; judged so far {len(js.done)}, queued {len(queued)}")
        await q.post("/collective_rpc", {"method": "set_ablation", "args": ["{}"]})

    await asyncio.gather(*[worker(u) for u in qurls])
    log(f"generation done; waiting for {sum(not t.done() for t in jtasks)} judgments")
    await asyncio.gather(*jtasks)
    log("done")


def tables():
    C = conditions()
    HI, LO = prompts()
    B = sorted({p["b"] for p in HI + LO})
    js, cs = JsonlStore(RES / f"{ST}_judge.jsonl").done, JsonlStore(RES / f"{ST}_cond.jsonl").done
    rng = np.random.default_rng(0)
    base_nll = np.array(cs["base"]["nll"])

    def per_prompt(c, mode, ps):
        cc = [c] + ([f"{c}#2"] if f"{c}#2" in C else [])
        m = np.array([[js.get(f"{x}|{mode}|{p['pid']}|{i}", {}).get("match") for x in cc for i in range(N)] for p in ps], dtype=float)
        return m

    def rate(c, mode, ps):
        m = per_prompt(c, mode, ps)
        if not len(ps) or np.isnan(m).all():
            return None
        pr = np.nanmean(m, 1)
        bs = [np.nanmean(pr[rng.integers(0, len(pr), len(pr))]) for _ in range(2000)]
        return [float(np.nanmean(pr)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int((~np.isnan(m)).sum())]

    def diff(c, c0, mode, ps, bs):
        """Rate under c minus rate under control c0 on the same prompts: per behavior, and the mean over behaviors bs,
        each with a 95% bootstrap over prompts (resampled within behavior)."""
        D = {}
        for b in bs:
            q = [p for p in ps if p["b"] == b]
            if q:
                D[b] = np.nanmean(per_prompt(c, mode, q), 1) - np.nanmean(per_prompt(c0, mode, q), 1)
        if not D:
            return None, None
        boot = {b: [np.nanmean(d[rng.integers(0, len(d), len(d))]) for _ in range(2000)] for b, d in D.items()}
        per = {b: [float(np.nanmean(d)), float(np.percentile(boot[b], 2.5)), float(np.percentile(boot[b], 97.5))] for b, d in D.items()}
        pool = np.mean([boot[b] for b in D], 0)
        return per, [float(np.mean([v[0] for v in per.values()])), float(np.percentile(pool, 2.5)), float(np.percentile(pool, 97.5))]

    def pooled(c, mode, ps, bs):
        """Mean over behaviors of rate / base rate (each behavior counts once), with a prompt bootstrap."""
        out = []
        for b in bs:
            r, r0 = rate(c, mode, [p for p in ps if p["b"] == b]), rate("base", mode, [p for p in ps if p["b"] == b])
            if r and r0 and r0[0] > 0:
                out.append(r[0] / r0[0])
        return float(np.mean(out)) if out else None

    E = json.loads((RES / "shared_stats.json").read_text())["eligible"]
    rows = []
    for c in C:
        if c not in cs or "rep" in C[c]:
            continue
        r = {"c": c, "kind": C[c].get("kind"), "b": C[c].get("b"), "amp": "|x2" in c,
             "dnll": float((np.array(cs[c]["nll"]) - base_nll).sum() / np.array(cs[c]["ntok"]).sum())}
        bh = B if C[c]["hi"] == "all" else [C[c]["hi"]] if C[c]["hi"] else []
        bl = B if C[c]["lo"] == "all" else [C[c]["lo"]] if C[c]["lo"] else []
        r["hi"] = {b: rate(c, "hi", [p for p in HI if p["b"] == b]) for b in bh}
        r["lo"] = {b: rate(c, "lo", [p for p in LO if p["b"] == b]) for b in bl}
        r["hi_rel"], r["lo_rel"] = pooled(c, "hi", HI, bh), pooled(c, "lo", LO, bl)
        k, amp = c.split("|")[0], "|x2" if "|x2" in c else ""
        if k in CTRL and f"{CTRL[k]}{amp}" in C:  # paired against the matched random set, eligible behaviors pooled
            mode = "lo" if amp else "hi"
            r["d_rand"], r["d_rand_pool"] = diff(c, CTRL[k] + amp, mode, HI if mode == "hi" else LO, E)
        if c in ("dai_med100", "dai_med100|x2"):
            r["d_rand"], r["d_rand_pool"] = diff(c, "base", "hi", HI, E)
        if r["kind"] in ("loo100", "loomid100", "single"):
            r["d_rand"], _ = diff(c, f"rand_{c}", "hi", HI, [r["b"]])
        if r["kind"] in ("single", "rand_single"):
            r["d_base"], _ = diff(c, "base", "hi", HI, [r["b"]])
        rows.append(r)
    return rows, B


def loo_pool():
    """Leave-one-out sets: mean over the left-out behaviors of (set - matched random) on the left-out behavior's
    high-rate prompts, with a 95% prompt bootstrap resampled within behavior."""
    HI, _ = prompts()
    js = JsonlStore(RES / "sh_judge.jsonl").done
    E = json.loads((RES / "shared_stats.json").read_text())["eligible"]
    rng = np.random.default_rng(0)

    def pp(c, ps):
        return np.array([np.nanmean(np.array([js.get(f"{c}|hi|{p['pid']}|{i}", {}).get("match") for i in range(N)], dtype=float))
                         for p in ps])
    out = {}
    for kind in ["loo100", "loomid100"]:
        boots, means = [], []
        for b in E:
            q = [p for p in HI if p["b"] == b]
            d = pp(f"{kind}|{b}", q) - pp(f"rand_{kind}|{b}", q)
            means.append(float(np.nanmean(d)))
            boots.append([np.nanmean(d[rng.integers(0, len(d), len(d))]) for _ in range(4000)])
        pool = np.mean(boots, 0)
        out[kind] = [float(np.mean(means)), float(np.percentile(pool, 2.5)), float(np.percentile(pool, 97.5)), sum(m < 0 for m in means)]
    return out


def summary():
    rows, B = tables()
    R = {r["c"]: r for r in rows}
    f = lambda x: "  -  " if x is None else f"{x[0]:.3f}"
    print(" " * 26 + "".join(b[:11].rjust(12) for b in B) + "   mean rel   dNLL")
    for mode in ["hi", "lo"]:
        print(f"== {mode} prompts (held-out)")
        for c, r in R.items():
            if r[mode] and len(r[mode]) > 1:
                print(f"  {c[:24]:24s}" + "".join(f(r[mode].get(b)).rjust(12) for b in B)
                      + f"   {r[mode + '_rel'] if r[mode + '_rel'] is not None else float('nan'):8.2f}   {r['dnll']:.4f}")
    g = lambda x: f"{x[0]:+.3f} [{x[1]:+.2f},{x[2]:+.2f}]"
    print("== top set minus its matched random set, same prompts (dai_med100: minus unedited); per behavior, then the mean "
          "over the shared-score behaviors with a 95% prompt bootstrap")
    for c, r in R.items():
        if r.get("d_rand_pool"):
            print(f"  {c[:22]:22s}" + "".join((f"{r['d_rand'][b][0]:+.3f}" if b in r["d_rand"] else "-").rjust(12) for b in B)
                  + f"   mean {g(r['d_rand_pool'])}")
    print("== leave-one-out shared sets, zeroed, on the left-out behavior's held-out high-rate prompts (minus matched random)")
    for c, r in R.items():
        if r["kind"] in ("loo100", "loomid100"):
            b = r["b"]
            print(f"  {c[:44]:44s} {f(r['hi'][b])} (random {f(R['rand_' + c]['hi'][b])}, base {f(R['base']['hi'][b])})  "
                  f"diff {g(r['d_rand'][b])}  dNLL {r['dnll']:.4f}")
    if MODE == "single":
        print("== single neurons, zeroed, on their behavior's held-out high-rate prompts")
        for c, r in R.items():
            if r["kind"] == "single":
                b = r["b"]
                print(f"  {c[:44]:44s} {f(r['hi'][b])} (random {f(R['rand_' + c]['hi'][b])}, base {f(R['base']['hi'][b])})  "
                      f"minus random {g(r['d_rand'][b])}  minus base {g(r['d_base'][b])}  "
                      f"(random minus base {g(R['rand_' + c]['d_base'][b])})  dNLL {r['dnll']:.4f}")
        return
    for k, v in loo_pool().items():
        print(f"  {k}: mean over the left-out behaviors {g(v)}; negative in {v[3]}")


def export():
    rows, B = tables()
    if MODE == "single":
        (RES / "site_single.json").write_text(json.dumps({"rows": rows, "behaviors": B}))
        print("wrote", RES / "site_single.json")
        return
    st = json.loads((RES / "shared_stats.json").read_text())
    (RES / "site_shared.json").write_text(json.dumps({"rows": rows, "behaviors": B, "stats": st, "loo": loo_pool()}))
    print("wrote", RES / "site_shared.json")


if __name__ == "__main__":
    {"run": lambda: asyncio.run(run()), "summary": summary, "export": export}[sys.argv[1]]()
