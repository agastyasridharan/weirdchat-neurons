"""Phase 2d: behavior rate when the upstream sets (upsets.py) are zeroed or amplified (cenv; qwen_abl + gemma servers).

Conditions:
  base                              all 150 high-rate prompts (rate.py) + all 123 low-rate prompts (sidefx.py)
  <b>|<kind>        zeroed          own behavior's high-rate prompts; med100 and head10 on all five behaviors' prompts
                                    kinds: late, early, mid100, med100, randmed100, head10, randhead10
  <b>|<kind>|x2     amplified       own behavior's low-rate prompts; kinds: late, early, med100, randmed100, head10, randhead10
Follow-up conditions (added after the first pass; the run resumes):
  randhead10 also on all five behaviors' high-rate prompts (off-target control for head10)
  denying-ai-identity|med100 zeroed and x2/x4 on its high-rate prompts and on the legitimate-use sets of legit.py
  (ai_question, no_body, imagine): does this set carry the model's "I am an AI" self-description?
Each condition also gets damage (NLL on the 200 reference replies) and V at all 46 forks. 8 replies per prompt,
WeirdChat sampling, production judge (thinking on).

  python uprate.py run | summary | export
"""
import asyncio
import json
import os
import sys

import numpy as np

from common import MAX_RESP, RES, JsonlStore, Pool, generate, judge, log
from rate import B5, One, fork_V, nll
from rate import prompts as high_prompts
from sidefx import low_prompts

N = 8
ZERO = ["late", "early", "mid100", "med100", "randmed100", "head10", "randhead10"]
WIDE = {"med100", "head10", "randhead10"}
LEGIT = ["ai_question", "no_body", "imagine"]
AMP = ["late", "early", "med100", "randmed100", "head10", "randhead10"]


def conditions():
    S = json.loads((RES / "upstream_sets.json").read_text())

    def spec(s, scale=None):
        x = {"heads": s["heads"]} if "heads" in s else {"spec": s["spec"]}
        return x | ({"scale": scale} if scale else {})
    C = {"base": {"spec": {}, "hi": "all", "lo": "all"}}
    for b in B5:
        for k in ZERO:
            C[f"{b}|{k}"] = {"spec": spec(S[f"{b}|{k}"]), "hi": "all" if k in WIDE else b, "lo": None, "b": b, "kind": k}
    for b in B5:
        for k in AMP:
            C[f"{b}|{k}|x2"] = {"spec": spec(S[f"{b}|{k}"], 2), "hi": None, "lo": b, "b": b, "kind": k}
    d = "denying-ai-identity"
    C[f"{d}|med100|x2"]["hi"] = d
    C[f"{d}|med100|x4"] = {"spec": spec(S[f"{d}|med100"], 4), "hi": d, "lo": d, "b": d, "kind": "med100"}
    for c in ["base", f"{d}|med100", f"{d}|med100|x2", f"{d}|med100|x4"]:
        C[c]["legit"] = LEGIT
    return C


def legit_prompts(sets):
    """legit.py prompts as work items: pid 'L:<set>:<i>', judged with the set's first rubric."""
    import legit
    from common import prompt_ids
    return [{"pid": f"L:{s}:{i}", "b": legit.JUDGED[s][1][0], "user": t, "ids": prompt_ids(t)}
            for s in sets for i, t in enumerate(legit.JUDGED[s][2])]


def pick(P, who):
    return [] if who is None else P if who == "all" else [p for p in P if p["b"] == who]


async def run():
    C = conditions()
    HI, LO = high_prompts(), low_prompts()
    gen = JsonlStore(RES / "stage_a_gen.jsonl").done
    F = [json.loads(l) for l in (RES / "phase2_forks.jsonl").read_text().splitlines()]
    fork_ids = []
    for f in F:
        pid, si = f["rid"].split("|")
        fork_ids.append(gen[pid]["prompt_ids"] + list(gen[pid]["samples"][int(si)]["ids"])[:f["fork"][0]])
    refs = json.loads((RES / "ref_texts.json").read_text())
    gs, js, cs = (JsonlStore(RES / f"up_{n}.jsonl") for n in ("gen", "judge", "cond"))
    jpool = Pool("gemma", 96)
    qurls = [u for u, r in json.loads(open(os.environ["WCN_SERVERS"]).read()).items() if r == "qwen_abl"]
    import common, legit
    common.RUBRICS = common.load_rubrics() | legit.RUBRICS
    LG = legit_prompts(LEGIT)
    work = {c: [("hi", p) for p in pick(HI, C[c]["hi"])] + [("lo", p) for p in pick(LO, C[c]["lo"])]
            + [("legit", p) for p in LG if p["pid"].split(":")[1] in C[c].get("legit", [])] for c in C}
    log(f"{len(C)} conditions, {sum(map(len, work.values())) * N} replies; high {len(HI)} low {len(LO)} prompts; qwen {qurls}")
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
            assert r["heads"] == sum(len(v) for v in sp.get("heads", {}).values()), (c, r)
            if c not in cs.done:
                ns = await asyncio.gather(*[nll(q, x) for x in refs])
                vs = await asyncio.gather(*[fork_V(q, f, ids) for f, ids in zip(F, fork_ids)])
                cs.add({"key": c, "check": r, "nll": [n for n, _ in ns], "ntok": [t for _, t in ns], "V": vs})

            async def one(mode, p):
                key = f"{c}|{mode}|{p['pid']}"
                if key not in gs.done:
                    out = await generate(q, p["ids"] if "ids" in p else gen[p["pid"]]["prompt_ids"], N, MAX_RESP)
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
    """Per-condition rates with prompt-bootstrap CIs, damage, fork V; used by summary and export."""
    C = conditions()
    HI, LO = high_prompts(), low_prompts()
    js, cs = JsonlStore(RES / "up_judge.jsonl").done, JsonlStore(RES / "up_cond.jsonl").done
    F = [json.loads(l) for l in (RES / "phase2_forks.jsonl").read_text().splitlines()]
    rng = np.random.default_rng(0)
    base_nll = np.array(cs["base"]["nll"])

    def rate(c, mode, ps):
        m = np.array([[js.get(f"{c}|{mode}|{p['pid']}|{i}", {}).get("match") for i in range(N)] for p in ps], dtype=float)
        if not len(ps) or np.isnan(m).all():
            return None
        pr = np.nanmean(m, 1)
        bs = [np.nanmean(pr[rng.integers(0, len(pr), len(pr))]) for _ in range(2000)]
        return [float(np.nanmean(pr)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int((~np.isnan(m)).sum())]

    rows = []
    for c in C:
        if c not in cs:
            continue
        r = {"c": c, "b": C[c].get("b"), "kind": C[c].get("kind"), "amp": c.endswith("|x2"),
             "dnll": float((np.array(cs[c]["nll"]) - base_nll).sum() / np.array(cs[c]["ntok"]).sum()), "V": cs[c]["V"]}
        bs = B5 if C[c]["hi"] == "all" else [C[c]["hi"]] if C[c]["hi"] else []
        r["hi"] = {b: rate(c, "hi", [p for p in HI if p["b"] == b]) for b in bs}
        r["hi_held"] = {b: rate(c, "hi", [p for p in HI if p["b"] == b and p["split"] != "discovery"]) for b in bs}
        r["hi_disc"] = {b: rate(c, "hi", [p for p in HI if p["b"] == b and p["split"] == "discovery"]) for b in bs}
        bl = B5 if C[c]["lo"] == "all" else [C[c]["lo"]] if C[c]["lo"] else []
        r["lo"] = {b: rate(c, "lo", [p for p in LO if p["b"] == b]) for b in bl}
        LG = legit_prompts(C[c].get("legit", []))
        r["legit"] = {s: rate(c, "legit", [p for p in LG if p["pid"].split(":")[1] == s]) for s in C[c].get("legit", [])}
        if r["b"]:
            own = [i for i, f in enumerate(F) if f["behavior_id"] == r["b"]]
            r["V_own"] = float(np.mean([cs[c]["V"][i] for i in own if cs[c]["V"][i] is not None]))
            r["V_own_base"] = float(np.mean([cs["base"]["V"][i] for i in own if cs["base"]["V"][i] is not None]))
        rows.append(r)
    return rows


def summary():
    rows = tables()
    R = {r["c"]: r for r in rows}
    f = lambda x: "  -  " if x is None else f"{x[0]:.3f}"
    for b in B5:
        print(f"== {b}: high-rate prompts (base {f(R['base']['hi'][b])}; held-out {f(R['base']['hi_held'][b])}), "
              f"low-rate prompts (base {f(R['base']['lo'][b])})")
        for k in ZERO:
            r = R.get(f"{b}|{k}")
            if r:
                print(f"   zero {k:11s} rate {f(r['hi'][b])} [{r['hi'][b][1]:.2f},{r['hi'][b][2]:.2f}]  held-out {f(r['hi_held'][b])}  "
                      f"V own {r['V_own']:.3f} ({r['V_own_base']:.3f})  dNLL {r['dnll']:.4f}")
        for k in AMP:
            r = R.get(f"{b}|{k}|x2")
            if r:
                print(f"   x2   {k:11s} low-prompt rate {f(r['lo'][b])} [{r['lo'][b][1]:.2f},{r['lo'][b][2]:.2f}]  "
                      f"V own {r['V_own']:.3f}  dNLL {r['dnll']:.4f}")
    d = "denying-ai-identity"
    print(f"\n== {d}|med100: high-rate prompts and legitimate-use sets (share showing the legitimate behavior)")
    for c in ["base", f"{d}|med100", f"{d}|med100|x2", f"{d}|med100|x4"]:
        r = R.get(c)
        if r:
            print(f"   {c:34s} denial (high) {f(r['hi'].get(d))}  (low) {f(r['lo'].get(d))}  embodiment (high) {f(r['hi'].get('false-physical-embodiment'))}  "
                  + "  ".join(f"{s} {f(v)}" for s, v in r["legit"].items()) + f"  dNLL {r['dnll']:.4f}")
    print("\n== off-target: rate / base on each behavior's high-rate prompts")
    print(" " * 34 + "".join(b[:10].rjust(12) for b in B5))
    for k in ["med100", "head10", "randhead10"]:
        for b in B5:
            r = R.get(f"{b}|{k}")
            if r:
                print(f"  {b[:22]}|{k:9s}" + "".join((f"{r['hi'][x][0] / R['base']['hi'][x][0]:.2f}" if r['hi'][x] else "-").rjust(12) for x in B5))


def export():
    (RES / "site_upstream.json").write_text(json.dumps({"rows": tables(), "behaviors": B5}))
    print("wrote", RES / "site_upstream.json")


if __name__ == "__main__":
    {"run": lambda: asyncio.run(run()), "summary": summary, "export": export}[sys.argv[1]]()
