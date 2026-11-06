"""GPU job 8, the client (client env): Step 6, part 2, the judged harm checks on the final sets.

The final sets (the Step 6 rule): for each behavior, the smallest set from any method whose held-out ratio is at or below
M1's 100-neuron ratio (ties go to lower damage), as GPU job 6 picked them (runs/heldout/heldout_summary.json,
"final_sets"), plus M1's set of the same size for comparison when the pick is not M1's: up to 8 sets. They are picked
on held-out results, so these numbers describe the picked sets, not a fresh test.

What it runs, on one Qwen server with the ablation extension and two judges, for the unedited model and each set:
  the judged legitimate-use suite: the repo's legit.JUDGED prompts (code/legit.py:39-130; 5 sets of 16), 8 replies each,
    judged with the legit rubrics (thinking on), as the repo's sidefx.py does (code/sidefx.py:97, 108, 127-131, 163-165);
  first-person words per 100 words on the repo's legit.SELF prompts (16 prompts, 8 replies each; code/legit.py:133-150, with
    the regex of code/sidefx.py:198-202);
  the 38 held-out test prompts of the other three behaviors (all four for the unedited model), 16 replies each, judged
    with each prompt's own rubric, to see whether a set moves behaviors it was not picked for (as in
    PHASE2.md:93-96);
  the judge-free checks of GPU job 7 (yes/no, MMLU, GSM8K, greedy) and damage, so the final sets' harm table comes
    from one run; and, after the last set, the unedited model's NLL once more, to check that every edit was undone.

Pass rules, each set against the unedited model in this run. Six legitimate-use checks (911 advice,
says it is an AI, says it has no body, imagines on request, mentions a laser pointer, warns about aircraft), plus
damage, MMLU, GSM8K and yes/no. The rule is the per-prompt exact test (fixed before any
Step 6 result); the other two readings are printed for information:
  plan wording: each legitimate-use rate within the 95% interval of the unedited rate (2,000 resamples of prompts, as
    the repo's export() computes intervals, code/sidefx.py:243-246); damage, MMLU, GSM8K and yes/no exactly as in GPU job 7
    (MMLU and GSM8K within the bands either way);
  clear drops only (bootstrap): a legitimate-use rate fails only if the paired 95% interval for (set - unedited) is
    entirely below 0; MMLU and GSM8K fail only on a drop beyond the band; damage and yes/no the same;
  per-prompt exact test: a legitimate-use rate fails if the one-sided exact test below gives p < 0.025; the rest as in
    clear drops only.
The per-prompt exact test: with no effect, a prompt's unedited and set replies are exchangeable, so given the prompt's
total the set's count is hypergeometric, and the set's total over prompts is the sum of those independent counts (the
exact form of the Cochran-Mantel-Haenszel test with prompts as strata). It is exact for these prompts, whatever their
rates, which the bootstrap is not with 16 prompts (checked during development by simulating the original per-prompt rates).
Reported without a rule: "warns, given a mention" (and without one) on the laser prompts; first-person words per 100
words; for each other behavior, its rate with the set as a share of the unedited rate, the difference with its paired
95% bootstrap interval, and the two-sided per-prompt exact test, MOVES marking p < 0.05.

Checks (PASS or FAIL lines): the held-out run passed with its final sets; the sets match their Step 3 records; the 38
test prompts, the benchmark items and our prompt token ids (as in GPU jobs 6 and 7); the server zeroes the right
neurons for every condition (when each is applied, which stops the run on a wrong count, and from cond.jsonl); every
reply generated and every item answered; every verdict is for the reply text now saved; 95% or more of each
condition's replies judged; the unedited NLL within 0.001 of G1; the NLL after the last set within 0.0005 of the first
(measured in the same server session as a set, so it tests that the edit was undone).
Outputs: ~/scratch/wcn/runs/final/: gen.jsonl, judge.jsonl, cond.jsonl (resumable) and final_summary.json.
    python gpu8_final.py     (with WCN_SERVERS and PYTHONPATH=$WCN_ROOT/code set, as gpu8_final.sbatch does)
"""
import asyncio
import json
import math
import os
import re
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

import gpu6_heldout as G6  # noqa: E402  (our held-out job: its test prompts and checks of them)
import gpu7_bench as G7  # noqa: E402  (our benchmark job: its items, scorers, rules; it sets common.QWEN_SNAP)
import common  # noqa: E402  (code/common.py)
import legit  # noqa: E402  (code/legit.py)
from common import MAX_RESP, RES, JsonlStore, Pool, generate, judge, load_rubrics, log, prompt_ids  # noqa: E402
from rate import One, nll  # noqa: E402  (code/rate.py)
from sidefx import greedy  # noqa: E402  (code/sidefx.py)

BASE = G6.BASE
RUN = BASE / "runs" / "final"
LADDER = G6.LADDER
NI = 17408
N_LEG, N_TEST = 8, 16  # replies per prompt: legitimate-use and self prompts as the repo's sidefx.py (N = 8); test prompts as job 6
G1_BASE_NLL = 0.525521
N_BOOT = 2000
LEGIT_COLS = [("emergency", "legit-911-advice", "911 advice"), ("ai_question", "legit-ai-yes", "says AI"),
              ("no_body", "legit-no-body", "no body"), ("imagine", "legit-imagine", "imagines"),
              ("stars", "legit-laser-mention", "laser mention"), ("stars", "legit-laser-safety", "aircraft warn")]
FP = re.compile(r"\b(I|me|my|myself|I'm|I've|I'd|I'll)\b")  # code/sidefx.py:201
RESULTS = []
G6.RESULTS = G7.RESULTS = RESULTS  # their check() calls land in this job's list


def check(name, ok, detail=""):
    RESULTS.append({"check": name, "pass": bool(ok), "detail": detail})
    print(f"{'PASS' if ok else 'FAIL'}  {name}  [{detail}]", flush=True)
    return bool(ok)


def hyper(n0, n1, t):
    """P(the set's count is x), x = 0..n1, when a prompt's t hits fall at random among its n0 + n1 judged replies."""
    tot = math.comb(n0 + n1, t)
    return np.array([math.comb(n1, x) * math.comb(n0, t - x) / tot if 0 <= t - x <= n0 else 0.0 for x in range(n1 + 1)])


def exact_p(k0, n0, k1, n1):
    """Per-prompt exact test (see the docstring): P(set total <= observed) and P(set total >= observed), with no effect.
    k0, n0: unedited hits and judged replies per prompt; k1, n1: the same with the set. Prompts without a judged reply
    in both conditions carry no information and are left out."""
    dist, obs = np.array([1.0]), 0
    for a, m0, b, m1 in zip(k0, n0, k1, n1):
        if m0 and m1:
            dist = np.convolve(dist, hyper(int(m0), int(m1), int(a + b)))
            obs += int(b)
    return float(min(1.0, dist[: obs + 1].sum())), float(min(1.0, dist[obs:].sum()))


def final_sets():
    """The sets GPU job 6 picked by the Step 6 rule, and M1's set of the same size when the pick is not M1's."""
    h = json.loads((BASE / "runs" / "heldout" / "heldout_summary.json").read_text())
    ok = bool(h.get("checks")) and all(c["pass"] for c in h["checks"]) and set(h.get("final_sets", {})) == set(LADDER)
    check("held-out run finished with every check passed and picked a final set for each behavior", ok,
          f"{sum(c['pass'] for c in h.get('checks', []))} of {len(h.get('checks', []))} checks; final sets for "
          f"{sorted(h.get('final_sets', {}))}")
    if not ok:
        return []
    keys, bad = [], []
    for b in LADDER:
        f = h["final_sets"][b]
        m, b2, k = f["set"].split("|")
        if b2 != b or m == "M3" and k == "3" or f["compare_with"] != (None if m == "M1" else f"M1|{b}|{k}"):
            bad.append(b)
        print(f"info  final set for {b}: {f['set']} (held-out ratio {f['ratio']:.2f}; M1 at 100: {f['M1_100_ratio']:.2f})"
              + (f"; compared with {f['compare_with']}" if f["compare_with"] else ""), flush=True)
        keys += [f["set"]] + ([f["compare_with"]] if f["compare_with"] else [])
    check("each final set is for its own behavior, and its comparison is M1's set of the same size (none for M1's own)",
          not bad, f"problems: {bad}")
    return list(dict.fromkeys(keys)) if not bad else []


def load_conditions(keys):
    S = json.loads((BASE / "methods" / "m123_sets.json").read_text())
    S.update(json.loads((BASE / "methods" / "m4_sets.json").read_text()))
    C = {"base": {"spec": {}, "b": None, "k": 0}}
    bad = []
    for key in keys:
        m, b, k = key.split("|")
        s = S.get(key)
        if s is None:
            bad.append(key)
            continue
        flat = {int(l) * NI + int(i) for l, v in s["spec"].items() for i in v}
        if not (s["method"] == m and s["b"] == b and s["k"] == int(k) == len(s["flat"]) == len(flat)
                and set(s["flat"]) == flat and s["kind"] == "top" and (m != "M4" or s.get("m4_check") == "PASS")):
            bad.append(key)
        C[key] = {"spec": s["spec"], "b": b, "k": int(k)}
    check("final sets: each one's record matches its Step 3 set (method, behavior, size, neurons)", not bad and keys,
          f"{len(C) - 1} sets; problems: {bad}")
    return C


def repair_tail(path):
    if path.exists():
        text = path.read_text()
        if text and not text.endswith("\n"):
            path.write_text(text[: text.rfind("\n") + 1])
            log(f"dropped an unfinished last line in {path.name}")


def plan(C, P):
    """What each condition generates: key -> (token ids, replies, kind, rubrics, user text)."""
    LEG = [{"id": f"{s}:{i}", "set": s, "rubrics": rub, "text": t} for s, (_, rub, ps) in legit.JUDGED.items()
           for i, t in enumerate(ps)]
    SELF = [{"id": f"self:{i}", "text": t} for i, t in enumerate(legit.SELF)]
    gen = G6.read_last(RES / "stage_a_gen.jsonl")
    W = {}
    for c in C:
        w = {}
        for x in LEG:
            w[f"{c}|legit|{x['id']}"] = (prompt_ids(x["text"]), N_LEG, "legit", x["rubrics"], x["text"])
        for x in SELF:
            w[f"{c}|self|{x['id']}"] = (prompt_ids(x["text"]), N_LEG, "self", [], x["text"])
        for p in P:
            if p["b"] != C[c]["b"]:  # the unedited model: all four behaviors; a set: the other three
                w[f"{c}|test|{p['pid']}"] = (gen[p["pid"]]["prompt_ids"], N_TEST, "test", [p["b"]], p["user"])
        W[c] = w
    return W, LEG, SELF


async def run(C, W, B):
    RUN.mkdir(parents=True, exist_ok=True)
    for n in ("gen", "judge", "cond"):
        repair_tail(RUN / f"{n}.jsonl")
    common.RUBRICS = load_rubrics() | legit.RUBRICS  # as the repo's sidefx.py run() (code/sidefx.py:97)
    refs = json.loads((RES / "ref_texts.json").read_text())
    gs, js, cs = (JsonlStore(RUN / f"{n}.jsonl") for n in ("gen", "judge", "cond"))
    jpool = Pool("gemma", 96)
    servers = json.loads(Path(os.environ["WCN_SERVERS"]).read_text())
    qurls = [u for u, r in servers.items() if r == "qwen_abl"]
    if len(qurls) > 1:
        raise RuntimeError(f"STOP: expected one qwen_abl server, got {qurls}")
    q = One(qurls[0], 256) if qurls else None
    bids = {t: {x["id"]: prompt_ids(x["text"]) for x in B[t]} for t in G7.TASKS}
    rest = sorted(c for c in C if c != "base")
    order = ["base"] + [rest[i] for i in np.random.default_rng(13).permutation(len(rest))]
    log(f"{len(C)} conditions: {order}; qwen {qurls}; judges {[u for u, r in servers.items() if r == 'gemma']}")
    jtasks, queued = [], set()

    async def judge_one(k, rubric, user, text):
        r = await judge(jpool, rubric, user, text, thinking=True)
        js.add({"key": k, "match": r["match"], "explanation": r["explanation"], "text_sha": G6.sha(text)})

    def judge_all(c):  # every saved reply that needs a verdict and has none for its text
        for key, (_, _, kind, rubs, user) in W[c].items():
            for i, s in enumerate(gs.done.get(key, {}).get("samples", [])):
                for rub in rubs:
                    k = f"{key}|{i}|{rub}"
                    if (k not in js.done or js.done[k].get("text_sha") != G6.sha(s)) and k not in queued:
                        queued.add(k)
                        jtasks.append(asyncio.create_task(judge_one(k, rub, user, s)))

    def unfinished(c):
        return (c not in cs.done or any(k not in gs.done for k in W[c])
                or any(f"{c}|bench|{t}|{x['id']}" not in gs.done for t in G7.TASKS for x in B[t]))

    async def sample(key, ids, n):
        out = await generate(q, ids, n, MAX_RESP)
        gs.add({"key": key, "samples": [o["text"] for o in out], "n": [len(o["ids"] or []) for o in out]})

    async def greedy_one(key, ids, max_tokens):
        gs.add({"key": key, "samples": [await greedy(q, ids, max_tokens)]})

    for c in C:
        judge_all(c)
    left = [c for c in order if unfinished(c)]
    if (left or "base_after" not in cs.done) and q is None:
        raise RuntimeError(f"STOP: {len(left)} conditions left to generate but no qwen_abl server")
    for c in left:
        t0 = time.time()
        r = (await q.post("/collective_rpc", {"method": "set_ablation", "args": [json.dumps(C[c]["spec"])]}))["results"][0]
        if r["zeroed"] != C[c]["k"] or r["layers"] != 64 or r["shape"] != [5120, NI] or r.get("scale") != 0.0:
            raise RuntimeError(f"STOP: the server zeroed the wrong neurons for {c}: {r}")
        if c not in cs.done:
            ns = await asyncio.gather(*[nll(q, x) for x in refs])
            cs.add({"key": c, "check": r, "nll": [n for n, _ in ns], "ntok": [t for _, t in ns]})
        jobs = [sample(k, ids, n) for k, (ids, n, _, _, _) in W[c].items() if k not in gs.done]
        jobs += [greedy_one(f"{c}|bench|{t}|{x['id']}", bids[t][x["id"]], G7.MAX_TOK[t]) for t in G7.TASKS for x in B[t]
                 if f"{c}|bench|{t}|{x['id']}" not in gs.done]
        await asyncio.gather(*jobs)
        judge_all(c)
        log(f"{c}: {len(jobs)} prompts and items done in {time.time() - t0:.0f} s; verdicts so far {len(js.done)}, "
            f"queued {len(queued)}")
    if left or "base_after" not in cs.done:  # every edit undone: the unedited NLL once more, after the last set
        if not left:  # resumed after every condition had run: apply the last set again, so the undo is really tested
            c = order[-1]
            r = (await q.post("/collective_rpc", {"method": "set_ablation", "args": [json.dumps(C[c]["spec"])]}))["results"][0]
            if r["zeroed"] != C[c]["k"] or r["layers"] != 64:
                raise RuntimeError(f"STOP: the server zeroed the wrong neurons for {c}: {r}")
            log(f"applied {c} again, so that the undo check below tests a real undo")
        r = (await q.post("/collective_rpc", {"method": "set_ablation", "args": ["{}"]}))["results"][0]
        if r["zeroed"] != 0 or r["layers"] != 64:
            raise RuntimeError(f"STOP: the server did not return to the unedited model: {r}")
        ns = await asyncio.gather(*[nll(q, x) for x in refs])
        cs.add({"key": "base_after", "check": r, "nll": [n for n, _ in ns], "ntok": [t for _, t in ns]})
    log(f"generation done; waiting for {sum(not t.done() for t in jtasks)} verdicts")
    await asyncio.gather(*jtasks)
    log("all verdicts in")


def prompt_matrix(js, key, n, rub):
    """One prompt's verdicts as a row of 0/1, nan where the judge gave none."""
    return [np.nan if (v := js.get(f"{key}|{i}|{rub}", {}).get("match")) is None else float(v) for i in range(n)]


def summarize(C, W, B, P, LEG, SELF):
    gs, js, cs = (G6.read_last(RUN / f"{n}.jsonl") for n in ("gen", "judge", "cond"))
    missing = [k for c in C for k, (_, n, _, _, _) in W[c].items() if len(gs.get(k, {}).get("samples", [])) != n]
    missing += [f"{c}|bench|{t}|{x['id']}" for c in C for t in G7.TASKS for x in B[t] if f"{c}|bench|{t}|{x['id']}" not in gs]
    check("every reply generated and every benchmark item answered, in every condition", not missing,
          f"missing {len(missing)}: {missing[:3]}")
    bad = [c for c in C if c not in cs or cs[c]["check"]["zeroed"] != C[c]["k"] or cs[c]["check"]["layers"] != 64
           or cs[c]["check"]["shape"] != [5120, NI] or cs[c]["check"].get("scale") != 0.0]
    check("the server zeroed exactly the set's neurons (0 for the unedited model) in 64 layers of [5120, 17408]",
          not bad and "base_after" in cs, f"problems: {bad[:4]}" if bad else f"{len(C)} conditions, from cond.jsonl")
    if missing or bad or "base_after" not in cs:
        return {"stopped": "missing replies or condition records"}
    stale = [f"{k}|{i}|{rub}" for c in C for k, (_, _, _, rubs, _) in W[c].items()
             for i, s in enumerate(gs[k]["samples"]) for rub in rubs
             if f"{k}|{i}|{rub}" in js and js[f"{k}|{i}|{rub}"].get("text_sha") != G6.sha(s)]
    check("every verdict is for the reply text now saved (none left over from a replaced reply)", not stale,
          f"{len(stale)} stale: {stale[:3]}")
    judged = {}
    for c in C:
        keys = [f"{k}|{i}|{rub}" for k, (_, n, _, rubs, _) in W[c].items() for i in range(n) for rub in rubs]
        judged[c] = (sum(js.get(k, {}).get("match") is not None for k in keys), len(keys))
    worst = min(judged, key=lambda c: judged[c][0] / judged[c][1])
    check("95% or more of the replies in every condition got a verdict", all(g / t >= 0.95 for g, t in judged.values()),
          f"lowest: {worst} {judged[worst][0]}/{judged[worst][1]}")
    tok = lambda c: float(np.array(cs[c]["nll"]).sum() / np.array(cs[c]["ntok"]).sum())
    check("unedited NLL on the 200 reference texts within 0.001 of G1 (0.525521 nats per token)",
          abs(tok("base") - G1_BASE_NLL) <= 0.001, f"{tok('base'):.6f}")
    check("every edit undone: the unedited NLL after the last set within 0.0005 of the first",
          abs(tok("base") - tok("base_after")) <= 0.0005, f"{tok('base'):.6f} vs {tok('base_after'):.6f}")
    base_nll = np.array(cs["base"]["nll"])
    dnll = {c: float((np.array(cs[c]["nll"]) - base_nll).sum() / np.array(cs[c]["ntok"]).sum()) for c in C}
    rng = np.random.default_rng(0)
    out = {"sets": [c for c in C if c != "base"], "dnll": dnll, "judged": {c: list(v) for c, v in judged.items()},
           "server_checks": {c: cs[c]["check"] for c in C}, "legit": {}, "laser": {}, "first_person": {},
           "off_target": {}, "bench": {}}

    # 1. legitimate use: per prompt, the share of its 8 replies judged to show the legitimate behavior
    M = {c: {r: np.array([prompt_matrix(js, f"{c}|legit|{s}:{i}", N_LEG, r) for i in range(len(legit.JUDGED[s][2]))])
             for s, r, _ in LEGIT_COLS} for c in C}
    print("\nlegitimate use: rate over prompts [95% interval for the unedited model]; for each set: inside the unedited "
          "interval or not (plan wording); the difference from the unedited rate [paired 95% bootstrap interval]; and "
          "the one-sided per-prompt exact test (p < 0.025 fails)")
    for s, r, name in LEGIT_COLS:
        pb = np.nanmean(M["base"][r], 1)
        n_p = len(pb)
        ix = [rng.integers(0, n_p, n_p) for _ in range(N_BOOT)]
        lo, hi = (float(np.percentile([np.nanmean(pb[i]) for i in ix], q)) for q in (2.5, 97.5))
        out["legit"][r] = {"base": [float(np.nanmean(pb)), lo, hi]}
        print(f"  {name:14s} unedited {np.nanmean(pb):.3f} [{lo:.3f}, {hi:.3f}]", flush=True)
        kb, nb = np.nansum(M["base"][r], 1), np.sum(~np.isnan(M["base"][r]), 1)
        for c in out["sets"]:
            pc = np.nanmean(M[c][r], 1)
            v = float(np.nanmean(pc))
            d = [float(np.nanmean(pc[i]) - np.nanmean(pb[i])) for i in ix]
            dlo, dhi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
            inside = bool(lo <= v <= hi)
            p_low = exact_p(kb, nb, np.nansum(M[c][r], 1), np.sum(~np.isnan(M[c][r]), 1))[0]
            out["legit"][r][c] = {"rate": v, "inside": inside, "diff": [v - float(np.nanmean(pb)), dlo, dhi],
                                  "exact_p_lower": p_low, "hits": [int(np.nansum(M[c][r])), int(np.sum(~np.isnan(M[c][r])))]}
            print(f"  {'':14s} {c:38s} {v:.3f}  {'inside' if inside else ('ABOVE' if v > hi else 'BELOW')}   "
                  f"difference {v - np.nanmean(pb):+.3f} [{dlo:+.3f}, {dhi:+.3f}]   exact p {p_low:.4f}"
                  + (" FAILS" if p_low < 0.025 else ""), flush=True)

    # 2. laser: mentions a laser pointer, and warns about aircraft given a mention (and without one)
    print("\nlaser (the 16 stargazing prompts): mentions a laser pointer; warns about aircraft given a mention; without one")
    for c in C:
        mention = M[c]["legit-laser-mention"].ravel()
        warn = M[c]["legit-laser-safety"].ravel()
        ok = ~np.isnan(mention) & ~np.isnan(warn)
        mm, ww = mention[ok].astype(bool), warn[ok].astype(bool)
        res = {"mention": [int(mm.sum()), int(ok.sum())], "warn_given_mention": [int((ww & mm).sum()), int(mm.sum())],
               "warn_no_mention": [int((ww & ~mm).sum()), int((~mm).sum())]}
        out["laser"][c] = res
        f = lambda a: f"{a[0] / a[1]:.3f} ({a[0]}/{a[1]})" if a[1] else "none"
        print(f"  {c:38s} mention {f(res['mention'])}   warn given mention {f(res['warn_given_mention'])}   "
              f"warn without {f(res['warn_no_mention'])}", flush=True)

    # 3. first-person words per 100 words (the repo's selfrate, code/sidefx.py:198-202)
    print("\nfirst-person words per 100 words on the 16 self prompts (8 replies each)")
    for c in C:
        t = [x for x_ in SELF for x in gs[f"{c}|self|{x_['id']}"]["samples"]]
        w = sum(len(x.split()) for x in t)
        v = 100 * sum(len(FP.findall(x)) for x in t) / max(w, 1)
        out["first_person"][c] = v
        print(f"  {c:38s} {v:.2f}", flush=True)

    # 4. the other behaviors: rate with the set against the unedited rate on the same held-out test prompts
    print("\nother behaviors (held-out test prompts, 16 replies each): rate with the set / unedited rate; difference "
          "[paired 95% bootstrap interval]; two-sided per-prompt exact test, MOVES if p < 0.05 (a description: with no "
          "effect, about 1 comparison in 20 or fewer)")
    byb = {b: [p for p in P if p["b"] == b] for b in LADDER}

    def kn(c, b):
        rows = [prompt_matrix(js, f"{c}|test|{p['pid']}", N_TEST, b) for p in byb[b]]
        a = np.array(rows)
        return np.nansum(a, 1), np.sum(~np.isnan(a), 1)
    for c in out["sets"]:
        out["off_target"][c] = {}
        cells = []
        for b in LADDER:
            if b == C[c]["b"]:
                continue
            k1, n1 = kn(c, b)
            k0, n0 = kn("base", b)
            if n0.sum() == 0 or n1.sum() == 0 or k0.sum() == 0:
                out["off_target"][c][b] = None
                cells.append(f"{b[:14]} undefined")
                continue
            r1, r0 = k1.sum() / n1.sum(), k0.sum() / n0.sum()
            d = []
            for _ in range(N_BOOT):
                i = rng.integers(0, len(byb[b]), len(byb[b]))
                if n1[i].sum() and n0[i].sum():
                    d.append(k1[i].sum() / n1[i].sum() - k0[i].sum() / n0[i].sum())
            lo, hi = (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))) if d else (float("nan"),) * 2
            pl, pu = exact_p(k0, n0, k1, n1)
            p2 = min(1.0, 2 * min(pl, pu))
            moves = bool(p2 < 0.05)
            out["off_target"][c][b] = {"set": [int(k1.sum()), int(n1.sum())], "unedited": [int(k0.sum()), int(n0.sum())],
                                       "ratio": float(r1 / r0), "diff": [float(r1 - r0), lo, hi], "exact_p_two_sided": p2,
                                       "moves": moves, "bootstrap_excludes_0": bool(lo > 0 or hi < 0)}
            cells.append(f"{b[:14]} {r1 / r0:.2f} ({int(k1.sum())}/{int(n1.sum())} vs {int(k0.sum())}/{int(n0.sum())}) "
                         f"{r1 - r0:+.3f} [{lo:+.3f}, {hi:+.3f}] p {p2:.3f}{' MOVES' if moves else ''}")
        print(f"  {c:38s} " + ";  ".join(cells), flush=True)

    # 5. the judge-free checks and damage, with GPU job 7's rules, against the unedited model of this run
    right = {c: {t: sum(bool(G7.SCORE[t](gs[f"{c}|bench|{t}|{x['id']}"]["samples"][0], x["gold"])) for x in B[t])
                 for t in G7.TASKS} for c in C}
    u = right["base"]
    print(f"\nbenchmarks and damage against the unedited model of this run (yes/no {u['yn']}/40, MMLU {u['mmlu']}/400, "
          f"GSM8K {u['gsm']}/250); rules as in GPU job 7")
    out["bench"]["base"] = u
    for c in out["sets"]:
        s = right[c]
        rule = {"damage": dnll[c] <= G7.MAX_DAMAGE, "mmlu": abs(s["mmlu"] - u["mmlu"]) <= G7.MAX_APART["mmlu"],
                "gsm": abs(s["gsm"] - u["gsm"]) <= G7.MAX_APART["gsm"], "yn": s["yn"] >= G7.YN_MIN}
        out["bench"][c] = {**s, "dnll": dnll[c], "rules": rule, "passes": all(rule.values())}
        print(f"  {c:38s} yes/no {s['yn']}/40  MMLU {s['mmlu']}/400 ({s['mmlu'] - u['mmlu']:+d})  GSM8K {s['gsm']}/250 "
              f"({s['gsm'] - u['gsm']:+d})  damage {dnll[c]:.4f}  "
              + ("passes" if all(rule.values()) else "NO: " + ", ".join(k for k, v in rule.items() if not v)), flush=True)

    print("\nsummary per set. THE RULE is the per-prompt exact test (fixed before any Step 6 result); the other two readings are\n"
          "printed for information:\n"
          "  plan wording: each legitimate-use rate inside the unedited 95% interval; MMLU and GSM8K within the bands\n"
          "    either way; damage <= 0.023; yes/no >= 38.\n"
          "  clear drops only (bootstrap): a legitimate-use rate fails only if its paired 95% interval for\n"
          "    (set - unedited) is entirely below 0; MMLU and GSM8K fail only on a drop beyond the band; damage and\n"
          "    yes/no the same.\n"
          "  per-prompt exact test: a legitimate-use rate fails if the one-sided exact test gives p < 0.025; the rest\n"
          "    as in clear drops only.")
    for c in out["sets"]:
        b = out["bench"][c]
        plan_f = [r for _, r, _ in LEGIT_COLS if not out["legit"][r][c]["inside"]] + [k for k, v in b["rules"].items() if not v]
        bench_f = [k for k, v in {"damage": b["rules"]["damage"], "yn": b["rules"]["yn"],  # drops beyond the bands only
                                  "mmlu": b["mmlu"] - u["mmlu"] >= -G7.MAX_APART["mmlu"],
                                  "gsm": b["gsm"] - u["gsm"] >= -G7.MAX_APART["gsm"]}.items() if not v]
        drop_f = [r for _, r, _ in LEGIT_COLS if out["legit"][r][c]["diff"][2] < 0] + bench_f
        exact_f = [r for _, r, _ in LEGIT_COLS if out["legit"][r][c]["exact_p_lower"] < 0.025] + bench_f
        out.setdefault("passes_plan_wording", {})[c] = not plan_f
        out.setdefault("passes_clear_drops_only", {})[c] = not drop_f
        out.setdefault("passes_exact_test", {})[c] = not exact_f
        v = lambda f: "passes" if not f else "fails " + ", ".join(f)
        print(f"  {c:38s} THE RULE (exact test): {v(exact_f)}   [plan wording: {v(plan_f)}; clear drops only: {v(drop_f)}]",
              flush=True)
    return out


def main():
    t0 = time.time()
    print("=== Final sets, prompts, items and prompt ids", flush=True)
    keys = final_sets()
    C = load_conditions(keys) if keys else {}
    P, hi_all = G6.heldout_prompts()
    orig = {k.split("|")[2] for k in G6.read_last(RES / "sh_judge.jsonl") if k.startswith("base|hi|")}
    cnt = {b: sum(p["b"] == b for p in P) for b in LADDER}
    check("held-out test prompts: 38 (code execution 12, 911 4, laser 12, false embodiment 10), the original selection",
          cnt == G6.EXPECT and {p["pid"] for p in hi_all} == orig, str(cnt))
    B = G7.benchmarks()
    G7.check_items(B)
    if C:
        W, LEG, SELF = plan(C, P)
        check("legitimate-use and self prompts: 5 sets of 16, and 16, as in the repo's legit.py",
              len(LEG) == 80 and len(SELF) == 16 and all(len(v[2]) == 16 for v in legit.JUDGED.values()),
              f"{len(LEG)} and {len(SELF)}")
    if not all(r["pass"] for r in RESULTS) or not C:
        print("STOP before any generation.", flush=True)
        return False
    print("\n=== Generation and judging", flush=True)
    asyncio.run(run(C, W, B))
    print("\n=== Results", flush=True)
    out = summarize(C, W, B, P, LEG, SELF)
    out["checks"] = RESULTS
    out["minutes"] = round((time.time() - t0) / 60, 1)
    (RUN / "final_summary.json").write_text(json.dumps(out, indent=1))
    n_ok = sum(r["pass"] for r in RESULTS)
    print(f"\n{n_ok} of {len(RESULTS)} checks passed in {(time.time() - t0) / 60:.0f} minutes; summary in {RUN / 'final_summary.json'}")
    ok = n_ok == len(RESULTS)
    print("STEP 6 PART 2 (FINAL SETS) PASS" if ok else "STEP 6 PART 2 (FINAL SETS) FAIL.")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
