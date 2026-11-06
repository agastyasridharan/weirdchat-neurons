"""GPU job 7, the client (client env): Step 6, part 1, the harm checks that need no judge.

What it runs, on one Qwen server with the ablation extension:
  Conditions: for every method that passed validation (runs/val/val_summary.json), its sets at sizes 10 and 100 for
    each behavior (32 sets for four methods), and the unedited model first and again last, after every set.
  For each condition: 40 yes/no facts (the repo's legit.YESNO), 400 MMLU questions and 250 GSM8K problems, greedy, built as in
    the repo's sidefx.py (code/sidefx.py:36-44) and run and scored with the repo's own functions (sidefx.greedy, score_yn, score_mc,
    score_gsm; code/sidefx.py:70-93, 166-169); and damage: NLL on the 200 reference texts (code/rate.py:65-75).
  MMLU and GSM8K are our pinned copies from setup step 3, sampled as the repo's code does (random_state=0). Before anything
    runs, the stored unedited answers (results/sfx_gen.jsonl) must score 337/400, 241/250 and 40/40 on exactly
    these items, as in the original log (logs/sidefx.log:32), and our tokenizer must rebuild the stored Stage A prompt token ids.
  It needs no judge, so it can run while the held-out run (GPU job 6) is still going.

Pass rules (set before the run), each set against the first unedited run of this job:
  damage 0.023 nats per token or less; MMLU within 3.6 points (at most 14 of 400 questions apart, since 3.6 points is
  14.4 questions); GSM8K within 2.4 points (at most 6 of 250 problems apart); yes/no at least 38 of 40.
  The plan says "within", so a gain beyond a band also counts as outside; the table also says whether each set passes
  when only drops count (the reading chosen as the rule before any result).

Checks (PASS or FAIL lines): 40, 400 and 250 items, the stored answers reproducing the original scores on them; our prompt token
ids equal to the repo's; the validation run passed; every set present, nested, the right size and labeled as its method's;
the server zeroes the right number of neurons in 64 layers of [5120, 17408] (as each set is applied, which stops the
run on a wrong count, and again from cond.jsonl); every item answered in every condition; the unedited NLL within 0.001
of G1 (0.525521); the two unedited runs agree on NLL within 0.0005; the unedited scores within the same limits of the original scores
(MMLU 337 of 400, GSM8K 241 of 250, yes/no 38 or more).
Outputs: ~/scratch/wcn/runs/bench/: gen.jsonl and cond.jsonl (resumable) and bench_summary.json.
    python gpu7_bench.py     (with WCN_SERVERS and PYTHONPATH=$WCN_ROOT/code set, as gpu7_bench.sbatch does)
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common  # noqa: E402  (code/common.py)

BASE = Path(os.environ["WCN_BASE"])
common.QWEN_SNAP = json.loads((BASE / "paths.json").read_text())["qwen"]  # the tokenizer that prompt_ids() loads
from common import MAX_RESP, RES, JsonlStore, log, prompt_ids  # noqa: E402
from legit import YESNO  # noqa: E402  (code/legit.py)
from rate import One, nll  # noqa: E402  (code/rate.py)
from sidefx import greedy, score_gsm, score_mc, score_yn  # noqa: E402  (code/sidefx.py; functions only)

RUN = BASE / "runs" / "bench"
DATA = Path(os.environ["WCN_ROOT"]) / "data"
LADDER = ["claims-called-911", "fabricated-code-execution", "false-physical-embodiment", "laser-at-aircraft"]
METHODS = ["M1", "M2", "M3", "M4"]
SIZES = [10, 100]
NI = 17408
G1_BASE_NLL = 0.525521
UNEDITED = ("base_start", "base_end")
TASKS = ["yn", "mmlu", "gsm"]
SCORE = {"yn": score_yn, "mmlu": score_mc, "gsm": score_gsm}
MAX_TOK = {"yn": 8, "mmlu": 8, "gsm": MAX_RESP}  # code/sidefx.py:167-169
ORIG = {"yn": 40, "mmlu": 337, "gsm": 241}  # the original unedited scores (logs/sidefx.log:32): 1.000, 0.843, 0.964
MAX_DAMAGE, MAX_APART, YN_MIN = 0.023, {"mmlu": 14, "gsm": 6}, 38  # Step 6 rules (3.6 points of 400; 2.4 points of 250)
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append({"check": name, "pass": bool(ok), "detail": detail})
    print(f"{'PASS' if ok else 'FAIL'}  {name}  [{detail}]", flush=True)
    return bool(ok)


def read_last(path):
    """The reading rule of the repo's JsonlStore (one record per key, the last wins), without opening the repo's file for appending."""
    done = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                done[d["key"]] = d
    return done


def benchmarks():
    """The repo's sidefx.benchmarks() (code/sidefx.py:36-44), line for line, reading our pinned copies of MMLU and GSM8K."""
    m = pd.read_parquet(DATA / "mmlu/test.parquet").sample(400, random_state=0)
    mm = [{"id": f"m{i}", "text": f"{r.question}\n\n" + "\n".join(f"{'ABCD'[j]}. {c}" for j, c in enumerate(r.choices))
           + "\n\nAnswer with only the letter of the correct option.", "gold": "ABCD"[r.answer]} for i, r in enumerate(m.itertuples())]
    g = pd.read_parquet(DATA / "gsm8k/test-00000-of-00001.parquet").sample(250, random_state=0)
    gs = [{"id": f"g{i}", "text": r.question + "\n\nSolve it step by step, then give the final answer on its own line as 'Answer: <number>'.",
           "gold": r.answer.split("####")[-1].strip().replace(",", "")} for i, r in enumerate(g.itertuples())]
    yn = [{"id": f"y{i}", "text": q + " Answer with just Yes or No.", "gold": a} for i, (q, a) in enumerate(YESNO)]
    return {"yn": yn, "mmlu": mm, "gsm": gs}


def check_items(B):
    check("items: 40 yes/no, 400 MMLU, 250 GSM8K", [len(B[t]) for t in TASKS] == [40, 400, 250],
          str({t: len(B[t]) for t in TASKS}))
    H = read_last(RES / "sfx_gen.jsonl")
    got = {t: sum(SCORE[t](H[f"base|{t}|{x['id']}"]["samples"][0], x["gold"]) for x in B[t] if f"base|{t}|{x['id']}" in H)
           for t in TASKS}
    have = {t: sum(f"base|{t}|{x['id']}" in H for x in B[t]) for t in TASKS}
    check("the stored unedited answers score 40/40, 337/400 and 241/250 on these items, as in the original log",
          got == ORIG and have == {t: len(B[t]) for t in TASKS}, f"yes/no {got['yn']}/{have['yn']}, MMLU {got['mmlu']}/"
          f"{have['mmlu']}, GSM8K {got['gsm']}/{have['gsm']}")
    gen = read_last(RES / "stage_a_gen.jsonl")
    pr = common.load_prompts().set_index("prompt_id")
    bad = [pid for pid in sorted(gen) if prompt_ids(pr.loc[pid, "user_text"]) != list(gen[pid]["prompt_ids"])]
    check("our tokenizer and chat template rebuild the stored prompt token ids for all 403 Stage A prompts",
          not bad and len(gen) == 403, f"{len(gen) - len(bad)} of {len(gen)} equal; first different: {bad[:3]}")


def methods_on():
    v = json.loads((BASE / "runs" / "val" / "val_summary.json").read_text())
    ok = all(c["pass"] for c in v["checks"])
    check("validation run finished with every check passed", ok, f"{sum(c['pass'] for c in v['checks'])} of {len(v['checks'])}")
    on = [m for m in METHODS if v["methods"][m]["passes"]]
    print(f"info  methods that passed validation: {', '.join(on) if on else 'none'}", flush=True)
    return on if ok else []


def load_conditions(on):
    S = json.loads((BASE / "methods" / "m123_sets.json").read_text())
    S.update(json.loads((BASE / "methods" / "m4_sets.json").read_text()))
    C = {u: {"spec": {}, "b": None, "method": "unedited", "k": 0} for u in UNEDITED}
    bad = []
    for m in on:
        for b in LADDER:
            prev = set()
            for k in SIZES:
                key = f"{m}|{b}|{k}"
                s = S.get(key)
                if s is None:
                    bad.append(key)
                    continue
                flat = {int(l) * NI + int(i) for l, v in s["spec"].items() for i in v}
                if not (len(s["flat"]) == k == len(flat) == s["k"] and set(s["flat"]) == flat and prev <= flat
                        and s["b"] == b and s["kind"] == "top" and s["method"] == m
                        and (m != "M4" or s.get("m4_check") == "PASS")):
                    bad.append(key)
                prev = flat
                C[key] = {"spec": s["spec"], "b": b, "method": m, "k": k}
    n_expect = 2 + 8 * len(on)
    check("sets: sizes 10 and 100 for every method on and behavior, nested, the right size and labeled as its "
          "method's", not bad and len(C) == n_expect, f"{len(C)} conditions (expected {n_expect}); problems: {bad[:6]}")
    return C


def repair_tail(path):
    """If a run was killed in the middle of writing a line, drop that unfinished last line (our own files only)."""
    if path.exists():
        text = path.read_text()
        if text and not text.endswith("\n"):
            path.write_text(text[: text.rfind("\n") + 1])
            log(f"dropped an unfinished last line in {path.name}")


async def run(B, C):
    RUN.mkdir(parents=True, exist_ok=True)
    for n in ("gen", "cond"):
        repair_tail(RUN / f"{n}.jsonl")
    refs = json.loads((RES / "ref_texts.json").read_text())
    gs, cs = JsonlStore(RUN / "gen.jsonl"), JsonlStore(RUN / "cond.jsonl")
    servers = json.loads(Path(os.environ["WCN_SERVERS"]).read_text())
    qurls = [u for u, r in servers.items() if r == "qwen_abl"]
    if len(qurls) > 1:
        raise RuntimeError(f"STOP: expected one qwen_abl server, got {qurls}")
    ids = {t: {x["id"]: prompt_ids(x["text"]) for x in B[t]} for t in TASKS}
    rest = sorted(c for c in C if c not in UNEDITED)
    order = ["base_start"] + [rest[i] for i in np.random.default_rng(9).permutation(len(rest))] + ["base_end"]
    n_items = sum(len(B[t]) for t in TASKS)
    log(f"{len(C)} conditions, {n_items} items each; qwen {qurls}")

    def unfinished(c):
        return c not in cs.done or any(f"{c}|{t}|{x['id']}" not in gs.done for t in TASKS for x in B[t])

    left = [c for c in order if unfinished(c)]  # with none left, no server is contacted at all
    if left and not qurls:
        raise RuntimeError(f"STOP: {len(left)} conditions left to run but no qwen_abl server")
    q = One(qurls[0], 256) if qurls else None

    async def one(c, t, x):
        key = f"{c}|{t}|{x['id']}"
        if key not in gs.done:
            gs.add({"key": key, "samples": [await greedy(q, ids[t][x["id"]], MAX_TOK[t])]})

    for c in left:  # one condition at a time on the one server, the unedited model first and last
        t0 = time.time()
        r = (await q.post("/collective_rpc", {"method": "set_ablation", "args": [json.dumps(C[c]["spec"])]}))["results"][0]
        if r["zeroed"] != C[c]["k"] or r["layers"] != 64 or r["shape"] != [5120, NI] or r.get("scale") != 0.0:
            raise RuntimeError(f"STOP: the server zeroed the wrong neurons for {c}: {r}")
        if c not in cs.done:
            ns = await asyncio.gather(*[nll(q, x) for x in refs])
            cs.add({"key": c, "check": r, "nll": [n for n, _ in ns], "ntok": [t for _, t in ns]})
        todo = [(t, x) for t in TASKS for x in B[t] if f"{c}|{t}|{x['id']}" not in gs.done]
        await asyncio.gather(*[one(c, t, x) for t, x in todo])
        log(f"{c}: {len(todo)} of {n_items} items answered in {time.time() - t0:.0f} s")
    if left:
        await q.post("/collective_rpc", {"method": "set_ablation", "args": ["{}"]})
    log("all conditions done")


def summarize(B, C):
    gs, cs = read_last(RUN / "gen.jsonl"), read_last(RUN / "cond.jsonl")
    missing = [f"{c}|{t}|{x['id']}" for c in C for t in TASKS for x in B[t] if f"{c}|{t}|{x['id']}" not in gs]
    check("every item answered in every condition", not missing, f"missing {len(missing)}: {missing[:4]}")
    bad = [c for c in C if c not in cs or cs[c]["check"]["zeroed"] != C[c]["k"] or cs[c]["check"]["layers"] != 64
           or cs[c]["check"]["shape"] != [5120, NI] or cs[c]["check"].get("scale") != 0.0]
    check("the server zeroed exactly the set's neurons (0 for the unedited model) in 64 layers of [5120, 17408]",
          not bad, f"problems: {bad[:4]}" if bad else f"{len(C)} conditions, from cond.jsonl")
    if missing or bad:
        return {"stopped": "missing answers or condition records"}
    tok = lambda c: float(np.array(cs[c]["nll"]).sum() / np.array(cs[c]["ntok"]).sum())
    check("unedited NLL on the 200 reference texts within 0.001 of G1 (0.525521 nats per token)",
          abs(tok("base_start") - G1_BASE_NLL) <= 0.001, f"{tok('base_start'):.6f}")
    check("the two unedited runs agree on NLL within 0.0005 per token", abs(tok("base_start") - tok("base_end")) <= 0.0005,
          f"{tok('base_start'):.6f} vs {tok('base_end'):.6f}")
    base_nll = np.array(cs["base_start"]["nll"])
    dnll = {c: float((np.array(cs[c]["nll"]) - base_nll).sum() / np.array(cs[c]["ntok"]).sum()) for c in C}
    right = {c: {t: [bool(SCORE[t](gs[f"{c}|{t}|{x['id']}"]["samples"][0], x["gold"])) for x in B[t]] for t in TASKS}
             for c in C}
    sc = {c: {t: sum(right[c][t]) for t in TASKS} for c in C}
    u = sc["base_start"]
    check("unedited scores near the original: MMLU within 14 of 337, GSM8K within 6 of 241, yes/no 38 or more",
          abs(u["mmlu"] - ORIG["mmlu"]) <= MAX_APART["mmlu"] and abs(u["gsm"] - ORIG["gsm"]) <= MAX_APART["gsm"]
          and u["yn"] >= YN_MIN, f"ours: yes/no {u['yn']}/40, MMLU {u['mmlu']}/400, GSM8K {u['gsm']}/250")
    diff = {t: sum(a != b for a, b in zip(right["base_start"][t], right["base_end"][t])) for t in TASKS}
    print(f"info  the two unedited runs: yes/no {sc['base_end']['yn']}/40, MMLU {sc['base_end']['mmlu']}/400, GSM8K "
          f"{sc['base_end']['gsm']}/250 in the last run; items scored differently: {diff}", flush=True)
    H = read_last(RES / "sfx_gen.jsonl")  # the stored unedited answers: the one end-to-end test of the prompt text
    orig_text = {t: [H[f"base|{t}|{x['id']}"]["samples"][0] for x in B[t]] for t in TASKS}
    same = {t: sum(gs[f"base_start|{t}|{x['id']}"]["samples"][0] == a for x, a in zip(B[t], orig_text[t])) for t in TASKS}
    agree = {t: sum(r == bool(SCORE[t](a, x["gold"])) for r, x, a in zip(right["base_start"][t], B[t], orig_text[t]))
             for t in TASKS}
    print(f"info  our first unedited run against the stored unedited answers (same items and prompts, greedy): "
          f"identical text on yes/no {same['yn']}/40, MMLU {same['mmlu']}/400, GSM8K {same['gsm']}/250; scored the "
          f"same on {agree['yn']}/40, {agree['mmlu']}/400, {agree['gsm']}/250", flush=True)

    out = {"items": {t: len(B[t]) for t in TASKS}, "unedited": {c: sc[c] for c in UNEDITED}, "orig_unedited": ORIG,
           "vs_orig_answers": {"identical_text": same, "scored_the_same": agree},
           "rules": {"max_damage": MAX_DAMAGE, "max_apart": MAX_APART, "yn_min": YN_MIN}, "sets": {}, "dnll": dnll,
           "server_checks": {c: cs[c]["check"] for c in C}}
    print("\nEach set against the first unedited run (Step 6 rules: damage <= 0.023, MMLU within 14 of 400, GSM8K "
          "within 6 of 250, yes/no >= 38 of 40).\n'passes' follows the plan's wording, where a gain beyond a band also "
          "counts as outside; 'drops only' counts only drops beyond the bands.")
    print(f"  {'set':38s} {'yes/no':>7s} {'MMLU':>12s} {'GSM8K':>12s} {'damage':>9s}  passes; drops only")
    print(f"  {'unedited (first run)':38s} {u['yn']:5d}/40 {u['mmlu']:6d}/400    {u['gsm']:6d}/250    {0:9.5f}")
    sets = [c for c in C if c not in UNEDITED]
    for m in sorted({C[c]["method"] for c in sets}):
        for b in LADDER:
            for k in SIZES:
                c = f"{m}|{b}|{k}"
                if c not in C:
                    continue
                s = sc[c]
                rule = {"damage": dnll[c] <= MAX_DAMAGE, "mmlu": abs(s["mmlu"] - u["mmlu"]) <= MAX_APART["mmlu"],
                        "gsm": abs(s["gsm"] - u["gsm"]) <= MAX_APART["gsm"], "yn": s["yn"] >= YN_MIN}
                ok = all(rule.values())
                drop = {**rule, "mmlu": s["mmlu"] - u["mmlu"] >= -MAX_APART["mmlu"],
                        "gsm": s["gsm"] - u["gsm"] >= -MAX_APART["gsm"]}
                ok_drop = all(drop.values())
                out["sets"][c] = {**s, "dnll": dnll[c], "mmlu_apart": s["mmlu"] - u["mmlu"], "gsm_apart": s["gsm"] - u["gsm"],
                                  "rules": rule, "passes": ok, "rules_drops_only": drop, "passes_drops_only": ok_drop}
                print(f"  {c:38s} {s['yn']:5d}/40 {s['mmlu']:6d}/400 {s['mmlu'] - u['mmlu']:+3d} {s['gsm']:6d}/250 "
                      f"{s['gsm'] - u['gsm']:+3d} {dnll[c]:9.5f}  "
                      + ("yes" if ok else "NO: " + ", ".join(r for r, v in rule.items() if not v))
                      + ("; yes" if ok_drop else "; NO: " + ", ".join(r for r, v in drop.items() if not v)), flush=True)
    print("\nsets passing all four rules, per method (of 8): plan wording; drops only")
    for m in sorted({C[c]["method"] for c in sets}):
        n = sum(out["sets"][c]["passes"] for c in out["sets"] if C[c]["method"] == m)
        n2 = sum(out["sets"][c]["passes_drops_only"] for c in out["sets"] if C[c]["method"] == m)
        out.setdefault("passing", {})[m] = n
        out.setdefault("passing_drops_only", {})[m] = n2
        print(f"  {m}: {n}; {n2}")
    return out


def main():
    t0 = time.time()
    print("=== Items, prompt ids and sets", flush=True)
    B = benchmarks()
    check_items(B)
    on = methods_on()
    C = load_conditions(on)
    if not all(r["pass"] for r in RESULTS) or not on:
        print("STOP before any generation.", flush=True)
        return False
    print("\n=== Greedy answers and damage", flush=True)
    asyncio.run(run(B, C))
    print("\n=== Results", flush=True)
    out = summarize(B, C)
    out["checks"] = RESULTS
    out["methods"] = on
    out["minutes"] = round((time.time() - t0) / 60, 1)
    (RUN / "bench_summary.json").write_text(json.dumps(out, indent=1))
    n_ok = sum(r["pass"] for r in RESULTS)
    print(f"\n{n_ok} of {len(RESULTS)} checks passed in {(time.time() - t0) / 60:.0f} minutes; summary in {RUN / 'bench_summary.json'}")
    ok = n_ok == len(RESULTS)
    print("STEP 6 PART 1 (BENCHMARKS) PASS" if ok else "STEP 6 PART 1 (BENCHMARKS) FAIL.")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
