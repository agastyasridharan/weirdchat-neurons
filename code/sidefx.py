"""Phase 2c: side effects of the top-100 neuron sets, and whether amplifying them causes the behavior (cenv).

Conditions (vLLM in-place weight edits through ablate_ext; prefix caching off):
  base
  <b>|top100            zeroed     -> legitimate-use sets (legit.py), first-person rate, yes/no facts, MMLU, GSM8K
  <b>|top100|x2,x4,x8   amplified  -> behavior rate on that behavior's LOW-rate WeirdChat prompts (Stage A on < 0.1)
  <b>|rand100|x8        amplified matched random set (sets.py), same prompts
Every condition also gets the damage measure (NLL on rate.py's 200 reference replies) and V at the 46 forks.
Replies are sampled as in WeirdChat (T=1, 1024 tokens; 8 per prompt); yes/no, MMLU and GSM8K are greedy.

  python sidefx.py run | summary
"""
import asyncio
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow as pa

import common
import legit
from common import MAX_RESP, RES, JsonlStore, Pool, generate, judge, load_prompts, load_rubrics, log, prompt_ids
from rate import B5, One, fork_V, nll

N = 8
ALPHAS = [2, 4, 8]
N_MMLU, N_GSM = 400, 250
MMLU = RES.parent / "data/mmlu/test.parquet"  # cais/mmlu all/test @c30699e (the shared HF cache copy is a dangling symlink)
GSM = "/data/hf_cache/datasets/openai___gsm8k/main/0.0.0/cc7b047b6e5bb11b4f1af84efc572db110a51b3c/gsm8k-test.arrow"


def benchmarks():
    m = pd.read_parquet(MMLU).sample(N_MMLU, random_state=0)
    mm = [{"id": f"m{i}", "text": f"{r.question}\n\n" + "\n".join(f"{'ABCD'[j]}. {c}" for j, c in enumerate(r.choices))
           + "\n\nAnswer with only the letter of the correct option.", "gold": "ABCD"[r.answer]} for i, r in enumerate(m.itertuples())]
    g = pa.ipc.open_stream(open(GSM, "rb")).read_all().to_pandas().sample(N_GSM, random_state=0)
    gs = [{"id": f"g{i}", "text": r.question + "\n\nSolve it step by step, then give the final answer on its own line as 'Answer: <number>'.",
           "gold": r.answer.split("####")[-1].strip().replace(",", "")} for i, r in enumerate(g.itertuples())]
    yn = [{"id": f"y{i}", "text": q + " Answer with just Yes or No.", "gold": a} for i, (q, a) in enumerate(legit.YESNO)]
    return mm, gs, yn


def low_prompts():
    J = JsonlStore(RES / "stage_a_judge_on.jsonl").done
    pr = load_prompts().set_index("prompt_id")
    m = defaultdict(list)
    for j in J.values():
        if j["match"] is not None:
            m[j["prompt_id"]].append(j["match"])
    return [{"pid": pid, "b": pr.loc[pid, "behavior_id"], "user": pr.loc[pid, "user_text"], "stage_a": float(np.mean(v))}
            for pid, v in sorted(m.items()) if pr.loc[pid, "behavior_id"] in B5 and np.mean(v) < 0.1]


def conditions():
    sets = json.loads((RES / "ablation_sets.json").read_text())
    C = {"base": {"spec": {}, "tasks": ["legit", "bench", "amp_all"]}}
    for b in B5:
        C[f"{b}|top100"] = {"spec": sets[f"{b}|top100"]["spec"], "tasks": ["legit", "bench"]}
    for b in B5:
        for a in ALPHAS:
            C[f"{b}|top100|x{a}"] = {"spec": {"scale": a, "spec": sets[f"{b}|top100"]["spec"]}, "tasks": ["amp"], "b": b}
        C[f"{b}|rand100|x8"] = {"spec": {"scale": 8, "spec": sets[f"{b}|rand100"]["spec"]}, "tasks": ["amp"], "b": b}
    return C


def score_mc(text, gold):
    m = re.match(r"\W*([ABCD])\b", text.strip())
    return m is not None and m.group(1) == gold


def score_gsm(text, gold):
    m = re.findall(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)", text) or re.findall(r"(-?[\d,]*\.?\d+)", text)
    if not m:
        return False
    try:
        return abs(float(m[-1].replace(",", "")) - float(gold)) < 1e-6
    except ValueError:
        return False


def score_yn(text, gold):
    w = re.sub(r"[^a-z]", " ", text.lower()).split()
    return bool(w) and w[0] == gold


async def greedy(pool, ids, max_tokens):
    body = {"model": "qwen", "prompt": ids, "max_tokens": max_tokens, "temperature": 0.0, "skip_special_tokens": True}
    out = await pool.post("/v1/completions", body)
    return out["choices"][0]["text"]


async def run():
    common.RUBRICS = load_rubrics() | legit.RUBRICS
    C = conditions()
    gen = JsonlStore(RES / "stage_a_gen.jsonl").done
    F = [json.loads(l) for l in (RES / "phase2_forks.jsonl").read_text().splitlines()]
    fork_ids = []
    for f in F:
        pid, si = f["rid"].split("|")
        fork_ids.append(gen[pid]["prompt_ids"] + list(gen[pid]["samples"][int(si)]["ids"])[:f["fork"][0]])
    refs = json.loads((RES / "ref_texts.json").read_text())
    mm, gsm, yn = benchmarks()
    LOW = low_prompts()
    LEG = [{"id": f"{s}:{i}", "set": s, "rubrics": rub, "text": t} for s, (_, rub, ps) in legit.JUDGED.items() for i, t in enumerate(ps)]
    SELF = [{"id": f"self:{i}", "text": t} for i, t in enumerate(legit.SELF)]
    gs, js, cs = (JsonlStore(RES / f"sfx_{n}.jsonl") for n in ("gen", "judge", "cond"))
    jpool = Pool("gemma", 96)
    qurls = [u for u, r in json.loads(open(os.environ["WCN_SERVERS"]).read()).items() if r == "qwen_abl"]
    log(f"{len(C)} conditions; legit {len(LEG)} + self {len(SELF)} prompts; yes/no {len(yn)}, MMLU {len(mm)}, GSM8K {len(gsm)}; "
        f"low-rate prompts {len(LOW)}; qwen {qurls}")
    jtasks, queued = [], set()

    def enqueue(key, rubric, user, text):
        k = f"{key}|{rubric}"
        if k not in js.done and k not in queued:
            queued.add(k)
            jtasks.append(asyncio.create_task(judge_one(k, rubric, user, text)))

    async def judge_one(k, rubric, user, text):
        r = await judge(jpool, rubric, user, text, thinking=True)
        js.add({"key": k, "match": r["match"], "explanation": r["explanation"]})

    def judge_all(c):
        for x in LEG:
            for i, s in enumerate(gs.done.get(f"{c}|legit|{x['id']}", {}).get("samples", [])):
                for rub in x["rubrics"]:
                    enqueue(f"{c}|legit|{x['id']}|{i}", rub, x["text"], s)
        for p in LOW:
            for i, s in enumerate(gs.done.get(f"{c}|amp|{p['pid']}", {}).get("samples", [])):
                enqueue(f"{c}|amp|{p['pid']}|{i}", p["b"], p["user"], s)

    queue = asyncio.Queue()
    for c in C:
        queue.put_nowait(c)
        judge_all(c)

    async def worker(url):
        q = One(url, 256)
        while not queue.empty():
            c = queue.get_nowait()
            spec = C[c]["spec"]
            r = (await q.post("/collective_rpc", {"method": "set_ablation", "args": [json.dumps(spec)]}))["results"][0]
            inner = spec.get("spec", spec) if "scale" in spec else spec
            assert r["zeroed"] == sum(len(v) for v in inner.values()), (c, r)
            if c not in cs.done:
                ns = await asyncio.gather(*[nll(q, x) for x in refs])
                vs = await asyncio.gather(*[fork_V(q, f, ids) for f, ids in zip(F, fork_ids)])
                cs.add({"key": c, "check": r, "nll": [n for n, _ in ns], "ntok": [t for _, t in ns], "V": vs})
            jobs = []

            async def sample(key, ids, n=N):
                if key not in gs.done:
                    out = await generate(q, ids, n, MAX_RESP)
                    gs.add({"key": key, "samples": [o["text"] for o in out], "n": [len(o["ids"] or []) for o in out]})

            async def greedy_one(key, text, max_tokens):
                if key not in gs.done:
                    gs.add({"key": key, "samples": [await greedy(q, prompt_ids(text), max_tokens)]})
            if "legit" in C[c]["tasks"]:
                jobs += [sample(f"{c}|legit|{x['id']}", prompt_ids(x["text"])) for x in LEG]
                jobs += [sample(f"{c}|self|{x['id']}", prompt_ids(x["text"])) for x in SELF]
            if "bench" in C[c]["tasks"]:
                jobs += [greedy_one(f"{c}|yn|{x['id']}", x["text"], 8) for x in yn]
                jobs += [greedy_one(f"{c}|mmlu|{x['id']}", x["text"], 8) for x in mm]
                jobs += [greedy_one(f"{c}|gsm|{x['id']}", x["text"], MAX_RESP) for x in gsm]
            amp = LOW if "amp_all" in C[c]["tasks"] else [p for p in LOW if "amp" in C[c]["tasks"] and p["b"] == C[c]["b"]]
            jobs += [sample(f"{c}|amp|{p['pid']}", gen[p["pid"]]["prompt_ids"]) for p in amp]
            await asyncio.gather(*jobs)
            judge_all(c)
            log(f"{url[-4:]} {c}: done generating; judged so far {len(js.done)}")
        await q.post("/collective_rpc", {"method": "set_ablation", "args": ["{}"]})

    await asyncio.gather(*[worker(u) for u in qurls])
    log(f"generation done; waiting for {sum(not t.done() for t in jtasks)} judgments")
    await asyncio.gather(*jtasks)
    log("done")


def summary():
    C = conditions()
    gs, js, cs = (JsonlStore(RES / f"sfx_{n}.jsonl").done for n in ("gen", "judge", "cond"))
    mm, gsm, yn = benchmarks()
    LOW = low_prompts()
    base_nll = np.array(cs["base"]["nll"])

    def dnll(c):
        return (np.array(cs[c]["nll"]) - base_nll).sum() / np.array(cs[c]["ntok"]).sum() if c in cs else float("nan")

    def jrate(c, s, rub):
        m = [js.get(f"{c}|legit|{s}:{i}|{k}|{rub}", {}).get("match") for i in range(16) for k in range(N)]
        m = [x for x in m if x is not None]
        return (sum(m) / len(m), len(m)) if m else (float("nan"), 0)

    def selfrate(c):
        t = [x for i in range(len(legit.SELF)) for x in gs.get(f"{c}|self|self:{i}", {}).get("samples", [])]
        w = sum(len(x.split()) for x in t)
        fp = sum(len(re.findall(r"\b(I|me|my|myself|I'm|I've|I'd|I'll)\b", x)) for x in t)
        return 100 * fp / max(w, 1), np.mean([bool(re.search(r"\b(I|me|my)\b", x)) for x in t]) if t else float("nan")

    def acc(c, task, items, fn):
        v = [fn(gs[f"{c}|{task}|{x['id']}"]["samples"][0], x["gold"]) for x in items if f"{c}|{task}|{x['id']}" in gs]
        return (np.mean(v), len(v)) if v else (float("nan"), 0)

    print("== side effects of zeroing each top-100 set (legitimate behavior rates; higher is better unless noted)")
    cols = [("emergency", "legit-911-advice", "911 advice"), ("ai_question", "legit-ai-yes", "says AI"), ("no_body", "legit-no-body", "no body"),
            ("imagine", "legit-imagine", "imagines"), ("stars", "legit-laser-mention", "laser mention"), ("stars", "legit-laser-safety", "aircraft warn")]
    print("condition".ljust(32) + "".join(n.rjust(14) for _, _, n in cols) + "  1st-pers/100w  yes/no   MMLU  GSM8K  dNLL/tok")
    for c in ["base"] + [f"{b}|top100" for b in B5]:
        row = "".join(f"{jrate(c, s, r)[0]:.3f} ({jrate(c, s, r)[1]})".rjust(14) for s, r, _ in cols)
        fp, _ = selfrate(c)
        print(c[:32].ljust(32) + row + f"  {fp:12.2f}  {acc(c, 'yn', yn, score_yn)[0]:.3f}  {acc(c, 'mmlu', mm, score_mc)[0]:.3f}  "
              f"{acc(c, 'gsm', gsm, score_gsm)[0]:.3f}  {dnll(c):.4f}")
    print("\n== amplification: behavior rate on each behavior's low-rate prompts (Stage A on-rate < 0.1)")
    for b in B5:
        ps = [p for p in LOW if p["b"] == b]

        def rate(c):
            m = [js.get(f"{c}|amp|{p['pid']}|{i}|{b}", {}).get("match") for p in ps for i in range(N)]
            m = [x for x in m if x is not None]
            return (sum(m) / len(m), len(m)) if m else (float("nan"), 0)
        cells = [("base", rate("base"))] + [(f"x{a}", rate(f"{b}|top100|x{a}")) for a in ALPHAS] + [("rand x8", rate(f"{b}|rand100|x8"))]
        dm = [f"{dnll(f'{b}|top100|x{a}'):.3f}" for a in ALPHAS] + [f"{dnll(f'{b}|rand100|x8'):.3f}"]
        print(f"  {b[:28]:28s} prompts {len(ps):2d}  " + "  ".join(f"{n} {r:.3f} ({k})" for n, (r, k) in cells) + f"   dNLL {' / '.join(dm)}")


def export():
    """results/site_sidefx.json for the dashboard: side-effect table, amplification rates with prompt-bootstrap CIs."""
    from common import get_split
    C = conditions()
    gs, js, cs = (JsonlStore(RES / f"sfx_{n}.jsonl").done for n in ("gen", "judge", "cond"))
    mm, gsm, yn = benchmarks()
    LOW, sp = low_prompts(), get_split()
    fpids = {json.loads(l)["prompt_id"] for l in (RES / "phase2_forks.jsonl").read_text().splitlines()}
    base_nll, rng = np.array(cs["base"]["nll"]), np.random.default_rng(0)

    def dnll(c):
        return float((np.array(cs[c]["nll"]) - base_nll).sum() / np.array(cs[c]["ntok"]).sum())

    def boot(m):  # m: prompts x samples (nan = unjudged) -> mean, 95% CI over prompts
        pr = np.nanmean(m, 1)
        bs = [np.nanmean(pr[rng.integers(0, len(pr), len(pr))]) for _ in range(2000)]
        return [float(np.nanmean(pr)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

    def acc(c, task, items, fn):
        return float(np.mean([fn(gs[f"{c}|{task}|{x['id']}"]["samples"][0], x["gold"]) for x in items]))
    LEGIT = [("emergency", "legit-911-advice"), ("ai_question", "legit-ai-yes"), ("no_body", "legit-no-body"),
             ("imagine", "legit-imagine"), ("stars", "legit-laser-mention"), ("stars", "legit-laser-safety")]
    side = []
    for c in ["base"] + [f"{b}|top100" for b in B5]:
        t = [x for i in range(len(legit.SELF)) for x in gs[f"{c}|self|self:{i}"]["samples"]]
        jm = lambda s, i, k, r: js.get(f"{c}|legit|{s}:{i}|{k}|{r}", {}).get("match")
        side.append({"c": c, "b": c.split("|")[0] if c != "base" else None,
                     "legit": {r: boot(np.array([[jm(s, i, k, r) for k in range(N)] for i in range(16)], dtype=float)) for s, r in LEGIT},
                     "self_fp": 100 * sum(len(re.findall(r"\b(I|me|my|myself|I'm|I've|I'd|I'll)\b", x)) for x in t) / sum(len(x.split()) for x in t),
                     "yn": acc(c, "yn", yn, score_yn), "mmlu": acc(c, "mmlu", mm, score_mc), "gsm": acc(c, "gsm", gsm, score_gsm),
                     "dnll": dnll(c) if c != "base" else 0.0})
    cond = {}
    for c in ["base", "laser-at-aircraft|top100"]:
        lm = np.array([js[f"{c}|legit|stars:{i}|{k}|legit-laser-mention"]["match"] for i in range(16) for k in range(N)], bool)
        aw = np.array([js[f"{c}|legit|stars:{i}|{k}|legit-laser-safety"]["match"] for i in range(16) for k in range(N)], bool)
        cond[c] = {"mention": int(lm.sum()), "warn_given_mention": float(aw[lm].mean()), "warn_no_mention": float(aw[~lm].mean()), "n": len(lm)}
    amp = []
    for b in B5:
        ps = [p for p in LOW if p["b"] == b]
        clean = [p for p in ps if sp[p["pid"]]["split"] != "discovery" and p["pid"] not in fpids]
        cs_ = ["base"] + [f"{b}|top100|x{a}" for a in ALPHAS] + [f"{b}|rand100|x8"]
        mat = lambda c, P: np.array([[js.get(f"{c}|amp|{p['pid']}|{i}|{b}", {}).get("match") for i in range(N)] for p in P], dtype=float)
        amp.append({"b": b, "n": len(ps), "n_clean": len(clean), "conds": ["base", "x2", "x4", "x8", "rand x8"],
                    "all": [boot(mat(c, ps)) for c in cs_], "clean": [boot(mat(c, clean)) for c in cs_],
                    "dnll": [0.0] + [dnll(c) for c in cs_[1:]],
                    "up_x2": int((np.nanmean(mat(cs_[1], ps), 1) > np.nanmean(mat("base", ps), 1)).sum())})
    out = {"side": side, "laser_cond": cond, "amp": amp, "n_mmlu": len(mm), "n_gsm": len(gsm), "n_yn": len(yn),
           "n_legit": sum(len(v[2]) for v in legit.JUDGED.values()), "n_self": len(legit.SELF), "n_low": len(LOW),
           "judged": sum(1 for v in js.values() if v["match"] is not None)}
    (RES / "site_sidefx.json").write_text(json.dumps(out, indent=1))
    print("wrote", RES / "site_sidefx.json", out["judged"], "judged")


if __name__ == "__main__":
    {"run": lambda: asyncio.run(run()), "summary": summary, "export": export}[sys.argv[1]]()
