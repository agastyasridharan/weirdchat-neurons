"""Phase 2b: behavior rate under neuron ablation (cenv; Qwen servers with ablate_ext, Gemma judges).

Conditions: base (no ablation) and every set in results/ablation_sets.json (sets.py: 5 behaviors x top10/100/1000
and matched rand100/1000). Prompts: every prompt of those 5 behaviors with Stage A thinking-on rate >= 0.1 (both
splits; split recorded), N samples each with WeirdChat sampling. Each condition runs on ALL 5 behaviors' prompts,
which gives the off-target matrix. Every reply is judged with its own prompt's rubric, thinking on.

Per condition also:
  damage: per-token NLL of 200 reference replies (Stage A, other behaviors). They were sampled from the base
          model, so the mean NLL increase over base estimates KL(base || ablated) per token.
  forks:  V at all 46 Phase 2 forks from vLLM top-20 logprobs (fork-level effect of each set).
Each Qwen server holds one condition at a time: set_ablation -> NLL + forks -> generate. Judging runs alongside.

  python rate.py run       # resumable
  python rate.py summary
"""
import asyncio
import json
import math
import sys
from collections import defaultdict

import numpy as np

from common import MAX_RESP, RES, JsonlStore, Pool, generate, get_split, judge, load_prompts, log, next_token_logprobs

N = 8
MIN_RATE = 0.1
B5 = ["claims-called-911", "denying-ai-identity", "fabricated-code-execution", "false-physical-embodiment",
      "laser-at-aircraft"]


class One(Pool):
    """A pool pinned to one server (its ablation state is per server)."""

    def __init__(self, url, per_server):
        super().__init__("none", per_server)
        self.cfg, self.urls, self.load = {url: {"max": None, "key": None}}, [url], {url: 0}

    def _reload(self):
        pass


def prompts():
    J = JsonlStore(RES / "stage_a_judge_on.jsonl").done
    pr = load_prompts().set_index("prompt_id")
    sp = get_split()
    m = defaultdict(list)
    for j in J.values():
        if j["match"] is not None:
            m[j["prompt_id"]].append(j["match"])
    out = []
    for pid, v in sorted(m.items()):
        b = pr.loc[pid, "behavior_id"]
        if b in B5 and np.mean(v) >= MIN_RATE:
            out.append({"pid": pid, "b": b, "split": sp[pid]["split"], "user": pr.loc[pid, "user_text"], "stage_a": float(np.mean(v))})
    return out


def conditions():
    sets = json.loads((RES / "ablation_sets.json").read_text())
    return {"base": {}} | {c: s["spec"] for c, s in sets.items()}


async def nll(pool, ref):
    body = {"model": "qwen", "prompt": ref["ids"], "max_tokens": 1, "prompt_logprobs": 0,
            "return_tokens_as_token_ids": True, "skip_special_tokens": False}
    out = await pool.post("/v1/completions", body)
    tot = 0.0
    for t, d in enumerate(out["choices"][0]["prompt_logprobs"]):
        if t < ref["prompt_len"] or d is None:
            continue
        hit = [v for k, v in d.items() if k.split(":")[-1] == str(ref["ids"][t])] or list(d.values())
        tot -= hit[0]["logprob"]
    return tot, len(ref["ids"]) - ref["prompt_len"]


async def fork_V(pool, f, ids):
    r = await next_token_logprobs(pool, ids, k=20)
    top = {int(k.split(":")[1]): v for k, v in r["logprobs"]["top_logprobs"][0].items()}
    p = np.array([math.exp(top[x["tok"]]) if x["tok"] in top else 0.0 for x in f["next_tokens"]])
    q = np.clip([x["q"] for x in f["next_tokens"]], .02, .98)
    return float((p * q).sum() / p.sum()) if p.sum() > 0 else None


async def run():
    P = prompts()
    C = conditions()
    gen = JsonlStore(RES / "stage_a_gen.jsonl").done
    F = [json.loads(l) for l in (RES / "phase2_forks.jsonl").read_text().splitlines()]
    fork_ids = []
    for f in F:
        pid, si = f["rid"].split("|")
        fork_ids.append(gen[pid]["prompt_ids"] + list(gen[pid]["samples"][int(si)]["ids"])[:f["fork"][0]])
    refs = json.loads((RES / "ref_texts.json").read_text())
    gs, js, cs = JsonlStore(RES / "rate_gen.jsonl"), JsonlStore(RES / "rate_judge.jsonl"), JsonlStore(RES / "rate_cond.jsonl")
    jpool = Pool("gemma", 96)
    qurls = [u for u, r in json.loads(open(__import__("os").environ["WCN_SERVERS"]).read()).items() if r == "qwen_abl"]
    log(f"{len(P)} prompts ({', '.join(f'{b[:12]} {sum(p['b'] == b for p in P)}' for b in B5)}), {N} samples; "
        f"{len(C)} conditions; qwen servers {qurls}")
    jtasks, queued = [], set()

    def enqueue(c, p, i, text):
        k = f"{c}|{p['pid']}|{i}"
        if k not in js.done and k not in queued:
            queued.add(k)
            jtasks.append(asyncio.create_task(judge_one(c, p, i, text)))

    async def judge_one(c, p, i, text):
        k = f"{c}|{p['pid']}|{i}"
        r = await judge(jpool, p["b"], p["user"], text, thinking=True)
        js.add({"key": k, "match": r["match"], "explanation": r["explanation"]})

    queue = asyncio.Queue()
    for c in C:
        if c in cs.done and all(f"{c}|{p['pid']}" in gs.done for p in P):
            for p in P:
                for i, s in enumerate(gs.done[f"{c}|{p['pid']}"]["samples"]):
                    enqueue(c, p, i, s["text"])
        else:
            queue.put_nowait(c)

    async def worker(url):
        q = One(url, 256)
        while not queue.empty():
            c = queue.get_nowait()
            todo = [p for p in P if f"{c}|{p['pid']}" not in gs.done]
            if c not in cs.done or todo:
                r = await q.post("/collective_rpc", {"method": "set_ablation", "args": [json.dumps(C[c])]})
                chk = r["results"][0]
                want = sum(len(v) for v in C[c].values())
                assert chk["zeroed"] == want, (c, chk, want)
                if c not in cs.done:
                    ns = await asyncio.gather(*[nll(q, x) for x in refs])
                    vs = await asyncio.gather(*[fork_V(q, f, ids) for f, ids in zip(F, fork_ids)])
                    cs.add({"key": c, "check": chk, "nll": [n for n, _ in ns], "ntok": [t for _, t in ns], "V": vs})

                async def one(p):
                    out = await generate(q, gen[p["pid"]]["prompt_ids"], N, MAX_RESP)
                    gs.add({"key": f"{c}|{p['pid']}", "samples": [{"text": o["text"], "n": len(o["ids"] or []),
                                                                    "finish": o["finish"]} for o in out]})
                    for i, o in enumerate(out):
                        enqueue(c, p, i, o["text"])
                await asyncio.gather(*[one(p) for p in todo])
                log(f"{url[-4:]} {c}: generated {len(todo)} prompts; judged so far {len(js.done)}")
            for p in P:
                for i, s in enumerate(gs.done[f"{c}|{p['pid']}"]["samples"]):
                    enqueue(c, p, i, s["text"])
        await q.post("/collective_rpc", {"method": "set_ablation", "args": ["{}"]})

    assert qurls or queue.empty(), "conditions left to generate but no qwen_abl server"
    await asyncio.gather(*[worker(u) for u in qurls])
    log(f"generation done; waiting for {sum(not t.done() for t in jtasks)} judgments")
    await asyncio.gather(*jtasks)
    log("done")


def summary():
    P = prompts()
    C = conditions()
    gs, js = JsonlStore(RES / "rate_gen.jsonl").done, JsonlStore(RES / "rate_judge.jsonl").done
    cs = JsonlStore(RES / "rate_cond.jsonl").done
    F = [json.loads(l) for l in (RES / "phase2_forks.jsonl").read_text().splitlines()]
    sets = json.loads((RES / "ablation_sets.json").read_text())
    base_nll = np.array(cs["base"]["nll"]) if "base" in cs else None
    rng = np.random.default_rng(0)

    def rate(c, b, split=None):
        per = []
        for p in P:
            if p["b"] != b or (split and p["split"] != split) or f"{c}|{p['pid']}" not in gs:
                continue
            m = [js.get(f"{c}|{p['pid']}|{i}", {}).get("match") for i in range(N)]
            m = [x for x in m if x is not None]
            if m:
                per.append((sum(m), len(m)))
        return per

    def ratio_ci(c, b):
        """rate(c)/rate(base) with a prompt-cluster bootstrap (same prompts in both)."""
        a, z = rate(c, b), rate("base", b)
        if not a or len(a) != len(z):
            return None
        a, z = np.array(a), np.array(z)
        r = a[:, 0].sum() / a[:, 1].sum() / max(1e-9, z[:, 0].sum() / z[:, 1].sum())
        bs = []
        for _ in range(2000):
            ix = rng.integers(len(a), size=len(a))
            den = z[ix, 0].sum() / z[ix, 1].sum()
            if den > 0:
                bs.append(a[ix, 0].sum() / a[ix, 1].sum() / den)
        return r, np.percentile(bs, 2.5), np.percentile(bs, 97.5)

    print(f"{len(P)} prompts x {N}; judged {len(js)}; rates are pooled k/n over prompts with Stage A on-rate >= {MIN_RATE}\n")
    hdr = "condition".ljust(34) + "".join(b[:12].rjust(14) for b in B5) + "   dNLL/tok  len  V at own forks"
    print(hdr)
    for c in C:
        cells = []
        for b in B5:
            per = rate(c, b)
            k, n = sum(x for x, _ in per), sum(y for _, y in per)
            cells.append(f"{k / n:.3f} ({n})".rjust(14) if n else "-".rjust(14))
        extra = ""
        if c in cs and base_nll is not None:
            ntok = np.array(cs[c]["ntok"])
            d = (np.array(cs[c]["nll"]) - base_nll).sum() / ntok.sum()
            ln = np.mean([s["n"] for p in P if f"{c}|{p['pid']}" in gs for s in gs[f"{c}|{p['pid']}"]["samples"]])
            own = sets[c]["forks"] if c in sets else range(len(F))
            v = [(cs[c]["V"][i], cs["base"]["V"][i]) for i in own if cs[c]["V"][i] is not None and cs["base"]["V"][i] is not None]
            extra = f"   {d:8.4f} {ln:4.0f}  {np.mean([x for x, _ in v]):.3f} (base {np.mean([y for _, y in v]):.3f})"
        print(c[:34].ljust(34) + "".join(cells) + extra)
    print("\nown-behavior rate ratio vs base [95% prompt-bootstrap CI]; held-out-split rates in brackets")
    for c, s in sets.items():
        r = ratio_ci(c, s["b"])
        ho, hb = rate(c, s["b"], "heldout"), rate("base", s["b"], "heldout")
        f = lambda per: sum(x for x, _ in per) / max(1, sum(y for _, y in per))
        if r:
            print(f"  {c:34s} {r[0]:.2f} [{r[1]:.2f}, {r[2]:.2f}]   held-out {f(ho):.3f} vs base {f(hb):.3f}")


if __name__ == "__main__":
    asyncio.run(run()) if sys.argv[1] == "run" else summary()
