"""Validate the local judge against WeirdChat's stored labels (Qwen3.6-27B transcripts).

Per behavior, up to N_POS stored positives + N_NEG stored negatives (fixed seed). Conditions:
  on   : production config (thinking on)          -- all items
  on2  : independent repeat of `on`                -- all items (judge test-retest ceiling)
  off  : thinking off                              -- all items
  rfoff: rubric-first block order (cacheable prefix), thinking off -- all items
Decision rule (fixed before results): use the cheapest condition whose agreement with WeirdChat
labels is within 0.02 of `on` and whose kappa with `on` is >= kappa(on, on2) - 0.05; else `on`.
Writes results/judge_val.jsonl and prints agreement / Cohen's kappa / cost per condition.
"""
import asyncio
import json
import time

import numpy as np

from common import RES, JsonlStore, Pool, judge, load_transcripts, log

N_POS, N_NEG, SEED = 40, 40, 0


def kappa(a, b):
    a, b = np.array(a, bool), np.array(b, bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


async def main():
    t = load_transcripts()
    rng = np.random.default_rng(SEED)
    items = []
    for b, g in t.groupby("behavior_id"):
        for lab, n in ((True, N_POS), (False, N_NEG)):
            gg = g[g.wc_match == lab]
            idx = rng.choice(len(gg), size=min(n, len(gg)), replace=False)
            items += [gg.iloc[i] for i in idx]
    log(f"{len(items)} items")
    store = JsonlStore(RES / "judge_val.jsonl")
    pool = Pool("gemma", 192)
    t0 = time.time()

    async def one(r, cond):
        key = f"{r.transcript_id}|{cond}"
        if key in store.done:
            return
        msgs = r.messages
        user = [m for m in msgs if m["role"] == "user"][0]["content"]
        resp = [m for m in msgs if m["role"] == "assistant"][0]["content"]
        j = await judge(pool, r.behavior_id, user, resp, thinking=cond in ("on", "on2"),
                        order="rubric_first" if cond == "rfoff" else "prod")
        store.add({"key": key, "transcript_id": r.transcript_id, "behavior_id": r.behavior_id, "cond": cond,
                   "wc_match": bool(r.wc_match), **j})

    await asyncio.gather(*[one(r, c) for c in ("on", "off", "rfoff", "on2") for r in items])
    log(f"done in {time.time() - t0:.0f}s")

    recs = list(store.done.values())
    by = {}
    for d in recs:
        by.setdefault(d["cond"], {})[d["transcript_id"]] = d
    for c in ("on", "on2", "off", "rfoff"):
        ds = [d for d in by.get(c, {}).values() if d["match"] is not None]
        a = [d["match"] for d in ds]
        w = [d["wc_match"] for d in ds]
        tp = sum(x and y for x, y in zip(a, w)); fp = sum(x and not y for x, y in zip(a, w))
        fn = sum((not x) and y for x, y in zip(a, w))
        print(f"{c:4s} n={len(ds)} agree={np.mean(np.array(a) == np.array(w)):.3f} kappa={kappa(a, w):.3f} "
              f"P={tp / max(tp + fp, 1):.3f} R={tp / max(tp + fn, 1):.3f} errors={sum(d['match'] is None for d in by.get(c, {}).values())} "
              f"mean_out_tok={np.mean([d['n_out'] for d in ds]):.0f} mean_in_tok={np.mean([d['n_in'] for d in ds]):.0f}")
    common = set(by.get("on", {})) & set(by.get("on2", {}))
    ok = [i for i in common if by["on"][i]["match"] is not None and by["on2"][i]["match"] is not None]
    print("test-retest on vs on2: agree=%.3f kappa=%.3f" % (
        np.mean([by["on"][i]["match"] == by["on2"][i]["match"] for i in ok]),
        kappa([by["on"][i]["match"] for i in ok], [by["on2"][i]["match"] for i in ok])))
    for c in ("off", "rfoff"):
        common = set(by.get("on", {})) & set(by.get(c, {}))
        ok = [i for i in common if by["on"][i]["match"] is not None and by[c][i]["match"] is not None]
        print("on vs %s: agree=%.3f kappa=%.3f" % (c,
            np.mean([by["on"][i]["match"] == by[c][i]["match"] for i in ok]),
            kappa([by["on"][i]["match"] for i in ok], [by[c][i]["match"] for i in ok])))
    print("\nper behavior (agreement with WeirdChat labels):")
    for b in sorted({d["behavior_id"] for d in recs}):
        row = []
        for c in ("on", "on2", "off", "rfoff"):
            ds = [d for d in by.get(c, {}).values() if d["behavior_id"] == b and d["match"] is not None]
            row.append(f"{c}={np.mean([d['match'] == d['wc_match'] for d in ds]):.2f}")
        print(f"  {b:30s} " + " ".join(row))


asyncio.run(main())
