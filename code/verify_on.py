"""Re-judge fork confirmation samples with the thinking-on (production) judge.

The fork search used the thinking-off judge for throughput. For a random subset of fork records, the
CONFIRM continuations on both sides of the fork (prefix y_<=t* and y_<=t*+1) are re-judged with
thinking on, giving P_on(t*), P_on(t*+1) to compare against the thinking-off values.

  python verify_on.py [N_FORKS]
"""
import asyncio
import random
import sys

import numpy as np

from common import RES, JsonlStore, Pool, judge, load_prompts, log, tok

N = int(sys.argv[1]) if len(sys.argv) > 1 else 80


async def main():
    forks = sorted(JsonlStore(RES / "forks_off.jsonl").done.values(), key=lambda f: f["rid"])
    samples = JsonlStore(RES / "forks_samples_off.jsonl").done
    gen = JsonlStore(RES / "stage_a_gen.jsonl").done
    pr = load_prompts().set_index("prompt_id")
    random.Random(0).shuffle(forks)
    store = JsonlStore(RES / "verify_on.jsonl")
    pool = Pool("gemma", 96)
    T = tok()
    todo = []
    for f in forks[:N]:
        resp = list(gen[f["prompt_id"]]["samples"][int(f["rid"].split("|")[1])]["ids"])
        for side, c in (("lo", f["fork"][0]), ("hi", f["fork"][1])):
            rec = samples.get(f"{f['rid']}|{c}|confirm")
            if rec is None:  # cut 0 / L: fixed endpoint, nothing to re-judge
                continue
            for i, (cont, m_off) in enumerate(zip(rec["conts"], rec["match"])):
                key = f"{f['rid']}|{side}|{i}"
                if key not in store.done:
                    todo.append((key, f, side, i, T.decode(resp[:c] + cont, skip_special_tokens=True), m_off))
    log(f"re-judging {len(todo)}")

    async def one(key, f, side, i, text, m_off):
        j = await judge(pool, f["behavior_id"], pr.loc[f["prompt_id"], "user_text"], text, thinking=True)
        store.add({"key": key, "rid": f["rid"], "side": side, "i": i, "off": m_off, "on": j["match"]})

    q = asyncio.Queue()
    for t in todo:
        q.put_nowait(t)

    async def worker():
        while not q.empty():
            await one(*q.get_nowait())

    await asyncio.gather(*[worker() for _ in range(len(pool.urls) * 96)])
    # summary
    by = {}
    for r in store.done.values():
        if r["on"] is None or r["off"] is None:
            continue
        by.setdefault((r["rid"], r["side"]), []).append((r["off"], r["on"]))
    agree = np.mean([a == b for v in by.values() for a, b in v])
    rows = []
    for rid in {k[0] for k in by}:
        if (rid, "lo") in by and (rid, "hi") in by:
            lo, hi = np.array(by[(rid, "lo")]), np.array(by[(rid, "hi")])
            rows.append((hi[:, 0].mean() - lo[:, 0].mean(), hi[:, 1].mean() - lo[:, 1].mean()))
    rows = np.array(rows)
    print(f"sample agreement off vs on: {agree:.3f}")
    print(f"forks with both sides: {len(rows)}; median jump off {np.median(rows[:, 0]):.3f} on {np.median(rows[:, 1]):.3f}; "
          f"corr {np.corrcoef(rows[:, 0], rows[:, 1])[0, 1]:.3f}; jump>=0.5 off {np.sum(rows[:, 0] >= .5)} on {np.sum(rows[:, 1] >= .5)}")


if __name__ == "__main__":
    asyncio.run(main())
