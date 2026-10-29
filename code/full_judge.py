"""P(L): re-judge each fork rollout's full response (unchanged text) K times with the thinking-off judge.

The fork search fixed P(L) = 1 because Stage A judged the rollout positive once. The judge samples at
T=1, so a single positive can be noise; rollouts with low P(L) are borderline/false positives, and a
fork found at the last token of such a rollout is an artifact of that one label.

  python full_judge.py [K]
"""
import asyncio
import sys

import numpy as np

from common import RES, JsonlStore, Pool, judge, load_prompts, log

K = int(sys.argv[1]) if len(sys.argv) > 1 else 16


async def main():
    forks = JsonlStore(RES / "forks_off.jsonl").done
    gen = JsonlStore(RES / "stage_a_gen.jsonl").done
    pr = load_prompts().set_index("prompt_id")
    store = JsonlStore(RES / "full_off.jsonl")
    pool = Pool("gemma", 96)
    todo = [(rid, i) for rid in sorted(forks) for i in range(K) if f"{rid}|{i}" not in store.done]
    log(f"judging {len(todo)}")

    async def one(rid, i):
        pid, si = rid.split("|")
        j = await judge(pool, gen[pid]["behavior_id"], pr.loc[pid, "user_text"], gen[pid]["samples"][int(si)]["text"], False)
        store.add({"key": f"{rid}|{i}", "rid": rid, "i": i, "match": j["match"]})

    q = asyncio.Queue()
    for t in todo:
        q.put_nowait(t)

    async def worker():
        while not q.empty():
            await one(*q.get_nowait())

    await asyncio.gather(*[worker() for _ in range(len(pool.urls) * 96)])
    by = {}
    for r in store.done.values():
        if r["match"] is not None:
            by.setdefault(r["rid"], []).append(r["match"])
    pl = np.array([np.mean(v) for v in by.values()])
    log(f"P(L) over {len(pl)} rollouts: median {np.median(pl):.2f}; <0.5: {np.sum(pl < .5)}; <0.25: {np.sum(pl < .25)}")


if __name__ == "__main__":
    asyncio.run(main())
