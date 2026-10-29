"""Check that remote (Runpod) Gemma judge servers are interchangeable with the local ones.

Same weights revision, vLLM version, flags and GPU type (H200); only the host differs. A random
subset of the judge-validation transcripts is re-judged (thinking off) once on the remote pool and
once on the local pool; both are compared with the stored thinking-off labels from judge_val and
with each other. Interchangeable servers give the same agreement and the same positive rate up to
sampling noise (the judge samples at T=1).

  WCN_SERVERS=servers_parity.json python judge_parity.py [N]   # roles: gemma (local), gemma_remote
"""
import asyncio
import random
import sys

import numpy as np

from common import RES, JsonlStore, Pool, judge, load_transcripts, log

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300


async def main():
    val = [r for r in JsonlStore(RES / "judge_val.jsonl").done.values() if r["cond"] == "off" and r["match"] is not None]
    random.Random(0).shuffle(val)
    val = val[:N]
    t = load_transcripts().set_index("transcript_id")
    store = JsonlStore(RES / "judge_parity.jsonl")
    pools = {"local": Pool("gemma", 64), "remote": Pool("gemma_remote", 64)}

    async def one(r, side):
        key = f"{r['transcript_id']}|{side}"
        if key in store.done:
            return
        msgs = t.loc[r["transcript_id"], "messages"]
        user = [m for m in msgs if m["role"] == "user"][0]["content"]
        resp = [m for m in msgs if m["role"] == "assistant"][0]["content"]
        j = await judge(pools[side], r["behavior_id"], user, resp, thinking=False)
        store.add({"key": key, "transcript_id": r["transcript_id"], "side": side, "orig": r["match"], "match": j["match"]})

    await asyncio.gather(*[one(r, s) for r in val for s in pools])
    d = {}
    for r in store.done.values():
        d.setdefault(r["transcript_id"], {})["orig"] = r["orig"]
        d[r["transcript_id"]][r["side"]] = r["match"]
    rows = [x for x in d.values() if x.get("local") is not None and x.get("remote") is not None]
    a = lambda k1, k2: np.mean([x[k1] == x[k2] for x in rows])
    log(f"n={len(rows)}  agree(local,orig) {a('local', 'orig'):.3f}  agree(remote,orig) {a('remote', 'orig'):.3f}  "
        f"agree(local,remote) {a('local', 'remote'):.3f}  pos rate orig {np.mean([x['orig'] for x in rows]):.3f} "
        f"local {np.mean([x['local'] for x in rows]):.3f} remote {np.mean([x['remote'] for x in rows]):.3f}")


if __name__ == "__main__":
    asyncio.run(main())
