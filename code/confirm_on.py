"""Fresh thinking-on confirmation of the Phase 2 fork set.

The Phase 2 set (rescore_on.py: thinking-on rise >= 0.5, P_on(L) >= 0.5, fork before the last token) was chosen
on the same continuations that measure its rise, so those rises are biased upward. Here each fork gets NEW
continuations on both sides (y_<=t* and y_<=t*+1; for t* = 0, the empty prefix), judged with thinking on.

  python confirm_on.py judge     # generate + judge (resumable)
  python confirm_on.py summary   # -> results/phase2_forks.jsonl
"""
import asyncio
import json
import sys

import numpy as np

from common import RES, JsonlStore, Pool, load_prompts, log
from forks import Rollout

N = 24


def phase2_set():
    rows = [json.loads(l) for l in (RES / "forks_on_rescored.jsonl").read_text().splitlines()]
    return [r for r in rows if r["jump_on"] >= .5 and r["PL_on"] >= .5 and r["fork"][1] < r["L"]]


async def run_judge():
    sel = phase2_set()
    gen = JsonlStore(RES / "stage_a_gen.jsonl").done
    pr = load_prompts().set_index("prompt_id")
    ss = JsonlStore(RES / "confirm_on_samples.jsonl")
    qpool, jpool = Pool("qwen", 256), Pool("gemma", 96)
    log(f"{len(sel)} Phase 2 forks; {N} fresh continuations per side")

    async def one(r):
        pid, si = r["rid"].split("|")
        ro = Rollout(r["rid"], gen[pid], gen[pid]["samples"][int(si)], None, pr.loc[pid, "user_text"],
                     qpool, jpool, ss, thinking=True)
        await asyncio.gather(*[ro._batch(str(c), ro.resp[:c], N, "fresh_on") for c in r["fork"]])

    await asyncio.gather(*[one(r) for r in sel])
    log("done")


def summary():
    sel = phase2_set()
    ss = JsonlStore(RES / "confirm_on_samples.jsonl").done
    out = []
    for r in sel:
        m = [[x for x in ss[f"{r['rid']}|{c}|fresh_on"]["match"] if x is not None] for c in r["fork"]]
        Plo, Phi = (float(np.mean(v)) for v in m)
        out.append({**r, "Plo_fresh": Plo, "Phi_fresh": Phi, "n_fresh": [len(v) for v in m], "jump_fresh": Phi - Plo})
    (RES / "phase2_forks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    sel_j = np.array([r["jump_on"] for r in out])
    fr = np.array([r["jump_fresh"] for r in out])
    print(f"{len(out)} forks -> phase2_forks.jsonl")
    print(f"rise on selection samples: mean {sel_j.mean():.3f} median {np.median(sel_j):.3f}")
    print(f"rise on fresh samples:     mean {fr.mean():.3f} median {np.median(fr):.3f}; corr {np.corrcoef(sel_j, fr)[0, 1]:.3f}")
    for t in (0.25, 0.5):
        print(f"  fresh rise >= {t}: {np.sum(fr >= t)}")
    by = {}
    for r in out:
        by.setdefault(r["behavior_id"], []).append(r["jump_fresh"])
    for b, v in sorted(by.items()):
        print(f"  {b:30s} n {len(v):2d}  fresh median {np.median(v):.2f}  >=0.5 {sum(x >= .5 for x in v):2d}  >=0.25 {sum(x >= .25 for x in v):2d}")


if __name__ == "__main__":
    if sys.argv[1] == "judge":
        asyncio.run(run_judge())
    else:
        summary()
