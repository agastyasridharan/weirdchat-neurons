"""Re-score Phase 2 candidate forks with the thinking-on (production) judge.

Candidates are forks whose fresh thinking-off jump P(t*+1) - P(t*) is >= MIN_JUMP. For each one, the existing
continuations are re-judged with thinking on (no new generation):
  - the CONFIRM samples on both sides of the fork (24 + 24),
  - every candidate next token's q samples (8-16 each),
  - the unchanged full response, K_FULL times, for P_on(L).
P_on(0) is the prompt's Stage A thinking-on rate (16 samples). Results already in verify_on.jsonl are reused.

  python rescore_on.py judge     # judge everything still missing
  python rescore_on.py summary   # per-fork thinking-on values -> forks_on_rescored.jsonl
"""
import asyncio
import sys

import numpy as np

from common import RES, JsonlStore, Pool, judge, load_prompts, log, tok

MIN_JUMP = 0.25
K_FULL = 8


def fresh_off(f, pl_off):
    lo, hi = f["fork"][:2]
    Plo, _, Phi, _ = f["fork_confirm"]
    if hi >= f["L"]:
        Phi = pl_off.get(f["rid"], Phi)
    return Phi - Plo


def load():
    forks = JsonlStore(RES / "forks_off.jsonl").done
    pl_off = {}
    for r in JsonlStore(RES / "full_off.jsonl").done.values():
        if r["match"] is not None:
            pl_off.setdefault(r["rid"], []).append(r["match"])
    pl_off = {k: float(np.mean(v)) for k, v in pl_off.items()}
    cands = sorted((f for f in forks.values() if fresh_off(f, pl_off) >= MIN_JUMP), key=lambda f: f["rid"])
    return cands, pl_off


def jobs(f, samples, resp):
    """(key, kind, text, off_match) for every continuation of this fork; key matches the verify_on.jsonl scheme."""
    T = tok()
    lo, hi = f["fork"][:2]
    out = []
    for side, c in (("lo", lo), ("hi", hi)):
        rec = samples.get(f"{f['rid']}|{c}|confirm")
        if rec is None:  # cut 0 or L: endpoint, measured separately
            continue
        for i, (cont, m) in enumerate(zip(rec["conts"], rec["match"])):
            out.append((f"{f['rid']}|{side}|{i}", side, resp[:c] + cont, m))
    for nt in f["next_tokens"]:
        if nt["tok"] == resp[lo] and hi == lo + 1:
            continue  # its q is the hi-side confirm
        for b in (0, 1):
            rec = samples.get(f"{f['rid']}|{lo}+{nt['tok']}|{b}")
            if rec is None:
                continue
            for i, (cont, m) in enumerate(zip(rec["conts"], rec["match"])):
                out.append((f"{f['rid']}|q{nt['tok']}|{b}|{i}", f"q{nt['tok']}", resp[:lo] + [nt["tok"]] + cont, m))
    return [(k, kind, T.decode(ids, skip_special_tokens=True), m) for k, kind, ids, m in out]


async def run_judge():
    cands, _ = load()
    samples = JsonlStore(RES / "forks_samples_off.jsonl").done
    gen = JsonlStore(RES / "stage_a_gen.jsonl").done
    pr = load_prompts().set_index("prompt_id")
    prev = JsonlStore(RES / "verify_on.jsonl").done
    store = JsonlStore(RES / "rescore_on.jsonl")
    for k, r in prev.items():  # reuse the earlier 80-fork check
        if k not in store.done and r["on"] is not None:
            store.add({"key": k, "rid": r["rid"], "kind": r["side"], "off": r["off"], "on": r["on"]})
    todo = []
    for f in cands:
        pid, si = f["rid"].split("|")
        s = gen[pid]["samples"][int(si)]
        for k, kind, text, m in jobs(f, samples, list(s["ids"])):
            if k not in store.done:
                todo.append((k, f, kind, text, m))
        for i in range(K_FULL):
            k = f"{f['rid']}|full|{i}"
            if k not in store.done:
                todo.append((k, f, "full", s["text"], None))
    log(f"{len(cands)} candidate forks; judging {len(todo)} (thinking on)")
    pool = Pool("gemma", 96)

    async def one(k, f, kind, text, m):
        j = await judge(pool, f["behavior_id"], pr.loc[f["prompt_id"], "user_text"], text, thinking=True)
        store.add({"key": k, "rid": f["rid"], "kind": kind, "off": m, "on": j["match"]})

    q = asyncio.Queue()
    for t in todo:
        q.put_nowait(t)

    async def worker():
        while not q.empty():
            await one(*q.get_nowait())

    await asyncio.gather(*[worker() for _ in range(len(pool.urls) * 96)])
    log("done")


def summary():
    cands, pl_off = load()
    store = JsonlStore(RES / "rescore_on.jsonl").done
    ja_on = JsonlStore(RES / "stage_a_judge_on.jsonl").done
    base_on = {}
    for r in ja_on.values():
        if r["match"] is not None:
            base_on.setdefault(r["prompt_id"], []).append(r["match"])
    by = {}
    for r in store.values():
        if r["on"] is not None:
            by.setdefault((r["rid"], r["kind"]), []).append((r["off"], r["on"]))
    rows, missing = [], 0
    for f in cands:
        rid, (lo, hi) = f["rid"], f["fork"][:2]

        def side(kind, c):
            if c == 0:
                v = base_on[f["prompt_id"]]
                return float(np.mean(v)), len(v)
            if c >= f["L"]:
                v = [b for _, b in by.get((rid, "full"), [])]
                return (float(np.mean(v)), len(v)) if v else (None, 0)
            v = [b for _, b in by.get((rid, kind), [])]
            return (float(np.mean(v)), len(v)) if v else (None, 0)

        (Plo, nlo), (Phi, nhi) = side("lo", lo), side("hi", hi)
        full = [b for _, b in by.get((rid, "full"), [])]
        if Plo is None or Phi is None:
            missing += 1
            continue
        nts = []
        for nt in f["next_tokens"]:
            if nt["tok"] == f["fork_token"] and hi == lo + 1:
                q, n = Phi, nhi
            else:
                v = [b for _, b in by.get((rid, f"q{nt['tok']}"), [])]
                q, n = (float(np.mean(v)), len(v)) if v else (None, 0)
            nts.append({**nt, "q_off": nt["q"], "q": q, "n": n})
        ok = [nt for nt in nts if nt["q"] is not None]
        V = sum(nt["p"] * nt["q"] for nt in ok) / max(sum(nt["p"] for nt in ok), 1e-9)
        rows.append({"key": rid, "rid": rid, "prompt_id": f["prompt_id"], "behavior_id": f["behavior_id"],
                     "L": f["L"], "fork": [lo, hi], "fork_token": f["fork_token"], "fork_token_str": f["fork_token_str"],
                     "jump_off": fresh_off(f, pl_off), "Plo_on": Plo, "Phi_on": Phi, "n_lo": nlo, "n_hi": nhi,
                     "jump_on": Phi - Plo, "PL_on": float(np.mean(full)) if full else None, "V_on": V,
                     "next_tokens": nts})
    out = RES / "forks_on_rescored.jsonl"
    out.write_text("".join(__import__("json").dumps(r) + "\n" for r in rows))
    j_off = np.array([r["jump_off"] for r in rows])
    j_on = np.array([r["jump_on"] for r in rows])
    print(f"{len(rows)} forks rescored ({missing} incomplete) -> {out.name}")
    print(f"jump off median {np.median(j_off):.3f} mean {j_off.mean():.3f}; on median {np.median(j_on):.3f} mean {j_on.mean():.3f}; corr {np.corrcoef(j_off, j_on)[0, 1]:.3f}")
    for t in (0.1, 0.25, 0.5):
        print(f"  jump_on >= {t}: {np.sum(j_on >= t)}")
    pl = np.array([r["PL_on"] for r in rows if r["PL_on"] is not None])
    print(f"P_on(L): median {np.median(pl):.2f}; <0.5: {np.sum(pl < .5)}")
    V = np.array([r["V_on"] for r in rows])
    P = np.array([r["Plo_on"] for r in rows])
    print(f"V_on vs P_on(t*): MAD {np.mean(np.abs(V - P)):.3f}, bias {np.mean(V - P):+.3f}, r {np.corrcoef(V, P)[0, 1]:.3f}")
    beh = {}
    for r in rows:
        beh.setdefault(r["behavior_id"], []).append(r["jump_on"])
    for b, v in sorted(beh.items()):
        v = np.array(v)
        print(f"  {b:30s} n {len(v):3d}  on-jump median {np.median(v):.2f}  >=0.5 {np.sum(v >= .5):2d}  >=0.25 {np.sum(v >= .25):2d}")


if __name__ == "__main__":
    if sys.argv[1] == "judge":
        asyncio.run(run_judge())
    else:
        summary()
