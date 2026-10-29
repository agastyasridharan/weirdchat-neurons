"""Stage B: locate fork points in behavior rollouts by prefix resampling.

For a behavior rollout r (prompt x, response tokens y_1..y_L), P(c) = P(behavior | x, y_<=c), estimated
by sampling continuations of the prefix (same sampling as WeirdChat, total response capped at 1024)
and judging prefix+continuation with the WeirdChat judge. P(0) is the Stage A base rate; P(L) = 1.

Per rollout, coarse-to-fine:
  1. sentence-level cuts: coarse grid of <= 6 boundaries, then repeatedly refine inside the interval
     with the largest increase until it is a single segment (the fork segment);
  2. token-level bisection inside the fork segment for the first token cut whose P crosses the
     midpoint of the segment's endpoints -> fork position t* (decisive token y_{t*+1});
  3. next-token values at t*: for each likely next token v (sampling mass >= 5%, at most 4, plus the
     actual token), q_v = P(behavior | x, y_<=t*, v). The fork's next-token value is sum_v p(v) q_v.
Sampling is sequential everywhere: 8 continuations, plus 8 more unless the first 8 agree.
The search picks the largest measured jump, so its own estimates are biased (winner's curse); the fork
is therefore re-measured with CONFIRM fresh continuations on each side, and those unbiased values are
what the jump and the next-token decomposition are evaluated against.
Rollouts run in rounds (one per prompt, then a second) so the most diverse set finishes first.

  python forks.py [--off] [--max-rollouts N] [--split discovery|heldout] [--behavior ID] [--out TAG] [--per-prompt K]
  (defaults: discovery split, all behaviors, output tag on/off; Phase 2d ran --split heldout --behavior denying-ai-identity --out on_dai_held --per-prompt 3)
"""
import asyncio
import json
import math
import sys

import numpy as np

from common import (MAX_RESP, RES, JsonlStore, Pool, generate, judge, load_prompts, log,
                    next_token_logprobs, tok)

R_PER_PROMPT = 2
MIN_BASE_K = 2           # drop rule: >= 2/64 (~3%) behavior samples in bf16
COARSE = 6
MIN_SEG = 6
EOS = {248046, 248044}
SEED = 0
CONCURRENT_ROLLOUTS = 256
CONFIRM = 24


def boundaries(ids: list[int]) -> list[int]:
    """Cut positions (prefix lengths) at sentence/line ends, segments >= MIN_SEG tokens; includes L."""
    T = tok()
    pieces = [T.decode([t]) for t in ids]
    cuts = []
    for t, s in enumerate(pieces):
        nxt = pieces[t + 1] if t + 1 < len(pieces) else ""
        end = "\n" in s or (s.rstrip(" *\"')") != s.rstrip() or s.rstrip()[-1:] in ".!?:;") and \
            s.rstrip()[-1:] in ".!?:;*\"')" and (nxt[:1].isspace() or nxt == "")
        if end and t + 1 - (cuts[-1] if cuts else 0) >= MIN_SEG:
            cuts.append(t + 1)
    if not cuts or cuts[-1] != len(ids):
        cuts.append(len(ids))
    return cuts


class Rollout:
    def __init__(self, rid, d, sample, base_p, user_text, qpool, jpool, samples_store, thinking):
        self.rid, self.d, self.user, self.base_p = rid, d, user_text, base_p
        ids = list(sample["ids"])
        while ids and ids[-1] in EOS:
            ids.pop()
        self.resp = ids
        self.L = len(ids)
        self.q, self.j, self.ss, self.thinking = qpool, jpool, samples_store, thinking
        self.obs = {}  # key -> [k_pos, n]

    async def _batch(self, key, prefix_resp, n, batch_idx):
        sk = f"{self.rid}|{key}|{batch_idx}"
        if sk in self.ss.done:
            rec = self.ss.done[sk]
        else:
            ids = self.d["prompt_ids"] + prefix_resp
            outs = await generate(self.q, ids, n=n, max_tokens=max(MAX_RESP - len(prefix_resp), 1))
            texts = [tok().decode(prefix_resp + list(o["ids"]), skip_special_tokens=True) for o in outs]
            js = await asyncio.gather(*[judge(self.j, self.d["behavior_id"], self.user, t, self.thinking)
                                        for t in texts])
            rec = {"key": sk, "rid": self.rid, "cut": key, "batch": batch_idx,
                   "conts": [list(o["ids"]) for o in outs], "finish": [o["finish"] for o in outs],
                   "match": [x["match"] for x in js]}
            self.ss.add(rec)
        return [m for m in rec["match"] if m is not None]

    async def est(self, key, prefix_resp, k0=8, k1=8, adaptive=True):
        if key in self.obs:
            return self.obs[key][0] / max(self.obs[key][1], 1)
        m = await self._batch(key, prefix_resp, k0, 0)
        if not adaptive or 0 < sum(m) < len(m):
            m += await self._batch(key, prefix_resp, k1, 1)
        self.obs[key] = [sum(m), len(m)]
        return sum(m) / max(len(m), 1)

    async def P(self, c):
        if c == 0:
            return self.base_p
        if c >= self.L:
            return 1.0
        return await self.est(str(c), self.resp[:c])

    async def run(self):
        cuts = [0] + boundaries(self.resp)
        known = {0: self.base_p, self.L: 1.0}
        lo_i, hi_i = 0, len(cuts) - 1
        # coarse-to-fine over sentence boundaries
        while hi_i - lo_i > 1:
            inner = list(range(lo_i + 1, hi_i))
            pick = sorted({inner[int(round(x))] for x in np.linspace(0, len(inner) - 1, min(COARSE, len(inner)))})
            vals = await asyncio.gather(*[self.P(cuts[i]) for i in pick])
            for i, v in zip(pick, vals):
                known[cuts[i]] = v
            pts = [lo_i] + pick + [hi_i]
            jumps = [(known[cuts[b]] - known[cuts[a]], a, b) for a, b in zip(pts, pts[1:])]
            _, lo_i, hi_i = max(jumps)
        a, b = cuts[lo_i], cuts[hi_i]
        Pa, Pb = known[a], known[b]
        # token-level bisection inside the fork segment
        thr = (Pa + Pb) / 2
        lo, hi = a, b
        while hi - lo > 1:
            mid = (lo + hi) // 2
            v = await self.est(str(mid), self.resp[:mid])
            known[mid] = v
            if v >= thr:
                hi = mid
            else:
                lo = mid
        known[lo] = await self.P(lo) if lo not in known else known[lo]
        known[hi] = await self.P(hi) if hi not in known else known[hi]
        # fresh, unselected re-estimates on both sides of the fork
        async def fresh(c):
            if c == 0:
                return self.base_p, None
            if c >= self.L:
                return 1.0, None
            m = await self._batch(str(c), self.resp[:c], CONFIRM, "confirm")
            return sum(m) / max(len(m), 1), len(m)
        (Plo_c, nlo_c), (Phi_c, nhi_c) = await asyncio.gather(fresh(lo), fresh(hi))
        # next-token values at the fork
        lp = await next_token_logprobs(self.q, self.d["prompt_ids"] + self.resp[:lo], k=20)
        top = lp["logprobs"]["top_logprobs"][0]
        raw = sorted(((int(k.split(":")[1]), v) for k, v in top.items()), key=lambda x: -x[1])
        probs = np.exp([v for _, v in raw])
        keep = np.searchsorted(np.cumsum(probs / probs.sum()), 0.95) + 1  # top_k=20 then top_p=0.95
        nucleus = [(t, float(p)) for (t, _), p in zip(raw[:keep], probs[:keep] / probs[:keep].sum())]
        actual = self.resp[lo]
        cand = [(t, p) for t, p in nucleus if p >= 0.05][:4]
        if actual not in [t for t, _ in cand]:
            cand.append((actual, dict(nucleus).get(actual, 0.0)))
        async def qv(t):
            if t == actual and hi == lo + 1:
                return Phi_c, nhi_c
            v = await self.est(f"{lo}+{t}", self.resp[:lo] + [t])
            return v, self.obs[f"{lo}+{t}"][1]
        qs = await asyncio.gather(*[qv(t) for t, _ in cand])
        T = tok()
        rec = {
            "key": self.rid, "rid": self.rid, "prompt_id": self.d["prompt_id"], "behavior_id": self.d["behavior_id"],
            "L": self.L, "base_p": self.base_p, "curve": sorted((c, known[c]) for c in known),
            "n_obs": {k: v[1] for k, v in self.obs.items()},
            "seg": [a, b, Pa, Pb], "fork": [lo, hi, known[lo], known[hi]],
            "fork_confirm": [Plo_c, nlo_c, Phi_c, nhi_c],
            "fork_token": actual, "fork_token_str": T.decode([actual]),
            "prefix_tail": T.decode(self.resp[max(0, lo - 40):lo]), "seg_text": T.decode(self.resp[a:b]),
            "next_tokens": [{"tok": t, "str": T.decode([t]), "p": p, "q": q, "n": n}
                            for (t, p), (q, n) in zip(cand, qs)],
            "nucleus_mass_covered": sum(p for t, p in cand),
        }
        return rec


async def main():
    thinking = "--off" not in sys.argv
    maxr = int(sys.argv[sys.argv.index("--max-rollouts") + 1]) if "--max-rollouts" in sys.argv else 10 ** 9
    tag = "on" if thinking else "off"
    arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    split, beh, otag = arg("--split", "discovery"), arg("--behavior", None), arg("--out", tag)
    pr = load_prompts().set_index("prompt_id")
    gen = JsonlStore(RES / "stage_a_gen.jsonl").done
    ja = JsonlStore(RES / f"stage_a_judge_{tag}.jsonl").done
    by_prompt = {}
    for k, v in ja.items():
        by_prompt.setdefault(v["prompt_id"], {})[v["i"]] = v["match"]
    rng = np.random.default_rng(SEED)
    rpp = int(arg("--per-prompt", R_PER_PROMPT))
    rounds = [[] for _ in range(rpp)]
    for pid, d in gen.items():
        if d["split"] != split or pid not in by_prompt or (beh and d["behavior_id"] != beh):
            continue
        m = by_prompt[pid]
        if not thinking and len(m) < len(d["samples"]):  # base rate not fully judged yet (thinking-on Stage A judged 16)
            continue
        pos = sorted(i for i, x in m.items() if x)
        valid = [x for x in m.values() if x is not None]
        if len(pos) < MIN_BASE_K:
            continue
        for rnd, i in enumerate(rng.choice(pos, size=min(rpp, len(pos)), replace=False)):
            rounds[rnd].append((pid, int(i), sum(valid) / len(valid)))
    todo = [t for rnd in rounds for t in sorted(rnd)][:maxr]
    log(f"{len(todo)} rollouts from {len({t[0] for t in todo})} prompts")
    qpool, jpool = Pool("qwen", 64), Pool("gemma", 256)
    ss = JsonlStore(RES / f"forks_samples_{otag}.jsonl")
    out = JsonlStore(RES / f"forks_{otag}.jsonl")

    async def one(pid, i, bp):
        rid = f"{pid}|{i}"
        if rid in out.done:
            return
        r = Rollout(rid, gen[pid], gen[pid]["samples"][i], bp, pr.loc[pid, "user_text"], qpool, jpool, ss, thinking)
        try:
            out.add(await r.run())
        except Exception as e:  # keep going; record failure
            log(f"FAIL {rid}: {type(e).__name__}: {e}")
            return
        log(f"fork {len(out.done)}/{len(todo)} {gen[pid]['behavior_id']} jump {out.done[rid]['fork'][2]:.2f}->{out.done[rid]['fork'][3]:.2f} "
            f"tok={out.done[rid]['fork_token_str']!r}")

    q = asyncio.Queue()
    for t in todo:
        q.put_nowait(t)

    async def worker():
        while not q.empty():
            await one(*q.get_nowait())

    await asyncio.gather(*[worker() for _ in range(CONCURRENT_ROLLOUTS)])


if __name__ == "__main__":
    asyncio.run(main())
