"""Phase 2d: denying-ai-identity forks on HELD-OUT prompts (forks.py --split heldout --out on_dai_held), genv.

For each held-out fork with a fresh thinking-on rise >= MIN_RISE before the last token: attribution of M to all
neurons and heads (as attrib.py / upstream.py), and V when the discovery-derived denying-ai-identity sets are zeroed
(top100, its late and early parts, mid100, med100, head10, and their random controls), against the fork's own
top-100 neurons and a matched random set.
  -> results/heldfork_dai.json (per fork: token, rise, V under each set, overlap of own top-100 with the sets)

  python heldfork.py
"""
import json

import numpy as np
import torch

from attrib import NI, QCLAMP, attrib_one, matched
from common import RES, JsonlStore, log
from upstream import NH, UModel

MIN_RISE = 0.3
B = "denying-ai-identity"


def main():
    gen = JsonlStore(RES / "stage_a_gen.jsonl").done
    R = [json.loads(l) for l in (RES / "forks_on_dai_held.jsonl").read_text().splitlines()]
    sel = [r for r in R if r["fork_confirm"][2] - r["fork_confirm"][0] >= MIN_RISE and r["fork"][0] < r["L"] - 1]
    old, up = json.loads((RES / "ablation_sets.json").read_text()), json.loads((RES / "upstream_sets.json").read_text())
    SETS = {k: old[f"{B}|{k}"]["spec"] for k in ["top100", "rand100", "top1000"]} | \
           {k: up[f"{B}|{k}"]["spec"] for k in ["late", "early", "mid100", "med100", "randmed100"]}
    HEADS = {k: up[f"{B}|{k}"]["heads"] for k in ["head10", "randhead10"]}
    flat = lambda s: [int(l) * NI + i for l, v in s.items() for i in v]
    M = UModel()
    rng, out = np.random.default_rng(0), []
    for r in sel:
        pid, si = r["rid"].split("|")
        ids = gen[pid]["prompt_ids"] + list(gen[pid]["samples"][int(si)]["ids"])[:r["fork"][0]]
        f = {"next_tokens": [x for x in r["next_tokens"]]}
        M.hacts, M.hzero = {}, None
        a, (toks, q) = attrib_one(M, f, ids)
        H, _ = M.heads(range(64))
        M.hacts = {}

        def V(neurons=(), heads=None):
            M.set_ablation(list(neurons))
            M.hzero = {int(l): v for l, v in heads.items()} if heads else None
            v = float(M.metric(ids, toks, q)[1])
            M.set_ablation([])
            M.hzero = None
            return v
        own = np.argsort(-a["A"].ravel())[:100]
        Hf = np.nan_to_num(H, nan=-np.inf).ravel()
        oh = np.argsort(-Hf)[:10]
        rec = {"rid": r["rid"], "tok": r["fork_token_str"], "pos": r["fork"][0], "L": r["L"],
               "rise": [r["fork_confirm"][0], r["fork_confirm"][2]], "V0": a["V0"],
               "V": {k: V(flat(s)) for k, s in SETS.items()} | {k: V(heads=h) for k, h in HEADS.items()}
               | {"own100": V(own), "own_rand100": V(matched(own, a["absa"], rng))},
               "overlap": {k: len(set(own.tolist()) & set(flat(s))) for k, s in SETS.items()},
               "own_heads": [[int(h) // NH, int(h) % NH] for h in oh],
               "own_layers": np.bincount(own // NI // 8, minlength=8).tolist()}
        hs = {(int(l), h) for l, v in HEADS["head10"].items() for h in v}
        rec["head_overlap"] = len({(l, h) for l, h in rec["own_heads"]} & hs)
        out.append(rec)
        log(f"{r['fork_token_str']!r} rise {rec['rise']}: V0 {a['V0']:.3f}; " + " ".join(f"{k} {v:.3f}" for k, v in rec["V"].items())
            + f"; own-top100 overlap with top100 {rec['overlap']['top100']}, med100 {rec['overlap']['med100']}; head overlap {rec['head_overlap']}")
        torch.cuda.empty_cache()
    (RES / "heldfork_dai.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
