"""Direct-logit readout of the top neurons in each behavior set (CPU, genv; reads weights from safetensors).

For neuron (l, i), its output direction is column i of down_proj; through the final norm (Qwen3.5: x * (1 + w))
and the unembedding this promotes tokens W_U (1 + w) d_i (ignoring the norm's per-token scale). Since the SwiGLU
activation can have either sign, both ends are listed, with z-scores of the forks' deciding tokens. A_last is
the neuron's attribution at the fork position itself, averaged over the behavior's forks (+ = pushes toward
the behavior there; the rest of its attribution comes through earlier positions).

  python neuron_logits.py [K]   (default top 10 per behavior)
"""
import json
import sys

import numpy as np
import torch
from safetensors import safe_open

from attrib import NI
from common import QWEN_SNAP, RES, tok

K = int(sys.argv[1]) if len(sys.argv) > 1 else 10


def main():
    idx = json.loads(open(f"{QWEN_SNAP}/model.safetensors.index.json").read())["weight_map"]

    def get(name):
        with safe_open(f"{QWEN_SNAP}/{idx[name]}", "pt") as f:
            return f.get_tensor(name).float()

    WU = get("lm_head.weight")
    g = 1 + get("model.language_model.norm.weight")
    T = tok()
    sets = json.loads((RES / "ablation_sets.json").read_text())
    F = [json.loads(l) for l in (RES / "phase2_forks.jsonl").read_text().splitlines()]
    cache = {}
    for c, s in sets.items():
        if s["kind"] != "top" or s["k"] != 10:
            continue
        b = s["b"]
        fork_toks = sorted({F[i]["fork_token"] for i in s["forks"]})
        print(f"\n== {b}  (fork tokens: {[T.decode([t]) for t in fork_toks]})")
        neurons = [(int(l), i) for l, v in s["spec"].items() for i in v]
        for l, i in neurons:
            if l not in cache:
                cache = {l: get(f"model.language_model.layers.{l}.mlp.down_proj.weight")}
            d = cache[l][:, i]
            z = WU @ (g * d)
            zs = (z - z.mean()) / z.std()
            up = torch.topk(zs, 8).indices.tolist()
            dn = torch.topk(-zs, 8).indices.tolist()
            al = [float(np.load(RES / "attrib" / f"{f}.npz")["A_last"][l, i]) for f in s["forks"]]
            fz = " ".join(f"{T.decode([t])!r}:{zs[t]:+.1f}" for t in fork_toks)
            print(f"  L{l:2d} n{i:5d}  A_last mean {np.mean(al):+.3f}  fork-token z {fz}\n"
                  f"      + {[T.decode([t]) for t in up]}\n      - {[T.decode([t]) for t in dn]}")


if __name__ == "__main__":
    main()
