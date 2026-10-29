"""Phase 2b inputs: behavior-level neuron sets for rate ablation (genv; reads results/attrib/).

For behavior b with Phase 2 forks F_b, score(n) = sum_{f in F_b} A_f(n) / s_f, where A_f is the summed
activation x gradient attribution of fork f and s_f the sum of its 100 largest positive attributions (so every
fork counts equally). Set top-k = the k highest-scoring neurons. Controls: random neurons matched one-for-one
on layer and on the rank of mean sum|a| across F_b (attrib.matched).

Also writes the reference texts for the damage check: Stage A replies (judged negative) to prompts of the
behaviors NOT in the rate experiment, truncated to 256 response tokens.

  python sets.py   -> results/ablation_sets.json, results/ref_texts.json
"""
import json

import numpy as np

from attrib import NI, NL, matched
from common import RES, JsonlStore, load_prompts

B5 = ["claims-called-911", "denying-ai-identity", "fabricated-code-execution", "false-physical-embodiment",
      "laser-at-aircraft"]
KS_TOP, KS_RAND = [10, 100, 1000], [100, 1000]


def spec(flat):
    d = {}
    for l, i in zip(*np.divmod(np.asarray(flat, dtype=np.int64), NI)):
        d.setdefault(str(int(l)), []).append(int(i))
    return d


def main():
    F = [json.loads(l) for l in (RES / "phase2_forks.jsonl").read_text().splitlines()]
    sets = {}
    for b in B5:
        ix = [i for i, f in enumerate(F) if f["behavior_id"] == b]
        score, absa = np.zeros(NL * NI), np.zeros((NL, NI))
        for i in ix:
            z = np.load(RES / "attrib" / f"{i}.npz")
            A = z["A"].ravel().astype(np.float64)
            score += A / np.sort(A)[::-1][:100].sum()
            absa += z["absa"] / len(ix)
        order = np.argsort(-score)
        rng = np.random.default_rng(1)
        for k in KS_TOP:
            sets[f"{b}|top{k}"] = {"b": b, "k": k, "kind": "top", "spec": spec(order[:k]), "forks": ix}
        for k in KS_RAND:
            sets[f"{b}|rand{k}"] = {"b": b, "k": k, "kind": "rand", "spec": spec(matched(order[:k], absa, rng)), "forks": ix}
        top = order[:10]
        print(f"{b:28s} forks {len(ix):2d}  top-10 (layer, neuron, score): "
              + ", ".join(f"({t // NI},{t % NI},{score[t]:.2f})" for t in top))
    (RES / "ablation_sets.json").write_text(json.dumps(sets))

    gen = JsonlStore(RES / "stage_a_gen.jsonl").done
    J = JsonlStore(RES / "stage_a_judge_on.jsonl").done
    pr = load_prompts().set_index("prompt_id")
    refs = []
    for key, j in sorted(J.items()):
        if j["match"] is False and pr.loc[j["prompt_id"], "behavior_id"] not in B5:
            g = gen[j["prompt_id"]]
            resp = list(g["samples"][j["i"]]["ids"])[:256]
            refs.append({"key": key, "prompt_len": len(g["prompt_ids"]), "ids": g["prompt_ids"] + resp})
    refs = [refs[i] for i in np.random.default_rng(2).choice(len(refs), 200, replace=False)]
    (RES / "ref_texts.json").write_text(json.dumps(refs))
    print(f"{len(sets)} ablation sets; {len(refs)} reference texts")


if __name__ == "__main__":
    main()
