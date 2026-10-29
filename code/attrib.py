"""Phase 2a: attribute the fork metric M = logit(V) to MLP neurons (Qwen3.6-27B, HF bf16, genv).

For fork f (prefix x = prompt + y_<=t*), V = sum_v p(v|x) q_v / sum_v p(v|x) over the Phase 1 candidate tokens,
q_v from the thinking-on judge (clamped to [.02, .98]); M = logit(V). A neuron's activation is the down_proj input
a = silu(gate) * up. Attribution = a * dM/da (first-order effect of zeroing it), summed over all positions.

  python attrib.py SHARD NSHARD
    attrib: per fork, results/attrib/<i>.npz with A (sum over positions), A_last (last position only),
            absa (sum_pos |a|, for matched controls), M0, V0
    verify: actual zero-ablation (all positions) of (1) the 16 most positive and 16 most negative neurons singly,
            (2) the top-k positive set for k in KS, and (3) two random sets per k matched on layer and sum|a| rank.
            -> results/attrib/<i>.verify.json
"""
import json
import sys
import time

import numpy as np
import torch

from common import QWEN_SNAP, RES, JsonlStore, log

OUT = RES / "attrib"
KS = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
NL, NI = 64, 17408
QCLAMP = (0.02, 0.98)


def forks():
    return [json.loads(l) for l in (RES / "phase2_forks.jsonl").read_text().splitlines()]


def prefix_ids(f, gen):
    pid, si = f["rid"].split("|")
    return gen[pid]["prompt_ids"] + list(gen[pid]["samples"][int(si)]["ids"])[:f["fork"][0]]


class Model:
    def __init__(self):
        from transformers import AutoModelForCausalLM
        t0 = time.time()
        self.m = AutoModelForCausalLM.from_pretrained(QWEN_SNAP, dtype=torch.bfloat16, device_map="cuda").eval()
        self.m.requires_grad_(False)
        self.downs = [self.m.model.layers[l].mlp.down_proj for l in range(NL)]
        self.acts, self.ablate = {}, {}
        for l, d in enumerate(self.downs):
            d.register_forward_pre_hook(self._pre(l))
        log(f"loaded in {time.time() - t0:.0f}s")

    def _pre(self, l):
        def hook(mod, inp):
            x = inp[0]
            if l in self.ablate:
                x = x.clone()
                x[..., self.ablate[l]] = 0
            if x.requires_grad:
                x.retain_grad()
                self.acts[l] = x
            return (x,)
        return hook

    def metric(self, ids, toks, q, grad=False):
        self.acts = {}
        x = torch.tensor([ids], device="cuda")
        if grad:
            emb = self.m.model.embed_tokens(x).requires_grad_(True)
            logits = self.m(inputs_embeds=emb, logits_to_keep=1).logits[0, -1].float()
        else:
            with torch.no_grad():
                logits = self.m(x, logits_to_keep=1).logits[0, -1].float()
        p = torch.softmax(logits, -1)[toks]
        V = (p * q).sum() / p.sum()
        return torch.log(V / (1 - V)), V, p

    def set_ablation(self, flat):
        self.ablate = {}
        for l, i in zip(*np.divmod(np.asarray(flat, dtype=np.int64), NI)):
            self.ablate.setdefault(int(l), []).append(int(i))
        self.ablate = {l: torch.tensor(v, device="cuda") for l, v in self.ablate.items()}


def attrib_one(M, f, ids):
    toks = torch.tensor([x["tok"] for x in f["next_tokens"]], device="cuda")
    q = torch.tensor([x["q"] for x in f["next_tokens"]], device="cuda").clamp(*QCLAMP)
    M.set_ablation([])
    m, V, p = M.metric(ids, toks, q, grad=True)
    m.backward()
    A = np.zeros((NL, NI), np.float32)
    A_last = np.zeros((NL, NI), np.float32)
    absa = np.zeros((NL, NI), np.float32)
    for l, a in M.acts.items():
        ag = (a.detach().float() * a.grad.float())[0]
        A[l] = ag.sum(0).cpu().numpy()
        A_last[l] = ag[-1].cpu().numpy()
        absa[l] = a.detach().float()[0].abs().sum(0).cpu().numpy()
    M.acts = {}
    return dict(A=A, A_last=A_last, absa=absa, M0=float(m), V0=float(V)), (toks, q)


def matched(sel, absa, rng):
    """Random neurons in the same layer whose sum|a| rank is within +-25 (widened if needed) of each selected neuron's."""
    order = np.argsort(absa, 1)
    rank = np.argsort(order, 1)
    taken, out = set(int(s) for s in sel), []
    for s in sel:
        l, i = divmod(int(s), NI)
        r = rank[l, i]
        w, pool = 25, []
        while not pool:  # widen the window when nearby ranks are used up (large k)
            pool = [l * NI + int(j) for j in order[l, max(0, r - w):r + w + 1] if l * NI + int(j) not in taken]
            w *= 2
        j = pool[rng.integers(len(pool))]
        taken.add(j)
        out.append(j)
    return out


def verify_one(M, f, ids, r, tq):
    toks, q = tq
    rng = np.random.default_rng(0)
    A = r["A"].ravel()
    m0 = r["M0"]

    def run(flat):
        M.set_ablation(flat)
        m, V, p = M.metric(ids, toks, q)
        return float(m), float(V)

    out = {"M0": m0, "V0": r["V0"], "single": [], "sets": []}
    order = np.argsort(-A)
    for flat in list(order[:16]) + list(order[::-1][:16]):
        m, V = run([flat])
        out["single"].append({"n": int(flat), "A": float(A[flat]), "dM": m - m0, "V": V})
    for k in KS:
        top = order[:k]
        m, V = run(top)
        ctrl = [run(matched(top, r["absa"], rng)) for _ in range(2)]
        out["sets"].append({"k": k, "M": m, "V": V, "M_rand": [c[0] for c in ctrl], "V_rand": [c[1] for c in ctrl],
                            "A_sum": float(A[top].sum())})
    M.set_ablation([])
    return out


def main(shard, nshard):
    OUT.mkdir(exist_ok=True)
    gen = JsonlStore(RES / "stage_a_gen.jsonl").done
    F = forks()
    todo = [(i, f) for i, f in enumerate(F) if i % nshard == shard and not (OUT / f"{i}.verify.json").exists()]
    if not todo:
        return
    M = Model()
    for i, f in todo:
        ids = prefix_ids(f, gen)
        t0 = time.time()
        r, tq = attrib_one(M, f, ids)
        np.savez_compressed(OUT / f"{i}.npz", **r, key=f["key"])
        t1 = time.time()
        v = verify_one(M, f, ids, r, tq)
        v["key"] = f["key"]
        (OUT / f"{i}.verify.json").write_text(json.dumps(v))
        top = v["sets"]
        log(f"fork {i} {f['behavior_id']} len {len(ids)}: M0 {r['M0']:.2f} V0 {r['V0']:.3f}; attrib {t1 - t0:.0f}s "
            f"verify {time.time() - t1:.0f}s; V after top-k " + " ".join(f"{s['k']}:{s['V']:.2f}/{np.mean(s['V_rand']):.2f}" for s in top))
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
