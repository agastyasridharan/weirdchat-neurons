"""Phase 2d: one step upstream of the late writer neurons (Qwen3.6-27B, HF bf16, genv).

Per Phase 2 fork, two forward/backward passes:
  1. M = logit(V) as in attrib.py. Attribution of M to every attention head: the head's slice of the input to
     o_proj (full attention, 24 heads x 256) or out_proj (Gated DeltaNet, 48 heads x 128), a * dM/da summed over
     the slice and all positions. (Neuron attribution to M is already in results/attrib/<i>.npz.)
  2. T = the late writers' linearized contribution to M: sum over the fork's behavior's top-100 neurons in layers
     >= LATE of g * a, with g = dM/da from pass 1 held fixed. Attribution of T to every neuron and head below LATE
     is the first-order change in what the writers contribute when that unit is zeroed: how much it acts THROUGH
     the writers (any path).
  -> results/upstream/<i>.npz: H, HT (64 x 48 head attribution to M and to T; NaN pads the 24-head layers),
     AT (LATE x 17408 neuron attribution to T, zero above), habs (sum_pos |out| per head), T0, M0

  python upstream.py SHARD NSHARD
  python upstream.py verify   # zero single heads (6 most positive, 6 most negative by H) at every 2nd fork and compare the
                              # actual change in M with -H -> results/upstream/heads_verify.json
"""
import json
import sys
import time

import numpy as np
import torch

from attrib import NI, NL, QCLAMP, Model, forks, prefix_ids
from common import RES, JsonlStore, log

OUT = RES / "upstream"
LATE = 48
NH = 48


class UModel(Model):
    def __init__(self):
        super().__init__()
        self.hacts, self.hd = {}, []
        for l, layer in enumerate(self.m.model.layers):
            full = hasattr(layer, "self_attn")
            mod = layer.self_attn.o_proj if full else layer.linear_attn.out_proj
            self.hd.append(256 if full else 128)
            assert mod.in_features == 6144, (l, mod.in_features)
            mod.register_forward_pre_hook(self._opre(l))

    def _opre(self, l):
        def hook(mod, inp):
            hz = getattr(self, "hzero", None)  # {layer: [head, ...]} to zero
            if hz and l in hz:
                x, hd = inp[0].clone(), self.hd[l]
                for h in hz[l]:
                    x[..., h * hd:(h + 1) * hd] = 0
                return (x,)
            if inp[0].requires_grad:
                inp[0].retain_grad()
                self.hacts[l] = inp[0]
        return hook

    def heads(self, layers):
        H, habs = np.full((NL, NH), np.nan, np.float32), np.full((NL, NH), np.nan, np.float32)
        for l in layers:
            x = self.hacts[l]
            nh = 6144 // self.hd[l]
            ag = (x.detach().float() * x.grad.float())[0].view(-1, nh, self.hd[l])
            H[l, :nh] = ag.sum((0, 2)).cpu().numpy()
            habs[l, :nh] = x.detach().float()[0].view(-1, nh, self.hd[l]).norm(dim=2).sum(0).cpu().numpy()
        return H, habs


def writers(f, i):
    """The behavior's top-100 set (sets.py) restricted to layers >= LATE; for behaviors without a set (fewer than
    3 forks), the fork's own 100 highest-attribution neurons there."""
    sets = json.loads((RES / "ablation_sets.json").read_text())
    if f"{f['behavior_id']}|top100" in sets:
        s = {int(l): v for l, v in sets[f"{f['behavior_id']}|top100"]["spec"].items()}
    else:
        A = np.load(RES / "attrib" / f"{i}.npz")["A"][LATE:].ravel()
        s = {}
        for j in np.argsort(-A)[:100]:
            s.setdefault(LATE + int(j) // NI, []).append(int(j) % NI)
    return {l: torch.tensor(sorted(v), device="cuda") for l, v in s.items() if l >= LATE}


def one(M, f, ids, W):
    toks = torch.tensor([x["tok"] for x in f["next_tokens"]], device="cuda")
    q = torch.tensor([x["q"] for x in f["next_tokens"]], device="cuda").clamp(*QCLAMP)
    M.set_ablation([])
    M.hacts = {}
    m, V, _ = M.metric(ids, toks, q, grad=True)
    m.backward()
    H, habs = M.heads(range(NL))
    g = {l: M.acts[l].grad[0][:, idx].detach().clone() for l, idx in W.items()}
    M.acts, M.hacts = {}, {}
    M.metric(ids, toks, q, grad=True)
    T = sum((M.acts[l][0][:, idx].float() * g[l].float()).sum() for l, idx in W.items())
    T.backward()
    AT = np.zeros((NL, NI), np.float32)
    for l in range(LATE):
        a = M.acts[l]
        AT[l] = (a.detach().float() * a.grad.float())[0].sum(0).cpu().numpy()
    HT, _ = M.heads(range(LATE))
    M.acts, M.hacts = {}, {}
    return dict(H=H, HT=HT, AT=AT, habs=habs, T0=float(T), M0=float(m), V0=float(V))


def main(shard, nshard):
    OUT.mkdir(exist_ok=True)
    gen = JsonlStore(RES / "stage_a_gen.jsonl").done
    F = forks()
    todo = [(i, f) for i, f in enumerate(F) if i % nshard == shard and not (OUT / f"{i}.npz").exists()]
    if not todo:
        return
    M = UModel()
    for i, f in todo:
        t0 = time.time()
        r = one(M, f, prefix_ids(f, gen), writers(f, i))
        np.savez_compressed(OUT / f"{i}.npz", **r, key=f["key"])
        log(f"fork {i} {f['behavior_id'][:20]}: M0 {r['M0']:.2f} T0 {r['T0']:.2f}; heads sum {np.nansum(r['H']):.2f} "
            f"top head {np.nanmax(r['H']):.2f}; upstream-through-writers sum {r['AT'][:LATE].sum():.2f}; {time.time() - t0:.0f}s")
        torch.cuda.empty_cache()


def verify():
    gen = JsonlStore(RES / "stage_a_gen.jsonl").done
    F = forks()
    M = UModel()
    out = []
    for i, f in list(enumerate(F))[::2]:
        ids = prefix_ids(f, gen)
        toks = torch.tensor([x["tok"] for x in f["next_tokens"]], device="cuda")
        q = torch.tensor([x["q"] for x in f["next_tokens"]], device="cuda").clamp(*QCLAMP)
        H = np.load(OUT / f"{i}.npz")["H"]
        Hf = np.nan_to_num(H, nan=0.0).ravel()
        M.set_ablation([])
        m0 = float(M.metric(ids, toks, q)[0])
        order = np.argsort(-Hf)
        for j in list(order[:6]) + list(order[::-1][:6]):
            l, h = divmod(int(j), NH)
            M.hzero = {l: [h]}
            out.append({"fork": i, "head": [l, h], "H": float(Hf[j]), "dM": float(M.metric(ids, toks, q)[0]) - m0})
            M.hzero = None
    (OUT / "heads_verify.json").write_text(json.dumps(out))
    x, y = -np.array([o["H"] for o in out]), np.array([o["dM"] for o in out])
    log(f"{len(out)} single-head ablations: r(predicted, actual) = {np.corrcoef(x, y)[0, 1]:.3f}; "
        f"median actual/predicted {np.median(y / x):.2f}")


if __name__ == "__main__":
    verify() if sys.argv[1] == "verify" else main(int(sys.argv[1]), int(sys.argv[2]))
