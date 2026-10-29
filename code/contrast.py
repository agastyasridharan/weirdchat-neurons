"""Phase 3a: reply-level attribution for every behavior, forkable or not (Qwen3.6-27B, HF bf16, genv).

A fork exists only where one token decides the behavior; gradual behaviors have none. Here the target is a whole
Stage A reply r (prompt x, reply y cut to MAXR tokens): L_r = mean_t log p(y_t | x, y_<t). Per reply we store
G_r = a * dL_r/da summed over all positions, for every MLP neuron (64 x 17408) and attention head (64 x 48, the
head's slice of the o_proj / out_proj input). Attribution is linear in the target, so any contrast between replies
is the same combination of the G_r:
  behavior contrast   mean G over a prompt's behavior replies - mean G over its non-behavior replies: the first-order
                      change in their log-likelihood ratio when the unit is zeroed (positive = the unit makes the
                      behavior reply more likely than the other replies to the same prompt)
  placebo             one half of the non-behavior replies - the other half: the noise floor for generic units
Prompts: discovery split, >= 2 behavior and >= 2 non-behavior replies under the thinking-on judge (Stage A's 16 per
prompt plus contrast_judge.py's re-judged thinking-off positives); up to NREP of
each (random, fixed seed), up to PER_B prompts per behavior (random, fixed seed).

  python contrast.py list            # prompts per behavior that qualify
  python contrast.py verify          # 2 prompts per behavior: zero the top-k contrast neurons (and matched random ones)
                                     and compare the actual change in the contrast with -sum(attribution)
                                     -> results/contrast/verify.json
  python contrast.py SHARD NSHARD    -> results/contrast/<pid>.npz: G (n x 64 x 17408), GH (n x 64 x 48), both bf16
                                        stored as int16 bits; lab (1 behavior / 0 not), L, ntok, absa (sum_pos |a|,
                                        mean over replies, bf16 bits)
"""
import json
import sys
import time
from collections import defaultdict

import numpy as np
import torch

from attrib import NI, NL
from common import RES, JsonlStore, get_split, load_prompts, log
from upstream import UModel

OUT = RES / "contrast"
MAXR, NREP, PER_B = 512, 6, 40


def bits(x):
    return x.to(torch.bfloat16).view(torch.int16).cpu().numpy()


def unbits(x):
    return torch.from_numpy(np.ascontiguousarray(x)).view(torch.bfloat16).float().numpy()


def plan():
    """[(pid, behavior, [(sample index, label), ...])] for every qualifying discovery prompt, capped per behavior."""
    J = JsonlStore(RES / "stage_a_judge_on.jsonl").done | JsonlStore(RES / "contrast_judge_on.jsonl").done
    pr = load_prompts().set_index("prompt_id")
    sp = get_split()
    m = defaultdict(list)
    for j in J.values():
        if j["match"] is not None:
            m[j["prompt_id"]].append((j["i"], int(j["match"])))
    rng = np.random.default_rng(5)
    byb = defaultdict(list)
    for pid, v in sorted(m.items()):
        pos, neg = [i for i, y in v if y], [i for i, y in v if not y]
        if sp[pid]["split"] != "discovery" or len(pos) < 2 or len(neg) < 2:
            continue
        pos = sorted(rng.permutation(pos)[:NREP].tolist())
        neg = sorted(rng.permutation(neg)[:NREP].tolist())
        byb[pr.loc[pid, "behavior_id"]].append((pid, [(i, 1) for i in pos] + [(i, 0) for i in neg]))
    out = []
    for b, ps in sorted(byb.items()):
        for k in sorted(rng.permutation(len(ps))[:PER_B].tolist()):
            out.append((ps[k][0], b, ps[k][1]))
    return out, byb


def one(M, ids, P):
    """Mean log-prob of the reply tokens ids[P:] and its attribution to every neuron and head."""
    M.set_ablation([])
    M.acts, M.hacts = {}, {}
    x = torch.tensor([ids], device="cuda")
    R = len(ids) - P
    emb = M.m.model.embed_tokens(x).requires_grad_(True)
    logits = M.m(inputs_embeds=emb, logits_to_keep=R + 1).logits[0, :-1].float()
    L = torch.log_softmax(logits, -1).gather(1, x[0, P:, None]).mean()
    L.backward()
    G = torch.zeros(NL, NI, device="cuda")
    absa = torch.zeros(NL, NI, device="cuda")
    for l, a in M.acts.items():
        G[l] = (a.detach().float() * a.grad.float())[0].sum(0)
        absa[l] = a.detach().float()[0].abs().sum(0)
    H, _ = M.heads(range(NL))
    M.acts, M.hacts = {}, {}
    return G, torch.from_numpy(np.nan_to_num(H, nan=0.0)), absa, L.item()


def main(shard, nshard):
    OUT.mkdir(exist_ok=True)
    gen = JsonlStore(RES / "stage_a_gen.jsonl").done
    todo = [t for k, t in enumerate(plan()[0]) if k % nshard == shard and not (OUT / f"{t[0]}.npz").exists()]
    if not todo:
        return
    M = UModel()
    for pid, b, reps in todo:
        t0 = time.time()
        g = gen[pid]
        P = len(g["prompt_ids"])
        Gs, Hs, Ls, nt = [], [], [], []
        absa = torch.zeros(NL, NI, device="cuda")
        for i, _ in reps:
            resp = list(g["samples"][i]["ids"])[:MAXR]
            G, H, a, L = one(M, g["prompt_ids"] + resp, P)
            Gs.append(bits(G)), Hs.append(H.numpy().astype(np.float32)), Ls.append(L), nt.append(len(resp))
            absa += a / len(reps)
            torch.cuda.empty_cache()
        lab = np.array([y for _, y in reps])
        np.savez(OUT / f"{pid}.npz", G=np.stack(Gs), GH=np.stack(Hs), lab=lab, L=np.array(Ls), ntok=np.array(nt),
                 absa=bits(absa), idx=np.array([i for i, _ in reps]), b=b)
        Lp, Ln = np.mean([l for l, y in zip(Ls, lab) if y]), np.mean([l for l, y in zip(Ls, lab) if not y])
        log(f"{pid} {b[:22]}: {lab.sum()} behavior / {(1 - lab).sum()} other replies, prompt {P} tok, "
            f"mean logp {Lp:.3f} vs {Ln:.3f}; {time.time() - t0:.0f}s")


def contrast(z):
    G, lab = unbits(z["G"]), z["lab"]
    return G[lab == 1].mean(0) - G[lab == 0].mean(0)


def verify():
    from attrib import matched
    gen = JsonlStore(RES / "stage_a_gen.jsonl").done
    seen, todo = defaultdict(int), []
    for pid, b, reps in plan()[0]:
        if seen[b] < 2 and (OUT / f"{pid}.npz").exists():
            seen[b] += 1
            todo.append((pid, b, reps))
    M = UModel()
    rng, out = np.random.default_rng(0), []

    def R(ids_list, P, lab, flat):
        M.set_ablation(flat)
        Ls = []
        for ids in ids_list:
            x = torch.tensor([ids], device="cuda")
            with torch.no_grad():
                lg = M.m(x, logits_to_keep=len(ids) - P + 1).logits[0, :-1].float()
            Ls.append(float(torch.log_softmax(lg, -1).gather(1, x[0, P:, None]).mean()))
        Ls = np.array(Ls)
        return Ls[lab == 1].mean() - Ls[lab == 0].mean()

    for pid, b, reps in todo:
        z = np.load(OUT / f"{pid}.npz")
        C, absa, lab = contrast(z).ravel(), unbits(z["absa"]), z["lab"]
        g = gen[pid]
        P = len(g["prompt_ids"])
        ids_list = [g["prompt_ids"] + list(g["samples"][i]["ids"])[:MAXR] for i, _ in reps]
        r0 = R(ids_list, P, lab, [])
        order = np.argsort(-C)
        for k in [1, 10, 100]:
            top = order[:k]
            out.append({"pid": pid, "b": b, "k": k, "pred": -float(C[top].sum()), "act": R(ids_list, P, lab, top) - r0,
                        "rand": R(ids_list, P, lab, matched(top, absa, rng)) - r0})
            log(f"{pid} {b[:20]} k={k}: predicted {out[-1]['pred']:.4f} actual {out[-1]['act']:.4f} random {out[-1]['rand']:.4f}")
    M.set_ablation([])
    (OUT / "verify.json").write_text(json.dumps(out))
    x, y = np.array([o["pred"] for o in out]), np.array([o["act"] for o in out])
    log(f"{len(out)} set ablations: r(predicted, actual) = {np.corrcoef(x, y)[0, 1]:.3f}, median actual/predicted "
        f"{np.median(y / x):.2f}; random sets median |change| {np.median([abs(o['rand']) for o in out]):.4f}")


if __name__ == "__main__":
    if sys.argv[1] == "list":
        P, byb = plan()
        for b, ps in sorted(byb.items()):
            print(f"{b:32s} qualifying {len(ps):3d}  used {sum(1 for _, x, _ in P if x == b)}")
        print(len(P), "prompts,", sum(len(r) for _, _, r in P), "replies")
    elif sys.argv[1] == "verify":
        verify()
    else:
        main(int(sys.argv[1]), int(sys.argv[2]))
