"""Phase 2f: is there ONE neuron below layer 48 whose removal removes a behavior? Real single-neuron ablation screen
(cenv; qwen_abl servers), instead of trusting first-order scores (Phase 2d showed they can get the sign wrong).

Candidates (layers < 48, deduplicated; source tags kept):
  z:<b>      top NZ by the behavior's reply-contrast score z_b (sets3.py), for the 8 behaviors in the shared score
  shared     top NS by the shared (median-over-behaviors) score
  pair       top NP by min(z_denying-ai-identity, z_false-physical-embodiment)
  mid:<b>    top NF by fork attribution to M, per Phase 2 behavior (attrib/, as upsets.py)
  med:<b>    top NF by attribution through the late writers (upstream/, as upsets.py)
  rand       NR random neurons, each in the layer of a random candidate (the null distribution for single neurons)
Per candidate, zeroed alone on a qwen_abl server:
  L     mean log p of the first RMAX tokens of 192 Stage A replies (8 behaviors x 6 discovery prompts x 2 behavior
        + 2 other replies, from the contrast.py data); the behavior contrast change of a prompt is
        mean dL(behavior replies) - mean dL(other replies)
  V     fork value at the 46 Phase 2 forks
  nll   NLL of 40 reference replies (damage)
Each server also scores the unablated model ("base|<port>"), so every change is taken against the same server.

About 1 in 7,800 prompt-logprob requests came back corrupted (one reply's mean log p near -10, any server, nothing
else in that run moving). `redo` re-measures every neuron with a reply off by more than GLITCH nats/token from base;
the new record replaces the old one. A real effect would recur on the same reply.

`sets` measures, the same way, the 100-neuron sets that removed or cut behaviors in rate tests (Phase 2 top100 per
behavior, sharedmid100, selfpair100), as the scale a single neuron would have to reach (keys "set|<name>").

`picks` writes results/single_sets.json for rate3.py (RATE3_MODE=single): the PICKS below (the strongest contrast
change for the behaviors where one neuron rivals a working 100-neuron set, plus the two largest fork-value drops), each
with a random screened neuron from the nearest layer as its control.

  python screen1.py cands | run | redo | sets | summary | picks | export   (export -> results/site_screen.json)
"""
import asyncio
import json
import os
import sys
from collections import defaultdict

import numpy as np

from common import RES, JsonlStore, log
from rate import One, fork_V, nll

NL, NI, LATE = 64, 17408, 48
NZ, NS, NP, NF, NR = 60, 150, 40, 60, 150
NPROMPT, RMAX = 6, 256
GLITCH = 0.5
B8 = ["chemtrails-assertion", "claims-called-911", "cutting-instructions", "denying-ai-identity", "fabricated-code-execution",
      "false-physical-embodiment", "laser-at-aircraft", "recommends-drunk-driving"]
B5 = ["claims-called-911", "denying-ai-identity", "fabricated-code-execution", "false-physical-embodiment", "laser-at-aircraft"]


def top(x, k):
    x = np.where(np.arange(x.size) < LATE * NI, x, -np.inf)
    return [int(j) for j in np.argsort(-x)[:k]]


def cands():
    Z = np.load(RES / "shared_z.npz")
    src = defaultdict(list)
    for b in B8:
        for j in top(Z[f"z|{b}"], NZ):
            src[j].append(f"z:{b}")
    for j in top(Z["shared"], NS):
        src[j].append("shared")
    for j in top(Z["pair"], NP):
        src[j].append("pair")
    F = [json.loads(l) for l in (RES / "phase2_forks.jsonl").read_text().splitlines()]
    norm = lambda x: x / np.sort(x.ravel())[::-1][:100].sum()
    for b in B5:
        sM, sT = np.zeros(NL * NI), np.zeros(NL * NI)
        for i, f in enumerate(F):
            if f["behavior_id"] == b:
                sM += norm(np.load(RES / "attrib" / f"{i}.npz")["A"].astype(np.float64)).ravel()
                sT += norm(np.load(RES / "upstream" / f"{i}.npz")["AT"].astype(np.float64)).ravel()
        for j in top(sM, NF):
            src[j].append(f"mid:{b}")
        for j in top(sT, NF):
            src[j].append(f"med:{b}")
    rng = np.random.default_rng(13)
    pool = sorted(src)
    for _ in range(NR):
        l = pool[rng.integers(len(pool))] // NI
        j = l * NI + int(rng.integers(NI))
        while j in src:
            j = l * NI + int(rng.integers(NI))
        src[j].append("rand")
    out = [{"n": j, "src": s} for j, s in sorted(src.items())]
    (RES / "screen1_cands.json").write_text(json.dumps(out))
    print(f"{len(out)} candidates ({sum('rand' in c['src'] for c in out)} random); by source:",
          dict(sorted(defaultdict(int, {k: sum(k in c['src'] for c in out) for k in {t for c in out for t in c['src']}}).items())))
    print("layers/8:", np.bincount([c["n"] // NI // 8 for c in out], minlength=6).tolist())


def items():
    """The 192 screening replies: per behavior NPROMPT discovery prompts from contrast.py's data, 2 + 2 replies each."""
    gen = JsonlStore(RES / "stage_a_gen.jsonl").done
    byb = defaultdict(list)
    for f in sorted((RES / "contrast").glob("*.npz")):
        z = np.load(f)
        byb[str(z["b"])].append((f.stem, z["idx"], z["lab"]))
    rng = np.random.default_rng(17)
    out = []
    for b in B8:
        for k in sorted(rng.permutation(len(byb[b]))[:NPROMPT].tolist()):
            pid, idx, lab = byb[b][k]
            g = gen[pid]
            for y in (1, 0):
                for i in [int(i) for i, l in zip(idx, lab) if l == y][:2]:
                    out.append({"pid": pid, "b": b, "lab": y, "P": len(g["prompt_ids"]),
                                "ids": g["prompt_ids"] + list(g["samples"][i]["ids"])[:RMAX]})
    return out


async def mlogp(q, it):
    body = {"model": "qwen", "prompt": it["ids"], "max_tokens": 1, "prompt_logprobs": 0,
            "return_tokens_as_token_ids": True, "skip_special_tokens": False}
    out = (await q.post("/v1/completions", body))["choices"][0]["prompt_logprobs"]
    lp = []
    for t in range(it["P"], len(it["ids"])):
        d = out[t]
        hit = [v for k, v in d.items() if k.split(":")[-1] == str(it["ids"][t])] or list(d.values())
        lp.append(hit[0]["logprob"])
    return float(np.mean(lp))


def glitched():
    """Neurons whose record has one reply more than GLITCH nats/token from its server's base."""
    S = JsonlStore(RES / "screen1.jsonl").done
    base = defaultdict(list)
    for k, r in S.items():
        if k.startswith("base|"):
            base[r["server"]].append(np.array(r["L"]))
    B = {u: np.mean(v, 0) for u, v in base.items()}
    return {k for k, r in S.items() if not k.startswith(("base|", "set|")) and np.abs(np.array(r["L"]) - B[r["server"]]).max() > GLITCH}


def ref_sets():
    A, S = json.loads((RES / "ablation_sets.json").read_text()), json.loads((RES / "shared_sets.json").read_text())
    return {f"set|{b}|top100": A[f"{b}|top100"]["spec"] for b in B5} | {
        f"set|{k}": S[k]["spec"] for k in ["sharedmid100", "rand_sharedmid100", "selfpair100"]}


PICKS = [("cutting-instructions", 2, 9382, "contrast"), ("false-physical-embodiment", 23, 3665, "contrast"),
         ("chemtrails-assertion", 24, 8304, "contrast"), ("claims-called-911", 38, 2788, "contrast"),
         ("recommends-drunk-driving", 38, 16525, "contrast"), ("false-physical-embodiment", 28, 17401, "fork"),
         ("claims-called-911", 27, 9339, "fork")]


def picks():
    C = json.loads((RES / "screen1_cands.json").read_text())
    rand = sorted(c["n"] for c in C if c["src"] == ["rand"])
    rng, out, used = np.random.default_rng(3), {}, set()
    for b, l, i, why in PICKS:
        name = f"{b}|L{l}n{i}"
        out[name] = {"spec": {str(l): [i]}, "hi": b, "kind": "single", "b": b, "why": why}
        d = min(abs(n // NI - l) for n in rand if n not in used)
        r = int(rng.choice([n for n in rand if abs(n // NI - l) == d and n not in used]))
        used.add(r)
        out[f"rand_{name}"] = {"spec": {str(r // NI): [r % NI]}, "hi": b, "kind": "rand_single", "b": b}
    (RES / "single_sets.json").write_text(json.dumps(out, indent=1))
    print({k: v["spec"] for k, v in out.items()})


async def run(redo=False, sets=False):
    C = json.loads((RES / "screen1_cands.json").read_text())
    I = items()
    refs = json.loads((RES / "ref_texts.json").read_text())[:40]
    gen = JsonlStore(RES / "stage_a_gen.jsonl").done
    F = [json.loads(l) for l in (RES / "phase2_forks.jsonl").read_text().splitlines()]
    fork_ids = []
    for f in F:
        pid, si = f["rid"].split("|")
        fork_ids.append(gen[pid]["prompt_ids"] + list(gen[pid]["samples"][int(si)]["ids"])[:f["fork"][0]])
    st = JsonlStore(RES / "screen1.jsonl")
    qurls = [u for u, r in json.loads(open(os.environ["WCN_SERVERS"]).read()).items() if r == "qwen_abl"]
    queue = asyncio.Queue()
    G = glitched() if redo else set()
    for c in ([] if sets else C):
        if str(c["n"]) not in st.done or str(c["n"]) in G:
            queue.put_nowait(c)
    if sets:
        for k, sp in ref_sets().items():
            if k not in st.done or k in glitched():
                queue.put_nowait({"key": k, "spec": sp})
    log(f"{len(C)} candidates, {queue.qsize()} to do; {len(I)} replies, {len(refs)} refs, {len(F)} forks; qwen {qurls}")

    async def measure(q, key, spec):
        r = (await q.post("/collective_rpc", {"method": "set_ablation", "args": [json.dumps(spec)]}))["results"][0]
        assert r["zeroed"] == sum(len(v) for v in spec.values()), (key, r)
        L = await asyncio.gather(*[mlogp(q, it) for it in I])
        V = await asyncio.gather(*[fork_V(q, f, ids) for f, ids in zip(F, fork_ids)])
        ns = await asyncio.gather(*[nll(q, x) for x in refs])
        st.add({"key": key, "server": q.urls[0], "L": L, "V": V, "nll": [n for n, _ in ns], "ntok": [t for _, t in ns]})

    async def worker(url):
        q = One(url, 256)
        port = url.rsplit(":", 1)[1]
        for rep in range(2):
            if f"base|{port}|{rep}" not in st.done:
                await measure(q, f"base|{port}|{rep}", {})
        while not queue.empty():
            c = queue.get_nowait()
            if "spec" in c:
                await measure(q, c["key"], c["spec"])
                continue
            l, i = divmod(c["n"], NI)
            await measure(q, str(c["n"]), {str(l): [i]})
            if len(st.done) % 100 == 0:
                log(f"screened {len(st.done)}")
        await q.post("/collective_rpc", {"method": "set_ablation", "args": ["{}"]})

    await asyncio.gather(*[worker(u) for u in qurls])
    log("done")


def table():
    """Per candidate: behavior-contrast change per behavior (nats/token), fork V change per Phase 2 behavior, damage."""
    C = {str(c["n"]): c for c in json.loads((RES / "screen1_cands.json").read_text())}
    I = items()
    S = JsonlStore(RES / "screen1.jsonl").done
    F = [json.loads(l) for l in (RES / "phase2_forks.jsonl").read_text().splitlines()]
    base = defaultdict(list)
    for k, r in S.items():
        if k.startswith("base|"):
            base[r["server"]].append(r)
    B = {u: {"L": np.mean([r["L"] for r in v], 0), "V": np.array([[x if x is not None else np.nan for x in r["V"]] for r in v]).mean(0),
             "nll": np.mean([r["nll"] for r in v], 0), "ntok": np.sum(v[0]["ntok"])} for u, v in base.items()}
    rep_noise = {u: float(np.abs(np.subtract(*[np.array(r["L"]) for r in v[:2]])).mean()) for u, v in base.items() if len(v) > 1}
    pid = np.array([it["pid"] for it in I])
    lab = np.array([it["lab"] for it in I])
    beh = np.array([it["b"] for it in I])
    rows = []
    for k, r in S.items():
        if k.startswith("base|"):
            continue
        b0 = B[r["server"]]
        dL = np.array(r["L"]) - b0["L"]
        dc = {}
        for b in B8:
            ps = sorted(set(pid[beh == b]))
            dc[b] = float(np.mean([dL[(pid == p) & (lab == 1)].mean() - dL[(pid == p) & (lab == 0)].mean() for p in ps]))
        V = np.array([x if x is not None else np.nan for x in r["V"]])
        dV = {b: float(np.nanmean([V[i] - b0["V"][i] for i, f in enumerate(F) if f["behavior_id"] == b])) for b in B5}
        if k.startswith("set|"):
            rows.append({"set": k[4:], "src": ["set"], "dc": dc, "dV": dV, "dL_all": float(dL.mean()),
                         "damage": float((np.array(r["nll"]) - b0["nll"]).sum() / b0["ntok"])})
            continue
        n = int(k)
        rows.append({"n": n, "L": n // NI, "i": n % NI, "src": C[k]["src"], "dc": dc, "dV": dV,
                     "damage": float((np.array(r["nll"]) - b0["nll"]).sum() / b0["ntok"]), "dL_all": float(dL.mean())})
    return rows, rep_noise


def summary():
    rows, noise = table()
    log(f"{len(rows)} neurons screened; base repeat noise (mean |dL| per reply) {noise}; "
        f"still glitched (a reply off by > {GLITCH}): {sorted(glitched())}")
    SETS = [r for r in rows if "set" in r]
    rows = [r for r in rows if "set" not in r]
    R = [r for r in rows if "rand" in r["src"]]
    T = [r for r in rows if r["src"] != ["rand"]]
    if SETS:
        print("reference: 100-neuron sets measured the same way (behavior-contrast change per behavior; damage)")
        print(" " * 34 + "".join(b[:10].rjust(11) for b in B8))
        for r in SETS:
            print(f"   {r['set'][:30]:30s} " + "".join(f"{r['dc'][b]:+.4f}".rjust(11) for b in B8) + f"   damage {r['damage']:.4f}")
        best = {b: min(T, key=lambda r: r["dc"][b]) for b in B8}
        print("   best single neuron              " + "".join(f"{best[b]['dc'][b]:+.4f}".rjust(11) for b in B8))
    print("random single neurons (null): behavior-contrast change per behavior, 5th / 1st percentile and min (nats/token):")
    lo = {}
    for b in B8:
        x = np.array([r["dc"][b] for r in R])
        lo[b] = np.percentile(x, 1)
        print(f"   {b:30s} p5 {np.percentile(x, 5):+.4f}  p1 {lo[b]:+.4f}  min {x.min():+.4f}")
    print("random damage p50 / p99:", np.percentile([r["damage"] for r in R], [50, 99]).round(4))
    for b in B8:
        best = sorted(T, key=lambda r: r["dc"][b])[:6]
        print(f"== {b}: strongest single neurons (contrast change; x below the random 1st percentile; damage; sources)")
        for r in best:
            v = f" dV {r['dV'][b]:+.3f}" if b in r["dV"] else ""
            print(f"   L{r['L']:2d} n{r['i']:5d}  {r['dc'][b]:+.4f} ({r['dc'][b] / lo[b]:.1f}x){v}  damage {r['damage']:.4f}  {','.join(r['src'])[:70]}")
    print("== broadest: below the random 5th percentile for the most behaviors")
    p5 = {b: np.percentile([r["dc"][b] for r in R], 5) for b in B8}
    broad = sorted(T, key=lambda r: (-sum(r["dc"][b] < p5[b] for b in B8), sum(r["dc"][b] for b in B8)))[:12]
    for r in broad:
        print(f"   L{r['L']:2d} n{r['i']:5d}  {sum(r['dc'][b] < p5[b] for b in B8)}/8  " + " ".join(f"{r['dc'][b]:+.3f}" for b in B8)
              + f"  damage {r['damage']:.4f}  {','.join(r['src'])[:50]}")
    print("== largest fork-value drops (single neuron, mean over the behavior's forks)")
    for b in B5:
        r = min(T, key=lambda r: r["dV"][b])
        print(f"   {b:30s} L{r['L']:2d} n{r['i']:5d} dV {r['dV'][b]:+.3f}  contrast {r['dc'].get(b, 0):+.4f}  damage {r['damage']:.4f}")


def export():
    rows, _ = table()
    SETS = [r for r in rows if "set" in r]
    rows = [r for r in rows if "set" not in r]
    R = [r for r in rows if "rand" in r["src"]]
    T = [r for r in rows if r["src"] != ["rand"]]
    r5 = lambda x: round(float(x), 5)
    out = {"B8": B8, "B5": B5, "n": len(rows), "n_rand": len(R),
           "null": {b: [r5(np.percentile([r["dc"][b] for r in R], q)) for q in (5, 1)] + [r5(min(r["dc"][b] for r in R))] for b in B8},
           "damage_rand": [r5(x) for x in np.percentile([r["damage"] for r in R], [50, 99])],
           "sets": [{"set": r["set"], "dc": {b: r5(r["dc"][b]) for b in B8}, "damage": r5(r["damage"])} for r in SETS],
           "dc": {b: {"cand": [r5(r["dc"][b]) for r in T], "rand": [r5(r["dc"][b]) for r in R]} for b in B8},
           "best": {b: [{"L": r["L"], "i": r["i"], "dc": r5(r["dc"][b]), "damage": r5(r["damage"]), "src": r["src"]}
                        for r in sorted(T, key=lambda r: r["dc"][b])[:5]] for b in B8},
           "fork": {b: [{"L": r["L"], "i": r["i"], "dV": r5(r["dV"][b]), "damage": r5(r["damage"])}
                        for r in sorted(T, key=lambda r: r["dV"][b])[:3]] for b in B5}}
    (RES / "site_screen.json").write_text(json.dumps(out))
    print("wrote", RES / "site_screen.json")


if __name__ == "__main__":
    {"cands": cands, "export": export, "run": lambda: asyncio.run(run()), "redo": lambda: asyncio.run(run(redo=True)),
     "sets": lambda: asyncio.run(run(sets=True)), "picks": picks,
     "summary": summary}[sys.argv[1]]()
