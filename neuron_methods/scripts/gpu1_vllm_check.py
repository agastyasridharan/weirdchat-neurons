"""GPU job 1, part 3 (client env): gates G0 and G1 through the vLLM server, measured with the repo's client code
(code/rate.py nll and fork_V, code/common.py generate), so the numbers are computed exactly as in the original runs.
  G0: the ablation extension sees 64 layers and down_proj of shape [5120, 17408] (results/rate_cond.jsonl "check").
  G1: base NLL on the 200 reference texts within 0.001 nats/token of the original value (0.52562), and the damage of the repo's
      fabricated-code-execution top-100 set within 0.0005 of the original value (0.0019; results/rate_cond.jsonl, logs/rate2.log:19).
Information only: V at the 46 forks, that restoring the weights restores the NLL, and that sampling works.
Reads the repo's files only; prints PASS/FAIL lines.
    python gpu1_vllm_check.py http://127.0.0.1:PORT       (with WCN_SERVERS and PYTHONPATH=$WCN_ROOT/code set)
"""
import asyncio
import json
import sys
import time

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
from common import RES, JsonlStore, generate  # noqa: E402  (code/common.py)
from rate import One, fork_V, nll  # noqa: E402  (code/rate.py)

URL = sys.argv[1]
OK = []


def check(name, ok, detail=""):
    OK.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  [{detail}]", flush=True)


async def ablate(q, spec):  # as code/rate.py:131-133
    r = await q.post("/collective_rpc", {"method": "set_ablation", "args": [json.dumps(spec)]})
    return r["results"][0]


async def nlls(q, refs):
    ns = await asyncio.gather(*[nll(q, x) for x in refs])
    return np.array([n for n, _ in ns]), np.array([t for _, t in ns])


async def main():
    q = One(URL, 256)
    orig = JsonlStore(RES / "rate_cond.jsonl").done
    sets = json.loads((RES / "ablation_sets.json").read_text())
    refs = json.loads((RES / "ref_texts.json").read_text())
    gen = JsonlStore(RES / "stage_a_gen.jsonl").done
    forks = [json.loads(line) for line in (RES / "phase2_forks.jsonl").read_text().splitlines() if line.strip()]
    fork_ids = []
    for f in forks:  # as code/rate.py:91-94
        pid, si = f["rid"].split("|")
        fork_ids.append(gen[pid]["prompt_ids"] + list(gen[pid]["samples"][int(si)]["ids"])[:f["fork"][0]])
    hb, hf = orig["base"], orig["fabricated-code-execution|top100"]

    chk = await ablate(q, {})
    check("G0: the ablation extension sees 64 layers and down_proj [5120, 17408]",
          chk["layers"] == 64 and chk["shape"] == [5120, 17408] and chk["zeroed"] == 0, str(chk))

    t0 = time.time()
    nb, nt = await nlls(q, refs)
    ours, theirs = nb.sum() / nt.sum(), sum(hb["nll"]) / sum(hb["ntok"])
    check("the 200 reference texts have the original token counts", nt.tolist() == hb["ntok"], f"{int(nt.sum())} tokens")
    check("G1: base NLL within 0.001 of the original", abs(ours - theirs) <= 0.001,
          f"ours {ours:.6f}, original {theirs:.6f}, difference {ours - theirs:+.6f}; {time.time() - t0:.0f}s")
    per = np.abs(nb / nt - np.array(hb["nll"]) / np.array(hb["ntok"]))
    print(f"info  per-text NLL difference: median {np.median(per):.5f}, largest {per.max():.5f} nats/token", flush=True)
    vb = await asyncio.gather(*[fork_V(q, f, ids) for f, ids in zip(forks, fork_ids)])
    dv = np.abs(np.array([np.nan if v is None else v for v in vb]) - np.array(hb["V"], dtype=float))
    print(f"info  V at the 46 forks vs the original base run: median |difference| {np.nanmedian(dv):.4f}, "
          f"largest {np.nanmax(dv):.4f}, missing {sum(v is None for v in vb)}", flush=True)

    spec = sets["fabricated-code-execution|top100"]["spec"]
    chk = await ablate(q, spec)
    check("the original code-execution top-100 set applied (100 neurons zeroed)", chk["zeroed"] == 100, str(chk))
    na, _ = await nlls(q, refs)
    dmg = (na.sum() - nb.sum()) / nt.sum()
    orig_dmg = (sum(hf["nll"]) - sum(hb["nll"])) / sum(hf["ntok"])
    check("G1: damage of the original code-execution top-100 set within 0.0005 of the original", abs(dmg - orig_dmg) <= 0.0005,
          f"ours {dmg:.5f}, original {orig_dmg:.5f} nats/token")
    va = await asyncio.gather(*[fork_V(q, f, ids) for f, ids in zip(forks, fork_ids)])
    own = [i for i, f in enumerate(forks) if f["behavior_id"] == "fabricated-code-execution"]
    print(f"info  V at the {len(own)} code-execution forks with the set zeroed: ours mean "
          f"{np.mean([va[i] for i in own if va[i] is not None]):.3f}, original {np.mean([hf['V'][i] for i in own]):.3f}"
          f" (unedited, original: {np.mean([hb['V'][i] for i in own]):.3f})", flush=True)

    chk = await ablate(q, {})
    nr, _ = await nlls(q, refs)
    back = nr.sum() / nt.sum()
    check("restoring the weights gives the base NLL back", chk["zeroed"] == 0 and abs(back - ours) < 1e-4,
          f"{back:.6f} vs {ours:.6f}")

    pid = sorted(gen)[0]
    out = await generate(q, gen[pid]["prompt_ids"], 4, 64)
    check("sampling works with FlashInfer's sampler off (4 replies, up to 64 tokens)",
          len(out) == 4 and all(o["ids"] for o in out), ", ".join(o["finish"] for o in out))

    print(f"\n{sum(OK)} of {len(OK)} checks passed.")
    print("GATES G0 AND G1 PASS" if all(OK) else "GATES G0 AND G1 FAIL.")
    return all(OK)


if __name__ == "__main__":
    sys.exit(0 if asyncio.run(main()) else 1)
