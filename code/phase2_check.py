"""Phase 2 readiness: do HF bf16 forward/backward reproduce vLLM's next-token distribution at fork prefixes?

  cenv: python phase2_check.py collect   # vLLM raw top-20 logprobs at N fork prefixes (qwen server)
  genv: python phase2_check.py compare   # HF bf16 (+fla kernels) logprobs at the same prefixes, KL / top-1
                                         # agreement, and one backward pass of M = logit(V) for timing/memory
"""
import json
import random
import sys
import time

from common import RES, JsonlStore, log

N = 48
OUT = RES / "phase2_vllm_logprobs.jsonl"


def prefixes():
    forks = sorted(JsonlStore(RES / "forks_off.jsonl").done.values(), key=lambda f: f["rid"])
    random.Random(1).shuffle(forks)
    gen = JsonlStore(RES / "stage_a_gen.jsonl").done
    out = []
    for f in forks[:N]:
        pid, si = f["rid"].split("|")
        out.append((f, gen[pid]["prompt_ids"] + list(gen[pid]["samples"][int(si)]["ids"])[:f["fork"][0]]))
    return out


def collect():
    import asyncio
    from common import Pool, next_token_logprobs

    async def main():
        pool = Pool("qwen", 16)
        store = JsonlStore(OUT)
        res = await asyncio.gather(*[next_token_logprobs(pool, ids, k=20) for _, ids in prefixes()])
        for (f, ids), r in zip(prefixes(), res):
            top = {int(k.split(":")[1]): v for k, v in r["logprobs"]["top_logprobs"][0].items()}
            store.add({"key": f["rid"], "ids": ids, "top": top})
        log(f"collected {len(res)}")
    asyncio.run(main())


def compare():
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM
    from common import QWEN_SNAP
    recs = list(JsonlStore(OUT).done.values())
    forks = JsonlStore(RES / "forks_off.jsonl").done
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(QWEN_SNAP, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    log(f"loaded in {time.time() - t0:.0f}s; fla fast path: "
        f"{__import__('transformers.models.qwen3_5.modeling_qwen3_5', fromlist=['x']).is_fast_path_available}")
    kls, top1, maxdiff = [], [], []
    for r in recs:
        with torch.no_grad():
            lp = torch.log_softmax(model(torch.tensor([r["ids"]], device="cuda")).logits[0, -1].float(), -1)
        v = {int(k): x for k, x in r["top"].items()}
        toks = list(v)
        pv = np.exp(np.array([v[t] for t in toks]))
        ph = lp[toks].exp().cpu().numpy()
        kls.append(float(np.sum(pv * (np.log(pv) - np.log(ph))) ))  # KL(vLLM||HF) over vLLM's top-20 support
        top1.append(int(lp.argmax()) == max(v, key=v.get))
        maxdiff.append(float(np.max(np.abs(np.log(pv) - np.log(ph)))))
    log(f"n={len(recs)}  KL(vllm||hf) top-20 median {np.median(kls):.4f} max {np.max(kls):.4f}  "
        f"top-1 agree {np.mean(top1):.3f}  max|dlogp| median {np.median(maxdiff):.3f}")
    # one backward pass of the fork metric M = logit(sum_v p(v) q_v) wrt the residual stream, longest prefix
    r = max(recs, key=lambda r: len(r["ids"]))
    f = forks[r["key"]]
    toks = torch.tensor([x["tok"] for x in f["next_tokens"]], device="cuda")
    q = torch.tensor([x["q"] for x in f["next_tokens"]], device="cuda").clamp(0.02, 0.98)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model.zero_grad()
    out = model(torch.tensor([r["ids"]], device="cuda"), output_hidden_states=True)
    for h in out.hidden_states:
        h.retain_grad()
    p = torch.softmax(out.logits[0, -1].float(), -1)[toks]
    V = (p * q).sum() / p.sum()
    M = torch.log(V / (1 - V))
    M.backward()
    torch.cuda.synchronize()
    g = [h.grad[0, -1].norm().item() for h in out.hidden_states]
    log(f"backward ok: len {len(r['ids'])} tokens, {time.time() - t0:.1f}s, peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, "
        f"M {M.item():.3f}, |dM/dh_last| by layer (every 8th): {[round(x, 4) for x in g[::8]]}")


if __name__ == "__main__":
    collect() if sys.argv[1] == "collect" else compare()
