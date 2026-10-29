"""API smoke test: token-id generation, prefix continuation, next-token logprobs, judge parsing."""
import asyncio
import json

from common import Pool, generate, judge, load_prompts, next_token_logprobs, prompt_ids, tok


async def main():
    q, g = Pool("qwen", 64), Pool("gemma", 64)
    pr = load_prompts()
    r = pr[pr.behavior_id == "denying-ai-identity"].iloc[0]
    ids = prompt_ids(r.user_text)
    print("prompt tail:", repr(tok().decode(ids[-12:])))
    outs = await generate(q, ids, n=2, max_tokens=200)
    for o in outs:
        print("gen:", o["finish"], len(o["ids"] or []), repr(o["text"][:150]))
        assert o["ids"] and tok().decode(o["ids"], skip_special_tokens=True) == o["text"], "id/text mismatch"
    cont = await generate(q, ids + outs[0]["ids"][:20], n=1, max_tokens=30)
    print("cont:", repr(tok().decode(outs[0]["ids"][:20]) + "|" + cont[0]["text"][:80]))
    lp = await next_token_logprobs(q, ids)
    print("logprobs keys:", list(lp.keys()), json.dumps(lp["logprobs"])[:400])
    for th in (True, False):
        j = await judge(g, r.behavior_id, r.user_text, outs[0]["text"], thinking=th)
        print("judge thinking", th, j)


asyncio.run(main())
