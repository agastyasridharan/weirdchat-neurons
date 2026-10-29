"""Stage A: bf16 base rates. All 403 Qwen3.6-27B WeirdChat prompts x N samples from the empty
prefix (the full prompt), then judged. These rollouts are also the source rollouts for fork search.

  python stage_a.py gen     # sample (qwen pool)
  python stage_a.py judge [--off] [--per N]   # judge samples (gemma pool); --per: first N samples per prompt
"""
import asyncio
import sys

from common import RES, JsonlStore, Pool, generate, get_split, judge, load_prompts, log, prompt_ids, MAX_RESP

N = 64


async def gen():
    pr = load_prompts()
    split = get_split()
    store = JsonlStore(RES / "stage_a_gen.jsonl")
    pool = Pool("qwen", 8)  # each request carries n=64

    async def one(r):
        if r.prompt_id in store.done:
            return
        ids = prompt_ids(r.user_text)
        outs = await generate(pool, ids, n=N, max_tokens=MAX_RESP)
        store.add({"key": r.prompt_id, "prompt_id": r.prompt_id, "behavior_id": r.behavior_id,
                   "pattern_id": r.pattern_id, "split": split[r.prompt_id]["split"], "prompt_ids": ids,
                   "samples": [{"ids": o["ids"], "text": o["text"], "finish": o["finish"]} for o in outs]})
        log(f"gen {len(store.done)}/{len(pr)}")

    await asyncio.gather(*[one(r) for r in pr.itertuples()])


async def judge_all(thinking: bool, per: int | None = None):
    pr = load_prompts().set_index("prompt_id")
    gen_store = JsonlStore(RES / "stage_a_gen.jsonl")
    store = JsonlStore(RES / f"stage_a_judge_{'on' if thinking else 'off'}.jsonl")
    pool = Pool("gemma", 192)
    # discovery prompts first (fork search needs them), in prompt order so prompts complete early
    order = sorted(gen_store.done, key=lambda pid: (gen_store.done[pid]["split"] != "discovery", pid))
    todo = [(pid, i) for pid in order for i in range(min(per or N, len(gen_store.done[pid]["samples"])))
            if f"{pid}|{i}" not in store.done]
    log(f"judging {len(todo)}")

    async def one(pid, i):
        d = gen_store.done[pid]
        j = await judge(pool, d["behavior_id"], pr.loc[pid, "user_text"], d["samples"][i]["text"], thinking)
        store.add({"key": f"{pid}|{i}", "prompt_id": pid, "i": i, **j})
        if len(store.done) % 1000 == 0:
            log(f"judged {len(store.done)}")

    q = asyncio.Queue()
    for t in todo:
        q.put_nowait(t)

    async def worker():
        while not q.empty():
            await one(*q.get_nowait())

    await asyncio.gather(*[worker() for _ in range(len(pool.urls) * 192)])


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        asyncio.run(gen())
    else:
        asyncio.run(judge_all(thinking="--off" not in sys.argv,
                              per=int(sys.argv[sys.argv.index("--per") + 1]) if "--per" in sys.argv else None))
