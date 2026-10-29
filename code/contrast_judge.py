"""Phase 3a input: thinking-on judgments for the Stage A replies that the thinking-off judge called positive but that
were never judged with thinking on (only the first 16 of 64 per prompt were). Discovery split only. Kept apart from
stage_a_judge_on.jsonl, which defines the Stage A rates (adding judged positives there would bias them).

  python contrast_judge.py   -> results/contrast_judge_on.jsonl (same record format as stage_a_judge_on.jsonl)
"""
import asyncio

from common import RES, JsonlStore, Pool, get_split, judge, load_prompts, log


async def main():
    gen = JsonlStore(RES / "stage_a_gen.jsonl").done
    off, on = JsonlStore(RES / "stage_a_judge_off.jsonl").done, JsonlStore(RES / "stage_a_judge_on.jsonl").done
    store = JsonlStore(RES / "contrast_judge_on.jsonl")
    pr = load_prompts().set_index("prompt_id")
    sp = get_split()
    todo = [(j["prompt_id"], j["i"]) for k, j in sorted(off.items()) if j["match"] and k not in on and k not in store.done
            and sp[j["prompt_id"]]["split"] == "discovery"]
    log(f"judging {len(todo)} thinking-off positives with thinking on")
    pool = Pool("gemma", 96)
    q = asyncio.Queue()
    for t in todo:
        q.put_nowait(t)

    async def worker():
        while not q.empty():
            pid, i = q.get_nowait()
            d = gen[pid]
            j = await judge(pool, d["behavior_id"], pr.loc[pid, "user_text"], d["samples"][i]["text"], True)
            store.add({"key": f"{pid}|{i}", "prompt_id": pid, "i": i, **j})
            if len(store.done) % 250 == 0:
                log(f"judged {len(store.done)}")

    await asyncio.gather(*[worker() for _ in range(len(pool.urls) * 96)])
    log("done")


if __name__ == "__main__":
    asyncio.run(main())
