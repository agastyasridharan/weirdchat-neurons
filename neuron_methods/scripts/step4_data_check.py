"""STEP 4: the data gate. Checks, with no GPU, that the data we downloaded and the repo's files match what
the original runs used, exactly. A few minutes of light work; fine on a login node. Reads only; writes one log.
    source ~/scratch/wcn/setup/env.sh
    "$WCN_CLIENT_PY" ~/scratch/wcn/setup/step4_data_check.py

Each check prints PASS or FAIL; "DATA GATE PASS" at the end means all passed. The checks copy the logic
of the repo's code (file and line given at each check) instead of running it, then also load the data through
the repo's own code/common.py to confirm the repo's code reads our files the same way.
"""
import glob
import gzip
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

if "WCN_ROOT" not in os.environ:
    sys.exit("STOP: run 'source ~/scratch/wcn/setup/env.sh' first.")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BASE = Path(os.environ["WCN_BASE"])
P = Path(os.environ["WCN_ROOT"])
DATA, RES = P / "data", P / "results"
LOG = BASE / "logs" / f"step4_data_check_{time.strftime('%Y%m%d_%H%M%S')}.log"
_logf = open(LOG, "a")
RESULTS = []
FOUR = ["fabricated-code-execution", "claims-called-911", "laser-at-aircraft", "false-physical-embodiment"]


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _logf.write(s + "\n")
    _logf.flush()


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    log(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


def store(path):
    """The repo's JsonlStore(...).done: one record per key, the last line wins (code/common.py:244-261)."""
    done = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            done[d["key"]] = d
    return done


def _struct(x):  # code/common.py:29-33
    if isinstance(x, str):
        import ast
        return ast.literal_eval(x)
    return x


def load_prompts():  # code/common.py:36-41
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(DATA / "wc_q27/*prompts*")))])
    df["user_text"] = df.messages.apply(lambda m: [x for x in m if x["role"] == "user"][0]["content"])
    df["wc_n"] = df.match_summary.apply(lambda m: m["n_samples"])
    df["wc_k"] = df.match_summary.apply(lambda m: m["n_matched"])
    return df.drop(columns=["messages"]).reset_index(drop=True)


def load_transcripts():  # code/common.py:44-49
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(DATA / "wc_q27/*transcripts*")))])
    df["judgment"] = df.judgment.apply(_struct)
    df["messages"] = df.messages.apply(_struct)
    df["wc_match"] = df.judgment.apply(lambda j: j.get("match"))
    return df.reset_index(drop=True)


def load_rubrics():  # code/common.py:52-54
    r = pd.read_parquet(DATA / "rubrics.parquet")
    return {row.behavior_id: _struct(row.transcript_rubric)["text"] for row in r.itertuples()}


def make_split(prompts):  # code/common.py:57-72
    h = lambda s: hashlib.sha256(f"wcn-split-v1|{s}".encode()).hexdigest()  # noqa: E731
    out = {}
    for b, g in prompts.groupby("behavior_id"):
        pats = sorted(g.pattern_id.unique(), key=h)
        if len(pats) >= 4:
            side = {p: ("discovery" if i % 2 == 0 else "heldout") for i, p in enumerate(pats)}
            for r in g.itertuples():
                out[r.prompt_id] = {"split": side[r.pattern_id], "unit": "pattern"}
        else:
            for i, pid in enumerate(sorted(g.prompt_id, key=h)):
                out[pid] = {"split": "discovery" if i % 2 == 0 else "heldout", "unit": "prompt"}
    return out


def score_mc(text, gold):  # code/sidefx.py:70-72
    m = re.match(r"\W*([ABCD])\b", text.strip())
    return m is not None and m.group(1) == gold


def score_gsm(text, gold):  # code/sidefx.py:75-82
    m = re.findall(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)", text) or re.findall(r"(-?[\d,]*\.?\d+)", text)
    if not m:
        return False
    try:
        return abs(float(m[-1].replace(",", "")) - float(gold)) < 1e-6
    except ValueError:
        return False


def main():
    log(f"STEP 4 (data gate) start {time.ctime()} on {os.uname().nodename}. Log: {LOG}")

    # 1. Files from the repo, unchanged and unpacked
    split_sha = hashlib.sha256((DATA / "split.json").read_bytes()).hexdigest()
    check("data/split.json is byte-identical to the one at commit 071b01c",
          split_sha == "8a3c2451e0103f3858d609824ac5498bda72984295a73ef760f4f053b789b936", split_sha[:12])
    for f, n in {"forks_samples_off": 10371, "rate_gen": 3900, "sh_gen": 2468, "stage_a_gen": 403, "up_gen": 4133}.items():
        plain = sum(1 for line in open(RES / f"{f}.jsonl") if line.strip())
        gz = sum(1 for line in gzip.open(RES / f"{f}.jsonl.gz", "rt") if line.strip())
        check(f"results/{f}.jsonl unpacked whole", plain == gz == n, f"{plain} lines, .gz {gz}, expected {n}")

    # 2. WeirdChat prompts load as the repo's code loads them
    pr = load_prompts()
    files = sorted(glob.glob(str(DATA / "wc_q27/*prompts*")))
    check("11 prompt files found by the repo's pattern wc_q27/*prompts*", len(files) == 11, f"{len(files)} files")
    check("403 prompts, one row each", len(pr) == 403 and pr.prompt_id.is_unique, f"{len(pr)} rows")
    split = json.loads((DATA / "split.json").read_text())
    check("the 403 prompt ids are exactly the ids in data/split.json", set(pr.prompt_id) == set(split),
          f"ours {len(set(pr.prompt_id))}, split {len(split)}, shared {len(set(pr.prompt_id) & set(split))}")
    check("rebuilding the split from our files gives data/split.json exactly", make_split(pr) == split)

    # 3. The same prompts, behaviors, patterns and split as the original Stage A run (results/stage_a_gen.jsonl)
    gen = store(RES / "stage_a_gen.jsonl")
    pri = pr.set_index("prompt_id")
    bad = [k for k, g in gen.items() if k not in pri.index or g["behavior_id"] != pri.loc[k, "behavior_id"]
           or g["pattern_id"] != pri.loc[k, "pattern_id"] or g["split"] != split[k]["split"]]
    check("behavior, pattern and split match the original Stage A records for all 403 prompts",
          len(gen) == 403 and not bad, f"{len(gen)} records, {len(bad)} differ")

    # 4. Tokenization: our tokenizer and chat template give exactly the original token ids (code/common.py:93-98)
    from transformers import AutoTokenizer
    snap = json.loads((BASE / "paths.json").read_text())["qwen"]
    tok = AutoTokenizer.from_pretrained(snap)
    diff = []
    for pid, g in gen.items():
        ids = tok.apply_chat_template([{"role": "user", "content": pri.loc[pid, "user_text"]}],
                                      add_generation_prompt=True, enable_thinking=False, tokenize=True)
        if isinstance(ids, dict) or hasattr(ids, "keys"):
            ids = ids["input_ids"]
        if list(ids) != list(g["prompt_ids"]):
            diff.append(pid)
    check("prompt token ids equal the stored ones for all 403 prompts", not diff, f"{len(diff)} differ")

    # 5. Transcripts and rubrics load as the repo's code loads them
    tr = load_transcripts()
    check("transcripts load; every transcript's prompt is one of the 403", set(tr.prompt_id) <= set(pr.prompt_id),
          f"{len(tr)} transcripts, match known for {tr.wc_match.notna().mean():.3f}")
    rub = load_rubrics()
    check("rubrics load for all 11 behaviors, none empty",
          set(pr.behavior_id) <= set(rub) and all(len(rub[b]) > 50 for b in set(pr.behavior_id)), f"{len(rub)} rubrics")

    # 6. Our 38 test prompts: the repo's rate3.py selection (code/rate3.py:52-73), checked against the original runs
    J = store(RES / "stage_a_judge_on.jsonl")
    m = defaultdict(list)
    for j in J.values():
        if j["match"] is not None:
            m[j["prompt_id"]].append(j["match"])
    byb = defaultdict(lambda: {"hi": [], "lo": []})
    for pid, v in sorted(m.items()):
        if split[pid]["split"] == "discovery":
            continue
        byb[pri.loc[pid, "behavior_id"]]["hi" if np.mean(v) >= 0.1 else "lo"].append(pid)
    rng = np.random.default_rng(11)
    hi = []
    for b in sorted(byb):
        for mode in ("hi", "lo"):
            ps = byb[b][mode]
            pick = [ps[k] for k in sorted(rng.permutation(len(ps))[:12].tolist())]
            if mode == "hi":
                hi += pick
    orig_hi = {k.split("|")[2] for k in store(RES / "sh_judge.jsonl") if k.startswith("base|hi|")}
    check("rebuilt held-out selection equals the prompts in the original base runs (results/sh_judge.jsonl)",
          set(hi) == orig_hi, f"rebuilt {len(hi)}, original {len(orig_hi)}")
    test = [p for p in hi if pri.loc[p, "behavior_id"] in FOUR]
    cnt = {b: sum(pri.loc[p, "behavior_id"] == b for p in test) for b in FOUR}
    check("our test set: 38 prompts (code execution 12, 911 4, laser 12, embodiment 10)",
          [cnt[b] for b in FOUR] == [12, 4, 12, 10], str(cnt))

    # 7. Our validation pool: discovery prompts with Stage A rate >= 0.1 and no Phase 2 fork
    forks = [json.loads(line) for line in (RES / "phase2_forks.jsonl").read_text().splitlines() if line.strip()]
    fork_p = {f["prompt_id"] for f in forks}
    val = [pid for pid, v in sorted(m.items()) if split[pid]["split"] == "discovery" and np.mean(v) >= 0.1
           and pid not in fork_p and pri.loc[pid, "behavior_id"] in FOUR]
    vc = {b: sum(pri.loc[p, "behavior_id"] == b for p in val) for b in FOUR}
    check("forks file: 46 forks from 42 prompts", len(forks) == 46 and len(fork_p) == 42, f"{len(forks)} / {len(fork_p)}")
    check("validation pool: 44 prompts (code execution 7, 911 7, laser 15, embodiment 15)",
          [vc[b] for b in FOUR] == [7, 7, 15, 15], str(vc))

    # 8. Benchmarks: our MMLU and GSM8K copies + the repo's sampling (code/sidefx.py:36-44) must reproduce the original base
    #    accuracies from the stored answers: MMLU 0.843 = 337/400 and GSM8K 0.964 = 241/250 (logs/sidefx.log:32).
    #    A wrong file or row order would give about 25% on MMLU and near 0% on GSM8K.
    gs = store(RES / "sfx_gen.jsonl")
    mm = pd.read_parquet(DATA / "mmlu/test.parquet")
    g8 = pd.read_parquet(DATA / "gsm8k/test-00000-of-00001.parquet")
    check("MMLU test has 14,042 rows and GSM8K test 1,319", len(mm) == 14042 and len(g8) == 1319, f"{len(mm)}, {len(g8)}")
    ms = mm.sample(400, random_state=0)
    gold = ["ABCD"[r.answer] for r in ms.itertuples()]
    v = [score_mc(gs[f"base|mmlu|m{i}"]["samples"][0], gold[i]) for i in range(400) if f"base|mmlu|m{i}" in gs]
    check("MMLU: the stored answers score 337/400 against our copy", len(v) == 400 and sum(v) == 337, f"{sum(v)}/{len(v)}")
    gsm = g8.sample(250, random_state=0)
    gold = [r.answer.split("####")[-1].strip().replace(",", "") for r in gsm.itertuples()]
    v = [score_gsm(gs[f"base|gsm|g{i}"]["samples"][0], gold[i]) for i in range(250) if f"base|gsm|g{i}" in gs]
    check("GSM8K: the stored answers score 241/250 against our copy", len(v) == 250 and sum(v) == 241, f"{sum(v)}/{len(v)}")

    # 9. The repo's own loader reads our files the same way (imports code/common.py; reads only)
    sys.dont_write_bytecode = True      # reads only: no __pycache__ in the repo's code folder
    sys.path.insert(0, str(P / "code"))
    import common
    hp = common.load_prompts()
    check("the repo's common.load_prompts() gives the same prompts in the same order",
          hp.prompt_id.tolist() == pr.prompt_id.tolist() and hp.user_text.tolist() == pr.user_text.tolist())
    check("the repo's common.get_split() returns data/split.json unchanged",
          common.get_split() == split and hashlib.sha256((DATA / "split.json").read_bytes()).hexdigest() == split_sha)
    check("the repo's common.load_transcripts() and load_rubrics() load", len(common.load_transcripts()) == len(tr)
          and common.load_rubrics() == rub)

    n_fail = sum(not ok for _, ok in RESULTS)
    log(f"\n{len(RESULTS) - n_fail} of {len(RESULTS)} checks passed.")
    log("DATA GATE PASS" if n_fail == 0 else "DATA GATE FAIL: do not go on until every check passes.")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
