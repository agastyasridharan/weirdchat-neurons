"""STEP 3: download the two models, the WeirdChat files, MMLU and GSM8K, each pinned to an exact revision,
check every file against Hugging Face's own record, and put the data files where the repo's code looks.

Where: a LOGIN node (it needs internet), after steps 1 and 2. About 118 GB into ~/scratch/wcn/hf;
roughly 10 to 40 minutes. Safe to run again: finished files are skipped. It writes only inside
~/scratch/wcn.

Pinned revisions:
- Qwen/Qwen3.6-27B and google/gemma-4-31B-it: the snapshots in code/serve.sh:10-11.
- Transluce/WeirdChat: commit 3460e414..., which is tag v1.0.1, the version the repo's README.md:82 names.
- cais/mmlu: commit c30699e8..., the one in code/sidefx.py:32.
- openai/gsm8k: commit cc7b047b..., the one in code/sidefx.py:33.
"""
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

if "WCN_BASE" not in os.environ:
    sys.exit("STOP: run 'source ~/scratch/wcn/setup/env.sh' first.")
if os.environ.get("SLURM_JOB_ID"):
    sys.exit("STOP: inside a Slurm job (no internet). Run this on a login node.")

from huggingface_hub import HfApi, hf_hub_download, snapshot_download  # noqa: E402

BASE = Path(os.environ["WCN_BASE"])
ROOT = Path(os.environ["WCN_ROOT"])
LOG = BASE / "logs" / f"step3_download_{time.strftime('%Y%m%d_%H%M%S')}.log"
_logf = open(LOG, "a")


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    _logf.write(s + "\n")
    _logf.flush()


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(msg):
    log("STOP:", msg)
    log("STEP 3 FAILED")
    sys.exit(1)


MODELS = {
    "qwen": ("Qwen/Qwen3.6-27B", "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"),
    "gemma": ("google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475"),
}
WC_REPO, WC_REV, WC_TAG = "Transluce/WeirdChat", "3460e41491308601cd68ee377011c7b7a8f0b4b3", "v1.0.1"
BEHAVIORS = ["chemtrails-assertion", "claims-called-911", "cutting-instructions", "denying-ai-identity",
             "extreme-calorie-restriction", "fabricated-code-execution", "false-physical-embodiment",
             "laser-at-aircraft", "purging-advice", "recommends-drunk-driving", "unsolicited-sexual-advances"]
# (repo, revision, file in the repo, where it goes under WCN_ROOT). The repo's code finds the WeirdChat files with
# the patterns data/wc_q27/*prompts* and data/wc_q27/*transcripts* (code/common.py:37, 45), so the names
# here contain those words; sorted, they keep the behaviors in alphabetical order.
DATA = [(WC_REPO, WC_REV, "data/patterns.parquet", "data/patterns.parquet"),
        (WC_REPO, WC_REV, "data/rubrics.parquet", "data/rubrics.parquet")]
DATA += [(WC_REPO, WC_REV, f"data/prompts/qwen__qwen3.6-27b/{b}.parquet", f"data/wc_q27/prompts-{b}.parquet")
         for b in BEHAVIORS]
DATA += [(WC_REPO, WC_REV, f"data/transcripts/qwen__qwen3.6-27b/{b}-0000.parquet",
          f"data/wc_q27/transcripts-{b}-0000.parquet") for b in BEHAVIORS]
DATA += [("cais/mmlu", "c30699e8356da336a370243923dbaf21066bb9fe", "all/test-00000-of-00001.parquet",
          "data/mmlu/test.parquet"),                                   # the path in code/sidefx.py:32
         ("openai/gsm8k", "cc7b047b6e5bb11b4f1af84efc572db110a51b3c", "main/test-00000-of-00001.parquet",
          "data/gsm8k/test-00000-of-00001.parquet")]


def main():
    log(f"STEP 3 start on {os.uname().nodename}. Log: {LOG}")
    log("Hugging Face token in use:", "yes" if os.environ.get("HF_TOKEN") else "no (public files; fine)")
    if not (ROOT / ".git").exists():
        fail(f"{ROOT} is not a git clone. Run step 2 first.")
    api = HfApi()

    # WeirdChat: the tag v1.0.1 must point at the pinned commit, and the Qwen3.6-27B files must be exactly the 22 we expect
    refs = api.list_repo_refs(WC_REPO, repo_type="dataset")
    tag = {t.name: t.target_commit for t in refs.tags}.get(WC_TAG)
    log(f"WeirdChat tag {WC_TAG} -> {tag}")
    if tag != WC_REV:
        fail(f"WeirdChat tag {WC_TAG} no longer points at {WC_REV}.")
    files = api.list_repo_files(WC_REPO, repo_type="dataset", revision=WC_REV)
    q27 = sorted(f for f in files if "/qwen__qwen3.6-27b/" in f)
    want = sorted(d[2] for d in DATA if "/qwen__qwen3.6-27b/" in d[2])
    if q27 != want:
        fail(f"WeirdChat's Qwen3.6-27B file list differs from the expected 22 files: {q27}")
    log("WeirdChat at v1.0.1 has exactly the 22 expected Qwen3.6-27B files")

    # Data files: download, compare with Hugging Face's SHA-256, copy into the repo's data/ folder
    for repo, rev, path, dest in DATA:
        info = api.get_paths_info(repo, [path], revision=rev, repo_type="dataset")[0]
        expect = info.lfs.sha256 if info.lfs else None
        src = Path(hf_hub_download(repo, path, repo_type="dataset", revision=rev))
        got = sha256(src)
        if expect is None:
            log(f"NOTE {repo}/{path}: Hugging Face gave no SHA-256 for this file; size checked instead")
            if info.size is not None and src.stat().st_size != info.size:
                fail(f"{repo}/{path}: size {src.stat().st_size} differs from Hugging Face's {info.size}")
        elif got != expect:
            fail(f"{repo}/{path}: SHA-256 {got} differs from Hugging Face's {expect}")
        out = ROOT / dest
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists() or sha256(out) != got:
            shutil.copyfile(src, out)
        if sha256(out) != got:
            fail(f"copy of {path} to {out} is not identical")
        log(f"OK  {dest:52s} {info.size:>10,d} bytes  sha256 {got}  ({repo}@{rev[:7]}:{path})")

    # Models: first confirm both pinned revisions are reachable (so a problem with the second shows up before
    # ~56 GB of the first is downloaded), then download each and compare every file's size with HF's record
    infos = {}
    for name, (repo, rev) in MODELS.items():
        infos[name] = api.model_info(repo, revision=rev, files_metadata=True)
        if infos[name].sha != rev:
            fail(f"{repo}: revision {rev} resolved to {infos[name].sha}")
    snaps = {}
    for name, (repo, rev) in MODELS.items():
        info = infos[name]
        total = sum(s.size or 0 for s in info.siblings)
        log(f"{repo}@{rev[:7]}: {len(info.siblings)} files, {total / 1e9:.1f} GB; downloading (resumes if interrupted)")
        t0 = time.time()
        snap = Path(snapshot_download(repo, revision=rev, max_workers=4))
        bad = []
        for s in info.siblings:
            p = snap / s.rfilename
            if not p.exists():
                bad.append(f"missing {s.rfilename}")
            elif s.size is not None and p.stat().st_size != s.size:
                bad.append(f"size {s.rfilename}: {p.stat().st_size} vs {s.size}")
        if bad:
            fail(f"{repo}: " + "; ".join(bad[:10]))
        hashed = [s for s in info.siblings if s.lfs is not None and (s.size or 0) < 50_000_000]
        for s in hashed:                               # SHA-256 of the small LFS files (tokenizer etc.)
            if sha256(snap / s.rfilename) != s.lfs.sha256:
                fail(f"{repo}: SHA-256 mismatch for {s.rfilename}")
        log(f"OK  {repo}: all {len(info.siblings)} files present with the right sizes; {len(hashed)} small LFS"
            f" files also SHA-256-checked; {time.time() - t0:.0f}s -> {snap}")
        snaps[name] = str(snap)

    (BASE / "paths.json").write_text(json.dumps(snaps, indent=1) + "\n")
    log(f"model folders saved in {BASE / 'paths.json'}: {snaps}")
    log("STEP 3 DONE")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:  # anything unexpected: put the full error in the log, then stop
        import traceback
        log(f"STOP: unexpected error: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        log("STEP 3 FAILED")
        sys.exit(1)
