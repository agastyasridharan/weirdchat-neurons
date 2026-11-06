"""Shared code for GPU jobs 4a and 4b (Step 3: choose the neurons). Runs in the attribution env.

The repo's code is imported as GPU job 2 did (gpu2_attrib_run.py): code/common.py and code/attrib.py from the repo, with the repo's
hard-coded model path pointed at our download before attrib.py copies it. Nothing is written inside the repo. Results
are read only: results/phase2_forks.jsonl and stage_a_gen.jsonl, and the results/attrib/ files that GPU job 2 made.
Every output goes to ~/scratch/wcn/methods/: first into a folder for this job, and into methods/ itself only if every
check of the job passed.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

WCN_BASE = Path(os.environ["WCN_BASE"])
WCN_ROOT = Path(os.environ["WCN_ROOT"])
SETUP = Path(__file__).resolve().parent
sys.path.insert(1, str(WCN_ROOT / "code"))  # after this folder (ours), before site-packages
import common  # noqa: E402  (code/common.py)

common.QWEN_SNAP = json.loads((WCN_BASE / "paths.json").read_text())["qwen"]
import attrib  # noqa: E402  (code/attrib.py; copies QWEN_SNAP from common at import, line 21)
import sets as repo_sets  # noqa: E402  (code/sets.py: spec() and B5; its main() is not run)

for mod in (common, attrib, repo_sets):
    assert Path(mod.__file__).resolve().parent == (WCN_ROOT / "code").resolve(), f"wrong module: {mod.__file__}"
assert attrib.QWEN_SNAP == common.QWEN_SNAP, "model path was not passed to the repo's attrib.py"

NL, NI, QCLAMP = attrib.NL, attrib.NI, attrib.QCLAMP
RES = common.RES
ATTRIB = RES / "attrib"
METHODS = WCN_BASE / "methods"
# The size ladder's four behaviors and its sizes. Denying being an AI is left out: its sets don't carry over to
# held-out prompts (PHASE2.md:86-92).
LADDER = ["claims-called-911", "fabricated-code-execution", "false-physical-embodiment", "laser-at-aircraft"]
SIZES = [3, 5, 10, 20, 50, 100]
LAYER_CUT = 48  # M2: layers 0-47 only
# Fork counts per behavior in results/phase2_forks.jsonl (46 forks from 42 prompts).
FORKS_PER_B = {"fabricated-code-execution": 13, "laser-at-aircraft": 8, "false-physical-embodiment": 5,
               "claims-called-911": 3}
N_FORKS = 46

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append({"check": name, "pass": bool(ok), "detail": detail})
    print(f"{'PASS' if ok else 'FAIL'}  {name}  [{detail}]", flush=True)
    return bool(ok)


def info(msg):
    print(f"info  {msg}", flush=True)


def stage(msg):
    print(f"\n=== {msg}  ({time.strftime('%H:%M:%S')})", flush=True)


def read_jsonl_last(path):
    """The reading rule of the repo's JsonlStore (code/common.py:248-253): one record per 'key', the last one wins. Read only:
    unlike JsonlStore, this does not open the repo's file for appending."""
    done = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                done[d["key"]] = d
    return done


def load_forks():
    F = attrib.forks()
    gen = read_jsonl_last(RES / "stage_a_gen.jsonl")
    ids = [attrib.prefix_ids(f, gen) for f in F]
    by_b = {}
    for i, f in enumerate(F):
        by_b.setdefault(f["behavior_id"], []).append(i)
    ok = len(F) == N_FORKS and all(len(by_b.get(b, [])) == n for b, n in FORKS_PER_B.items())
    check("inputs: 46 forks, with 13, 8, 5 and 3 for code execution, laser, false embodiment and 911", ok,
          ", ".join(f"{b} {len(v)}" for b, v in sorted(by_b.items())))
    lens = [len(x) for x in ids]
    info(f"prefix lengths {min(lens)} to {max(lens)} tokens")
    return F, ids, by_b


def load_attrib(F):
    """GPU job 2's attribution files: A (summed over positions), absa, V0, M0, key, for all 46 forks."""
    Z, bad = [], []
    for i, f in enumerate(F):
        z = np.load(ATTRIB / f"{i}.npz")
        r = {k: z[k] for k in ("A", "absa", "V0", "M0", "key")}
        if str(r["key"]) != f["key"] or r["A"].shape != (NL, NI) or r["absa"].shape != (NL, NI) or not np.isfinite(r["A"]).all():
            bad.append(i)
        Z.append(r)
    check("inputs: GPU job 2's attribution files match the 46 forks (key, shape, finite)", not bad, f"bad forks {bad}")
    return Z


def normalizer(A):
    """sets.py:40-41: a fork's scores are divided by the sum of its 100 largest attributions."""
    a = np.asarray(A).ravel().astype(np.float64)
    return np.sort(a)[::-1][:100].sum()


def behavior_score(As, norms):
    """sets.py:37-41: score(n) = sum over the behavior's forks of A_f(n) / s_f."""
    score = np.zeros(NL * NI)
    for A, s in zip(As, norms):
        score += np.asarray(A).ravel().astype(np.float64) / s
    return score


def top_order(score, k=100, exclude=None, layer_cut=None):
    """sets.py:43: order = argsort(-score), optionally without excluded neurons or layers at or above layer_cut."""
    s = np.array(score, dtype=np.float64)
    if exclude is not None and len(exclude):
        s[np.asarray(exclude, dtype=np.int64)] = -np.inf
    if layer_cut is not None:
        s[layer_cut * NI:] = -np.inf
    order = np.argsort(-s)[:k]
    if not np.isfinite(s[order]).all():
        raise RuntimeError("fewer than k neurons left to choose from")
    return [int(x) for x in order]


def mean_absa(Z, ix):
    """sets.py:37, 42: the behavior's mean sum|a| over its forks, for the matched controls."""
    absa = np.zeros((NL, NI))
    for i in ix:
        absa += Z[i]["absa"] / len(ix)
    return absa


def control(sel, absa, seed=1):
    """sets.py:44, 48: the repo's attrib.matched (code/attrib.py:100-115) with np.random.default_rng(1)."""
    return [int(x) for x in attrib.matched(list(sel), absa, np.random.default_rng(seed))]


def check_control(name, sel, ctrl):
    ok = (len(ctrl) == len(sel) == len(set(ctrl)) and not set(ctrl) & set(sel)
          and sorted(x // NI for x in ctrl) == sorted(x // NI for x in sel) and all(0 <= x < NL * NI for x in ctrl))
    return check(f"control {name}: {len(ctrl)} distinct neurons, none in the set, same layers one for one", ok,
                 f"{len(ctrl)} neurons, {len(set(ctrl) & set(sel))} shared with the set")


def spec(flat):
    return repo_sets.spec(flat)  # sets.py:25-29, the format the ablation extension reads


def flat_of(spec_):
    return {int(l) * NI + int(i) for l, v in spec_.items() for i in v}


def device_of(M):
    return next(M.m.parameters()).device


def fork_tq(M, f):
    """attrib.py:83-84."""
    import torch
    dev = device_of(M)
    toks = torch.tensor([x["tok"] for x in f["next_tokens"]], device=dev)
    q = torch.tensor([x["q"] for x in f["next_tokens"]], device=dev).clamp(*QCLAMP)
    return toks, q


def attrib_at(M, f, ids, flat=()):
    """The repo's attrib_one (attrib.py:82-97), except that the neurons in `flat` stay zeroed during the gradient pass,
    as the repo's model hook allows (attrib.py:50-60); attrib_one always clears the ablation first (line 85).
    Returns A (sum over positions), M, V and the number of layers whose activations were seen."""
    toks, q = fork_tq(M, f)
    M.set_ablation(list(flat))
    try:
        m, V, p = M.metric(ids, toks, q, grad=True)
        m.backward()
        A = np.zeros((NL, NI), np.float32)
        for l, a in M.acts.items():
            A[l] = (a.detach().float() * a.grad.float())[0].sum(0).cpu().numpy()
        n_layers = len(M.acts)
        out = (A, float(m.detach()), float(V.detach()), n_layers)
    finally:
        M.acts = {}
        M.set_ablation([])
    return out


def V_at(M, f, ids, flat=()):
    """V with the neurons in `flat` zeroed at all positions, no gradient: the run() in the repo's verify_one (attrib.py:124-127)."""
    toks, q = fork_tq(M, f)
    M.set_ablation(list(flat))
    try:
        m, V, p = M.metric(ids, toks, q)
        out = float(V)
    finally:
        M.set_ablation([])
    return out


def check_model(M):
    cfg = M.m.config.get_text_config()  # the text part if the config is the full multimodal one
    types = list(cfg.layer_types)
    shapes = {tuple(d.weight.shape) for d in M.downs}
    return check("model: layers, neurons per layer and layer types as expected",
                 cfg.num_hidden_layers == NL and cfg.intermediate_size == NI and len(M.downs) == NL
                 and types.count("linear_attention") == NL * 3 // 4 and types.count("full_attention") == NL // 4
                 and shapes == {(cfg.hidden_size, NI)},
                 f"{cfg.num_hidden_layers} layers, {cfg.intermediate_size} neurons each, "
                 f"{types.count('linear_attention')} DeltaNet and {types.count('full_attention')} attention, down_proj {shapes}")


def _plain(o):
    """numpy numbers and booleans as plain Python values, so a summary never fails to save at the end of a run."""
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"cannot save {type(o).__name__}")


class Outputs:
    """Writes into methods/tmp_<job>/ and moves the files into methods/ only if every check passed."""

    def __init__(self, tag, names):
        METHODS.mkdir(exist_ok=True)
        self.names = names
        there = [n for n in names if (METHODS / n).exists()]
        if there:
            print(f"STOP: {', '.join(there)} already in {METHODS}. To run again, first move those away.", flush=True)
            sys.exit(1)
        self.tmp = METHODS / f"tmp_{tag}_{os.environ.get('SLURM_JOB_ID', 'local')}"
        self.tmp.mkdir()

    def path(self, name):
        return self.tmp / name

    def write_json(self, name, obj):
        (self.tmp / name).write_text(json.dumps(obj, indent=1, default=_plain))

    def finish(self, all_ok):
        if not all_ok:
            print(f"\nNot every check passed, so the outputs stay in {self.tmp}", flush=True)
            return
        for n in self.names:
            os.replace(self.tmp / n, METHODS / n)
        self.tmp.rmdir()
        print(f"\nOutputs moved to {METHODS}: {', '.join(self.names)}", flush=True)


def summary_lines():
    n_ok = sum(r["pass"] for r in RESULTS)
    return n_ok, len(RESULTS)
