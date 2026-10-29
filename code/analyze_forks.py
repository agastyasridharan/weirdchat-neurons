"""Summarize Stage B fork search and validate the fork-token value metric.

Fork-token value: at fork prefix t*, V = sum_v p(v) q_v over candidate next tokens (renormalised over
the candidates' sampling mass), where q_v = P(behavior | prefix, v). If the fork is a next-token
decision, V should reproduce the directly measured P(t*), and the spread of q_v shows how sharp it is.
The differentiable metric for Phase 2 is M = logit(V) with q fixed and p from the model.

  python analyze_forks.py {on|off}
"""
import json
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr

from common import RES, JsonlStore

tag = sys.argv[1]
F = list(JsonlStore(RES / f"forks_{tag}.jsonl").done.values())
print(f"{len(F)} rollouts, {len({f['prompt_id'] for f in F})} prompts")
# P(L): the search assumed 1 (one Stage A positive); full_judge.py measures it by re-judging the full response
PL = {}
if (RES / f"full_{tag}.jsonl").exists():
    for r in JsonlStore(RES / f"full_{tag}.jsonl").done.values():
        if r["match"] is not None:
            PL.setdefault(r["rid"], []).append(r["match"])
    PL = {k: float(np.mean(v)) for k, v in PL.items()}
    pl = np.array([PL[f["rid"]] for f in F if f["rid"] in PL])
    print(f"P(L) measured for {len(pl)} rollouts: median {np.median(pl):.2f}; P(L) < 0.5: {np.sum(pl < .5)}; "
          f"P(L) < 0.25: {np.sum(pl < .25)}")
    for f in F:
        if f["rid"] in PL and f["fork"][1] >= f["L"]:
            f["fork_confirm"][2] = PL[f["rid"]]
            for x in f["next_tokens"]:
                if x["tok"] == f["fork_token"] and f["fork"][1] == f["fork"][0] + 1:
                    x["q"] = PL[f["rid"]]
    if "--stable" in sys.argv:  # keep rollouts whose full response is judged positive at least half the time
        F = [f for f in F if PL.get(f["rid"], 1.0) >= 0.5]
        print(f"--stable: {len(F)} rollouts with P(L) >= 0.5")
lg = lambda p: np.log(np.clip(p, 0.02, 0.98) / (1 - np.clip(p, 0.02, 0.98)))

rows = []
for f in F:
    lo, hi, Plo, Phi = f["fork"]
    nt = f["next_tokens"]
    mass = sum(x["p"] for x in nt)
    V = sum(x["p"] * x["q"] for x in nt) / mass if mass > 0 else np.nan
    act = [x for x in nt if x["tok"] == f["fork_token"]][0]
    alt = [x for x in nt if x["tok"] != f["fork_token"]]
    alt_q = sum(x["p"] * x["q"] for x in alt) / max(sum(x["p"] for x in alt), 1e-9) if alt else np.nan
    Plo_c, _, Phi_c, _ = f["fork_confirm"]
    rows.append(dict(b=f["behavior_id"], L=f["L"], lo=lo, rel=lo / max(f["L"], 1), base=f["base_p"], Plo=Plo_c, Phi=Phi_c,
                     jump=Phi_c - Plo_c, jump_sel=Phi - Plo, seg_jump=f["seg"][3] - f["seg"][2], V=V, mass=mass, p_act=act["p"],
                     q_act=act["q"], alt_q=alt_q, n_samples=sum(f["n_obs"].values())))

def show(name, xs):
    xs = np.array(xs, float)
    xs = xs[~np.isnan(xs)]
    print(f"  {name:34s} median {np.median(xs):6.3f}  IQR [{np.quantile(xs, .25):.3f}, {np.quantile(xs, .75):.3f}]")

print("\nWhere and how sharply the behavior is decided")
show("token jump, search estimates (biased)", [r["jump_sel"] for r in rows])
show("token jump, fresh confirmation", [r["jump"] for r in rows])
show("segment-level jump", [r["seg_jump"] for r in rows])
show("fork position (fraction of response)", [r["rel"] for r in rows])
show("fork position (tokens)", [r["lo"] for r in rows])
show("P(behavior) at empty prefix (base)", [r["base"] for r in rows])
show("P just before fork (fresh)", [r["Plo"] for r in rows])
show("p(actual fork token)", [r["p_act"] for r in rows])
show("q(actual token)", [r["q_act"] for r in rows])
show("q(alternative tokens), p-weighted", [r["alt_q"] for r in rows])
show("candidate nucleus mass covered", [r["mass"] for r in rows])
show("continuations sampled per rollout", [r["n_samples"] for r in rows])
sharp = [r for r in rows if r["jump"] >= 0.5]
print(f"  forks at the last token (hi == L): {sum(f['fork'][1] >= f['L'] for f in F)}")
print(f"  sharp single-token forks (jump >= 0.5): {len(sharp)}/{len(rows)}")
print(f"  forks within first 5 tokens: {sum(r['lo'] < 5 for r in rows)}/{len(rows)}")

print("\nFork-token value metric: V (from next-token decomposition) vs directly measured P(t*)")
ok = [r for r in rows if not np.isnan(r["V"])]
V = np.array([r["V"] for r in ok]); D = np.array([r["Plo"] for r in ok])
print(f"  n={len(ok)} pearson(logit) {pearsonr(lg(V), lg(D))[0]:.3f}  spearman {spearmanr(V, D)[0]:.3f}  "
      f"mean |V-P| {np.mean(np.abs(V - D)):.3f}  mean(V-P) {np.mean(V - D):+.3f}")
# expected sampling noise of |V-P| under the null that V == P: binomial SE of each estimate
print("\nPer behavior")
print(f"  {'behavior':30s} {'n':>3s} {'base':>5s} {'Plo':>5s} {'Phi':>5s} {'rel':>5s} {'sharp':>5s}")
for b in sorted({r["b"] for r in rows}):
    xs = [r for r in rows if r["b"] == b]
    print(f"  {b:30s} {len(xs):3d} {np.median([r['base'] for r in xs]):5.2f} {np.median([r['Plo'] for r in xs]):5.2f} "
          f"{np.median([r['Phi'] for r in xs]):5.2f} {np.median([r['rel'] for r in xs]):5.2f} {sum(r['jump'] >= .5 for r in xs):5d}")

# noise ceiling: if V were exactly P(t*), how far apart would the two estimates be from binomial sampling
# alone? Parametric bootstrap with each q_v and P(t*) := V resampled at their actual sample sizes.
rng = np.random.default_rng(0)
sims_mad, sims_r = [], []
for _ in range(500):
    Vs, Ps = [], []
    for f in F:
        nt = f["next_tokens"]
        mass = sum(x["p"] for x in nt)
        if mass <= 0 or f["fork_confirm"][1] is None:
            continue
        V0 = sum(x["p"] * x["q"] for x in nt) / mass
        qs = [rng.binomial(x["n"], x["q"]) / x["n"] if x["n"] else x["q"] for x in nt]
        Vs.append(sum(x["p"] * q for x, q in zip(nt, qs)) / mass)
        Ps.append(rng.binomial(f["fork_confirm"][1], V0) / f["fork_confirm"][1])
    Vs, Ps = np.array(Vs), np.array(Ps)
    sims_mad.append(np.mean(np.abs(Vs - Ps)))
    sims_r.append(pearsonr(lg(Vs), lg(Ps))[0])
print(f"  noise ceiling if V == P(t*): mean |V-P| {np.mean(sims_mad):.3f} [95% {np.quantile(sims_mad, .025):.3f}, "
      f"{np.quantile(sims_mad, .975):.3f}]  pearson(logit) {np.mean(sims_r):.3f} [95% {np.quantile(sims_r, .025):.3f}, "
      f"{np.quantile(sims_r, .975):.3f}]")

if "--examples" in sys.argv:
    print("\nSharpest forks per behavior (fresh jump)")
    for b in sorted({f["behavior_id"] for f in F}):
        fs = sorted([f for f in F if f["behavior_id"] == b], key=lambda f: -(f["fork_confirm"][2] - f["fork_confirm"][0]))[:3]
        print(f"\n## {b}")
        for f in fs:
            alts = ", ".join(f"{x['str']!r} p={x['p']:.2f} q={x['q']:.2f}" for x in sorted(f["next_tokens"], key=lambda x: -x["p"]))
            print(f"  [{f['fork'][0]}/{f['L']}] P {f['fork_confirm'][0]:.2f}->{f['fork_confirm'][2]:.2f}  ...{f['prefix_tail'][-90:]!r} || {alts}")
