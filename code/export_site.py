"""Collect every number shown on the Phase 1 results page into one JSON (results/site_phase1.json)."""
import json
import random
from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr, spearmanr

from common import RES, JsonlStore, load_prompts

out = {}
pr = load_prompts().set_index("prompt_id")
lg = lambda p: np.log(np.clip(p, 0.02, 0.98) / (1 - np.clip(p, 0.02, 0.98)))


def kappa(a, b):
    a, b = np.array(a, bool), np.array(b, bool)
    po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe))


# ---------- judge validation (stored WeirdChat transcripts)
jv = [r for r in JsonlStore(RES / "judge_val.jsonl").done.values() if r["match"] is not None]
by = defaultdict(dict)
for r in jv:
    by[r["transcript_id"]][r["cond"]] = r["match"]; by[r["transcript_id"]]["wc"] = r["wc_match"]
conds = {}
for c in ("on", "on2", "off", "rfoff"):
    xs = [(v["wc"], v[c]) for v in by.values() if c in v]
    w, m = np.array([a for a, _ in xs]), np.array([b for _, b in xs])
    conds[c] = dict(n=len(xs), agree=float((w == m).mean()), precision=float(m[w].sum() / max(m.sum(), 1) if False else (w & m).sum() / max(m.sum(), 1)),
                    recall=float((w & m).sum() / max(w.sum(), 1)), kappa_wc=kappa(w, m))
for c in ("on2", "off", "rfoff"):
    xs = [(v["on"], v[c]) for v in by.values() if c in v and "on" in v]
    conds[c]["kappa_on"] = kappa([a for a, _ in xs], [b for _, b in xs])
out["judge_val"] = conds

# ---------- parity (Runpod vs local)
pa = defaultdict(dict)
for r in JsonlStore(RES / "judge_parity.jsonl").done.values():
    pa[r["transcript_id"]][r["side"]] = r["match"]; pa[r["transcript_id"]]["orig"] = r["orig"]
rows = [x for x in pa.values() if x.get("local") is not None and x.get("remote") is not None]
a = lambda k1, k2: float(np.mean([x[k1] == x[k2] for x in rows]))
out["parity"] = dict(n=len(rows), local_orig=a("local", "orig"), remote_orig=a("remote", "orig"), local_remote=a("local", "remote"),
                     pos_orig=float(np.mean([x["orig"] for x in rows])), pos_local=float(np.mean([x["local"] for x in rows])),
                     pos_remote=float(np.mean([x["remote"] for x in rows])))

# ---------- Stage A rates: off (64/prompt), on (16/prompt, same samples), WeirdChat
on = JsonlStore(RES / "stage_a_judge_on.jsonl").done
off = JsonlStore(RES / "stage_a_judge_off.jsonl").done
gen = JsonlStore(RES / "stage_a_gen.jsonl").done
per = defaultdict(list)
for k, r in on.items():
    if r["match"] is None or k not in off or off[k]["match"] is None:
        continue
    per[r["prompt_id"]].append((off[k]["match"], r["match"]))
both = np.array([x for v in per.values() for x in v])
off64 = defaultdict(list)
for r in off.values():
    if r["match"] is not None:
        off64[r["prompt_id"]].append(r["match"])
prompts = []
for pid in per:
    prompts.append(dict(pid=pid, b=pr.loc[pid, "behavior_id"], split=gen[pid]["split"],
                        off16=float(np.mean([x for x, _ in per[pid]])), on16=float(np.mean([y for _, y in per[pid]])),
                        off64=float(np.mean(off64[pid])), k64=int(sum(off64[pid])), wc=float(pr.loc[pid, "wc_k"] / pr.loc[pid, "wc_n"])))
beh = []
for b in sorted({p["b"] for p in prompts}):
    xs = [p for p in prompts if p["b"] == b]
    beh.append(dict(b=b, n=len(xs), off=float(np.mean([p["off16"] for p in xs])), on=float(np.mean([p["on16"] for p in xs])),
                    wc=float(np.mean([p["wc"] for p in xs])), off64=float(np.mean([p["off64"] for p in xs])),
                    pass_k2=sum(p["k64"] >= 2 for p in xs), disc=sum(p["split"] == "discovery" for p in xs),
                    disc_pass=sum(p["k64"] >= 2 and p["split"] == "discovery" for p in xs)))
r_on = np.array([p["on16"] for p in prompts]); r_off = np.array([p["off16"] for p in prompts]); wc = np.array([p["wc"] for p in prompts])
out["stage_a"] = dict(
    n_prompts=len(prompts), n_samples_on=len(both), n_disc=sum(p["split"] == "discovery" for p in prompts),
    pooled_on=float(r_on.mean()), pooled_off=float(r_off.mean()), pooled_off64=float(np.mean([p["off64"] for p in prompts])),
    pooled_wc=float(wc.mean()), sp_on_wc=float(spearmanr(r_on, wc)[0]), sp_off_wc=float(spearmanr(r_off, wc)[0]),
    sp_on_off=float(spearmanr(r_on, r_off)[0]),
    agree_on_off=float((both[:, 0] == both[:, 1]).mean()), kappa_on_off=kappa(both[:, 0], both[:, 1]),
    p_on_given_off=float(both[both[:, 0] == 1, 1].mean()), p_off_given_on=float(both[both[:, 1] == 1, 0].mean()),
    zero_rate=float(np.mean([p["k64"] == 0 for p in prompts])), pass_k2=float(np.mean([p["k64"] >= 2 for p in prompts])),
    behaviors=beh)

# ---------- forks
F = list(JsonlStore(RES / "forks_off.jsonl").done.values())
PL = defaultdict(list)
for r in JsonlStore(RES / "full_off.jsonl").done.values():
    if r["match"] is not None:
        PL[r["rid"]].append(r["match"])
PL = {k: float(np.mean(v)) for k, v in PL.items()}
rows = []
NT = {}
for f in F:
    lo, hi, Plo_s, Phi_s = f["fork"]
    Plo, nlo, Phi, nhi = f["fork_confirm"]
    if hi >= f["L"]:
        Phi = PL[f["rid"]]
    nt = [dict(x) for x in f["next_tokens"]]
    for x in nt:
        if x["tok"] == f["fork_token"] and hi == lo + 1 and hi >= f["L"]:
            x["q"] = PL[f["rid"]]
    mass = sum(x["p"] for x in nt)
    V = sum(x["p"] * x["q"] for x in nt) / mass
    act = [x for x in nt if x["tok"] == f["fork_token"]][0]
    alt = [x for x in nt if x["tok"] != f["fork_token"]]
    NT[f["rid"]] = nt
    rows.append(dict(rid=f["rid"], b=f["behavior_id"], L=f["L"], lo=lo, rel=lo / f["L"], base=f["base_p"], PL=PL[f["rid"]],
                     Plo=Plo, Phi=Phi, jump=Phi - Plo, jump_sel=Phi_s - Plo_s, seg_jump=f["seg"][3] - f["seg"][2],
                     V=V, mass=mass, p_act=act["p"], q_act=act["q"],
                     alt_q=(sum(x["p"] * x["q"] for x in alt) / sum(x["p"] for x in alt)) if alt and sum(x["p"] for x in alt) > 0 else None,
                     n=sum(f["n_obs"].values()), at_end=hi >= f["L"]))
J = np.array([r["jump"] for r in rows])
Vs = np.array([r["V"] for r in rows]); Ps = np.array([r["Plo"] for r in rows])
# noise ceiling
rng = np.random.default_rng(0); mads, rs = [], []
for _ in range(500):
    v_, p_ = [], []
    for f, r in zip(F, rows):  # same corrected next-token values as the observed V; P side at its own n
        nt = NT[f["rid"]]; mass = sum(x["p"] for x in nt)
        qs = [rng.binomial(x["n"], x["q"]) / x["n"] if x["n"] else x["q"] for x in nt]
        v_.append(sum(x["p"] * q for x, q in zip(nt, qs)) / mass)
        n = f["fork_confirm"][1] or 64
        p_.append(rng.binomial(n, r["V"]) / n)
    v_, p_ = np.array(v_), np.array(p_)
    mads.append(np.mean(np.abs(v_ - p_))); rs.append(pearsonr(lg(v_), lg(p_))[0])
q = lambda x, a: float(np.quantile(x, a))
summ = lambda x: dict(med=float(np.median(x)), q1=q(x, .25), q3=q(x, .75))
out["forks"] = dict(
    n=len(rows), n_prompts=len({r["rid"].split("|")[0] for r in rows}),
    jump=summ(J), jump_sel=summ([r["jump_sel"] for r in rows]), seg_jump=summ([r["seg_jump"] for r in rows]),
    rel=summ([r["rel"] for r in rows]), lo=summ([r["lo"] for r in rows]), base=summ([r["base"] for r in rows]),
    Plo=summ(Ps), p_act=summ([r["p_act"] for r in rows]), q_act=summ([r["q_act"] for r in rows]),
    alt_q=summ([r["alt_q"] for r in rows if r["alt_q"] is not None]), mass=summ([r["mass"] for r in rows]),
    n_samples=summ([r["n"] for r in rows]), n_samples_total=int(sum(r["n"] for r in rows)),
    sharp=int((J >= .5).sum()), first5=int(sum(r["lo"] < 5 for r in rows)), at_end=int(sum(r["at_end"] for r in rows)),
    PL_med=float(np.median([r["PL"] for r in rows])), PL_lt50=int(sum(r["PL"] < .5 for r in rows)), PL_lt25=int(sum(r["PL"] < .25 for r in rows)),
    V_pearson_logit=float(pearsonr(lg(Vs), lg(Ps))[0]), V_spearman=float(spearmanr(Vs, Ps)[0]),
    V_mad=float(np.mean(np.abs(Vs - Ps))), V_bias=float(np.mean(Vs - Ps)),
    ceil_mad=float(np.mean(mads)), ceil_mad_lo=q(mads, .025), ceil_mad_hi=q(mads, .975),
    ceil_r=float(np.mean(rs)), ceil_r_lo=q(rs, .025), ceil_r_hi=q(rs, .975),
    rows=[{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows])
per_b = []
for b in sorted({r["b"] for r in rows}):
    xs = [r for r in rows if r["b"] == b]
    per_b.append(dict(b=b, n=len(xs), base=float(np.median([r["base"] for r in xs])), Plo=float(np.median([r["Plo"] for r in xs])),
                      Phi=float(np.median([r["Phi"] for r in xs])), jump=float(np.median([r["jump"] for r in xs])),
                      seg=float(np.median([r["seg_jump"] for r in xs])), rel=float(np.median([r["rel"] for r in xs])),
                      sharp=sum(r["jump"] >= .5 for r in xs), first5=sum(r["lo"] < 5 for r in xs)))
out["forks"]["behaviors"] = per_b

# example forks: sharpest per behavior (fresh jump), with alternatives
samples = JsonlStore(RES / "forks_samples_off.jsonl").done
ex = []
rowd = {r["rid"]: r for r in rows}
for b in sorted({f["behavior_id"] for f in F}):
    fs = sorted([f for f in F if f["behavior_id"] == b and f["fork"][1] < f["L"]], key=lambda f: -rowd[f["rid"]]["jump"])[:2]
    for f in fs:
        r = rowd[f["rid"]]
        ex.append(dict(b=b, rid=f["rid"], lo=f["fork"][0], L=f["L"], Plo=r["Plo"], Phi=r["Phi"],
                       user=pr.loc[f["prompt_id"], "user_text"][:400], tail=f["prefix_tail"][-160:], act=f["fork_token_str"],
                       alts=[dict(s=x["str"], p=round(x["p"], 3), q=round(x["q"], 3), act=x["tok"] == f["fork_token"])
                             for x in sorted(f["next_tokens"], key=lambda x: -x["p"])]))
out["examples"] = ex
# example curves (search estimates) for 3 illustrative rollouts
pick = [e["rid"] for e in ex if e["b"] in ("denying-ai-identity", "claims-called-911", "fabricated-code-execution")][::2][:3]
out["curves"] = [dict(rid=f["rid"], b=f["behavior_id"], L=f["L"], fork=f["fork"][:2], curve=f["curve"], PL=PL[f["rid"]])
                 for f in F if f["rid"] in pick]

# ---------- thinking-on check of forks
R = [r for r in JsonlStore(RES / "verify_on.jsonl").done.values() if r["on"] is not None and r["off"] is not None]
fb = {f["rid"]: f["behavior_id"] for f in F}
sides = {}
for s in ("lo", "hi"):
    x = np.array([(r["off"], r["on"]) for r in R if r["side"] == s], float)
    sides[s] = dict(n=len(x), off=float(x[:, 0].mean()), on=float(x[:, 1].mean()), agree=float((x[:, 0] == x[:, 1]).mean()),
                    on_given_off=float(x[x[:, 0] == 1, 1].mean()), off_given_on=float(x[x[:, 1] == 1, 0].mean()))
d = defaultdict(lambda: {"lo": [], "hi": []})
for r in R:
    d[r["rid"]][r["side"]].append((r["off"], r["on"]))
jumps = []
for rid, v in d.items():
    if v["lo"] and v["hi"]:
        jumps.append(dict(rid=rid, b=fb[rid], off=float(np.mean([a for a, _ in v["hi"]]) - np.mean([a for a, _ in v["lo"]])),
                          on=float(np.mean([b for _, b in v["hi"]]) - np.mean([b for _, b in v["lo"]]))))
jo = np.array([j["off"] for j in jumps]); jn = np.array([j["on"] for j in jumps])
vb = defaultdict(list)
for r in R:
    vb[fb[r["rid"]]].append((r["off"], r["on"]))
out["verify"] = dict(n_forks_sampled=len(d), n_both=len(jumps), n_samples=len(R),
                     agree=float(np.mean([r["off"] == r["on"] for r in R])), sides=sides,
                     mean_off=float(jo.mean()), mean_on=float(jn.mean()), med_off=float(np.median(jo)), med_on=float(np.median(jn)),
                     corr=float(np.corrcoef(jo, jn)[0, 1]), sharp_off=int((jo >= .5).sum()), sharp_on=int((jn >= .5).sum()),
                     big_off=int((jo >= .25).sum()), big_on=int((jn >= .25).sum()), keep=int(((jo >= .25) & (jn >= .1)).sum()),
                     jumps=[{k: (round(v, 4) if isinstance(v, float) else v) for k, v in j.items()} for j in jumps],
                     behaviors=[dict(b=b, n=len(v), off=float(np.mean([a for a, _ in v])), on=float(np.mean([c for _, c in v])),
                                     agree=float(np.mean([a == c for a, c in v]))) for b, v in sorted(vb.items())])
(RES / "site_phase1.json").write_text(json.dumps(out))
print("wrote", RES / "site_phase1.json", len(json.dumps(out)) // 1024, "KB")
print(json.dumps({k: v for k, v in out["judge_val"].items()}, indent=0))
print(json.dumps({k: v for k, v in out["forks"].items() if k not in ("rows", "behaviors")}))
print(json.dumps({k: v for k, v in out["verify"].items() if k not in ("jumps", "behaviors")}))
print([ (c["b"], c["rid"]) for c in out["curves"]])
