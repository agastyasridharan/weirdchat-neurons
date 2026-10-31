"""Embed the trimmed Phase 1 export into template.html -> index.html."""
import ast, json, pathlib, re

import numpy as np
here = pathlib.Path(__file__).parent
d = json.loads((here.parent / "site_trim.json").read_text())
SHOW = {"denying-ai-identity", "claims-called-911", "false-physical-embodiment",
        "fabricated-code-execution", "recommends-drunk-driving", "chemtrails-assertion"}
# no raw prompts, ids, or reply text from behaviors we don't quote
d["examples"] = [{k: v for k, v in e.items() if k not in ("user", "rid")} for e in d["examples"] if e["b"] in SHOW]
for c in d["curves"]: c.pop("rid", None)
for j in d["verify"]["jumps"]: j.pop("rid", None)
# thinking-on re-scoring of candidate forks (rescore_on.py summary)
R = [json.loads(l) for l in (here.parent / "results/forks_on_rescored.jsonl").read_text().splitlines()]
med = lambda xs: float(sorted(xs)[len(xs) // 2]) if len(xs) % 2 else float(sum(sorted(xs)[len(xs) // 2 - 1:len(xs) // 2 + 1]) / 2)
sel = [r for r in R if r["jump_on"] >= .5 and r["PL_on"] >= .5 and r["fork"][1] < r["L"]]
beh = {}
for r in R:
    beh.setdefault(r["behavior_id"], []).append(r)
def p2(b):
    return sum(r in sel for r in beh[b])
def alt_q(r):
    o = [n for n in r["next_tokens"] if n["q"] is not None and n["tok"] != r["fork_token"]]
    return sum(n["p"] * n["q"] for n in o) / sum(n["p"] for n in o) if o and sum(n["p"] for n in o) > 0 else None
act = [[n for n in r["next_tokens"] if n["tok"] == r["fork_token"]][0] for r in sel]
alts = [a for a in map(alt_q, sel) if a is not None]
toks = {}
for r in sel:
    toks.setdefault(r["behavior_id"], []).append(r["fork_token_str"])
d["rescore"] = {
    "n": len(R), "on_ge10": sum(r["jump_on"] >= .1 for r in R), "on_ge25": sum(r["jump_on"] >= .25 for r in R),
    "on_ge50": sum(r["jump_on"] >= .5 for r in R),
    "off_sharp": sum(r["jump_off"] >= .5 for r in R), "off_sharp_on_sharp": sum(r["jump_off"] >= .5 and r["jump_on"] >= .5 for r in R),
    "off_sharp_on25": sum(r["jump_off"] >= .5 and r["jump_on"] >= .25 for r in R),
    "med_off": med([r["jump_off"] for r in R]), "med_on": med([r["jump_on"] for r in R]),
    "PL_lt50": sum(r["PL_on"] < .5 for r in R),
    "V_mad": float(np.mean([abs(r["V_on"] - r["Plo_on"]) for r in R])),
    "V_bias": float(np.mean([r["V_on"] - r["Plo_on"] for r in R])),
    "V_r_logit": float(np.corrcoef(*(np.log(np.clip(x, .02, .98) / (1 - np.clip(x, .02, .98)))
                                     for x in (np.array([r["V_on"] for r in R]), np.array([r["Plo_on"] for r in R]))))[0, 1]),
    "jumps": [{"b": r["behavior_id"], "off": round(r["jump_off"], 4), "on": round(r["jump_on"], 4), "PL": round(r["PL_on"], 3),
               "sel": r in sel} for r in R],
    "behaviors": [{"b": b, "n": len(v), "med": med([r["jump_on"] for r in v]), "sharp": sum(r["jump_on"] >= .5 for r in v),
                   "big": sum(r["jump_on"] >= .25 for r in v), "pl_low": sum(r["PL_on"] < .5 for r in v), "p2": p2(b),
                   "toks": toks.get(b, [])} for b, v in sorted(beh.items())],
    "p2": {"n": len(sel), "prompts": len({r["prompt_id"] for r in sel}), "q_act": med([a["q"] for a in act]),
           "p_act": med([a["p"] for a in act]), "alt_q": med(alts), "rel": med([r["fork"][0] / r["L"] for r in sel]),
           "first5": sum(r["fork"][0] < 5 for r in sel)},
}
# fresh thinking-on confirmation of the Phase 2 set (confirm_on.py summary)
P2 = [json.loads(l) for l in (here.parent / "results/phase2_forks.jsonl").read_text().splitlines()]
d["rescore"]["fresh"] = {
    "n": len(P2), "mean_sel": float(np.mean([r["jump_on"] for r in P2])), "mean_fresh": float(np.mean([r["jump_fresh"] for r in P2])),
    "med_fresh": med([r["jump_fresh"] for r in P2]), "ge25": sum(r["jump_fresh"] >= .25 for r in P2), "ge50": sum(r["jump_fresh"] >= .5 for r in P2),
    "prompts50": len({r["prompt_id"] for r in P2 if r["jump_fresh"] >= .5}),
    "Plo": med([r["Plo_fresh"] for r in P2]), "Phi": med([r["Phi_fresh"] for r in P2]),
    "behaviors": [{"b": b, "n": len(v), "med": med([r["jump_fresh"] for r in v]), "ge50": sum(r["jump_fresh"] >= .5 for r in v)}
                  for b, v in sorted({r["behavior_id"]: [x for x in P2 if x["behavior_id"] == r["behavior_id"]] for r in P2}.items())],
}
# Phase 2 (export_phase2.py) and the neuron logit readouts (neuron_logits.py output, top-10 set per behavior)
d["p2"] = json.loads((here.parent / "results/site_phase2.json").read_text())
d["p2"]["sfx"] = json.loads((here.parent / "results/site_sidefx.json").read_text())  # sidefx.py export
d["p2"]["up"] = json.loads((here.parent / "results/site_upstream.json").read_text())  # uprate.py export
d["p2"]["held"] = json.loads((here.parent / "results/heldfork_dai.json").read_text())  # heldfork.py
d["p2"]["sh"] = json.loads((here.parent / "results/site_shared.json").read_text())  # rate3.py export
d["p2"]["scr"] = json.loads((here.parent / "results/site_screen.json").read_text())  # screen1.py export
d["p2"]["sn"] = json.loads((here.parent / "results/site_single.json").read_text())  # rate3.py export, RATE3_MODE=single
READS = {  # short reading of each output direction, only where the top tokens agree; sign-neutral
    (54, 5531): "calling / call", (53, 15385): "报警 (call police) / calling", (43, 11647): "immediately", (63, 12841): "AM / IM",
    (36, 4299): "urgent / emergency", (58, 1358): "reach ↔ call", (55, 5296): "SOS / help", (52, 4299): "emergency", (61, 15915): "are ↔ am",
    (63, 10920): "yes", (63, 5947): "indeed / 确实", (63, 12028): "yes", (62, 9486): "yes", (59, 5179): "不是 (is not) ↔ No",
    (52, 15334): "yes", (52, 3046): "yes", (38, 5297): "indeed / yes", (61, 17384): "yes / indeed",
    (62, 9559): "my", (62, 10923): "own / 自己", (63, 4052): "my / myself", (63, 9136): "ourselves / yourself ↔ I",
    (52, 1644): "checked / investigated / tested", (49, 1055): "my / 我", (56, 3964): "themselves / own ↔ your", (50, 5794): "extensively",
    (63, 9712): "cannot", (63, 10394): "I", (23, 11744): "imagine", (35, 1483): "yourself ↔ myself",
    (62, 11096): "meteor", (62, 702): "platform", (63, 8198): "star", (63, 15796): "safety", (60, 5746): "laser",
    (58, 14787): "aircraft / 飞机", (53, 10002): "advanced", (25, 7354): "projector / 投影", (63, 12981): "bal(ance)",
}
groups, cur = [], None
for line in (here.parent / "results_neuron_logits.txt").read_text().splitlines():
    m = re.match(r"== (\S+)", line)
    if m:
        if not groups or groups[-1]["b"] != m.group(1):
            groups.append({"b": m.group(1), "rows": []})
        cur = None
        continue
    m = re.match(r"\s+L\s*(\d+) n\s*(\d+)\s+A_last mean ([+-][\d.]+)", line)
    if m:
        L, n = int(m.group(1)), int(m.group(2))
        if any(r["L"] == L and r["n"] == n for r in groups[-1]["rows"]):
            cur = None
            continue
        cur = {"L": L, "n": n, "a": float(m.group(3)), "reads": READS.get((L, n))}
        groups[-1]["rows"].append(cur)
    elif cur is not None and line.strip()[:1] in ("+", "-"):
        cur["plus" if line.strip()[0] == "+" else "minus"] = [t for t in ast.literal_eval(line.strip()[2:]) if "\ufffd" not in t]  # drop partial-byte tokens
for g in groups:
    g["rows"].sort(key=lambda r: -r["L"])
d["p2"]["neurons"] = groups
blob = json.dumps(d, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
html = (here / "template.html").read_text().replace("/*DATA*/", blob)
(here / "index.html").write_text(html)
print(len(html), "bytes")
