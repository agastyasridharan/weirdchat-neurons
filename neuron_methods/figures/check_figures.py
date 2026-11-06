"""Check every number in the two figures against the result files. Run from anywhere: python figures/check_figures.py"""
import ast
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        fails.append(name)


def load(rel):
    return json.loads((RES / rel).read_text())


def table_of(path, names):
    """Read the module-level constants of a figure script without running it."""
    tree = ast.parse(path.read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names:
            out[node.targets[0].id] = ast.literal_eval(node.value)
    return out


def pct(x):
    return int(Decimal(str(x * 100)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


# 1. r_vs_size.png
H = load("heldout/heldout_summary.json")
F = table_of(HERE / "fig_r_vs_size.py", {"SIZES", "R", "RAND"})
for m, pts in F["R"].items():
    for s, (r, lo, hi) in zip(F["SIZES"], pts):
        key = f"M1|{s}" if (m, s) == ("M3", 3) else f"{m}|{s}"  # M3's 3-neuron set is M1's
        got = [round(v, 2) for v in H["R"][key]]
        check(f"plot {m} at {s}: R and interval", got == [r, lo, hi], f"file {got}, plot {[r, lo, hi]}")
for m, r in F["RAND"].items():
    check(f"plot {m} random 100-neuron set", round(H["rand"][m][0], 2) == r, f"file {H['rand'][m][0]:.4f}")
    check(f"{m} 100-neuron set minus random: interval below 0", H["rand"][m][3] < 0, f"D {H['rand'][m][1]:+.3f}")

# 2. harm_table.png
Fin, Bench = load("final/final_summary.json"), load("bench/bench_summary.json")
T = table_of(HERE / "fig_harm_table.py", {"ROWS"})["ROWS"]
BEH = {"911": "claims-called-911", "Code execution": "fabricated-code-execution",
       "False embodiment": "false-physical-embodiment", "Laser": "laser-at-aircraft"}
OTHER_NAME = {"claims-called-911": "911 claims"}
for beh, picked, left, (ok, legit), other, bench, dnll in T:
    b = BEH[beh]
    pick = H["final_sets"][b]
    m, _, size = pick["set"].split("|")
    check(f"{beh}: picked set", picked == f"{m}, {size} neurons", pick["set"])
    check(f"{beh}: behavior left", left == f"{pct(pick['ratio'])}%", f"{pick['ratio']:.4f}")
    fails_legit = [r for r, x in Fin["legit"].items() if x[pick["set"]]["exact_p_lower"] < 0.025]
    check(f"{beh}: legitimate use {'passes' if ok else 'fails'}", ok == (not fails_legit), str(fails_legit))
    if not ok:
        L = Fin["legit"]
        mb, ms = L["legit-laser-mention"]["base"][0], L["legit-laser-mention"][pick["set"]]["rate"]
        wb, ws = L["legit-laser-safety"]["base"][0], L["legit-laser-safety"][pick["set"]]["rate"]
        want = f"mentions\n{pct(mb)}% → {pct(ms)}%, warnings {pct(wb)}% → {pct(ws)}%"
        check(f"{beh}: legitimate-use text", legit.endswith(want), f"{mb:.4f}->{ms:.4f}, {wb:.4f}->{ws:.4f}")
    moved = {o: x for o, x in Fin["off_target"][pick["set"]].items() if x["moves"]}
    if other == "none flagged":
        check(f"{beh}: other behaviors, none flagged", not moved, str(list(moved)))
    else:
        (o, x), = moved.items()
        want = f"{OTHER_NAME[o]} fall\n({x['set'][0]} vs {x['unedited'][0]} of {x['set'][1]})"
        check(f"{beh}: other behaviors", other == want and x["set"][1] == x["unedited"][1] and x["set"][0] < x["unedited"][0],
              f"{x['set']} vs {x['unedited']}, two-sided p {x['exact_p_two_sided']:.3f}")
    s, u = Fin["bench"][pick["set"]], Fin["bench"]["base"]
    want = " / ".join(f"{d:+d}".replace("+0", "0").replace("-", "−") for d in (s["mmlu"] - u["mmlu"], s["gsm"] - u["gsm"], s["yn"] - u["yn"]))
    check(f"{beh}: benchmark changes", bench == want, want)
    if b == "claims-called-911":  # job 8's reading has the one-text glitch; see the figure's notes
        check(f"{beh}: dNLL from jobs 6 and 7", dnll == f"{pick['dnll']:.4f}" == f"{Bench['dnll'][pick['set']]:.4f}",
              f"job 6 {pick['dnll']:.6f}, job 7 {Bench['dnll'][pick['set']]:.6f}, job 8 {Fin['dnll'][pick['set']]:.4f}")
    else:
        check(f"{beh}: dNLL", dnll == f"{Fin['dnll'][pick['set']]:.4f}", f"{Fin['dnll'][pick['set']]:.6f}")

# 3. the glitch note: job 8's 911 reading with and without text 71
cond = {}
for line in (RES / "final" / "cond.jsonl").read_text().splitlines():
    r = json.loads(line)
    cond[r["key"]] = r
base, c = cond["base"], cond[H["final_sets"]["claims-called-911"]["set"]]
per = [(c["nll"][i] - base["nll"][i]) / base["ntok"][i] for i in range(len(base["nll"]))]
worst = max(range(len(per)), key=per.__getitem__)
keep = [i for i in range(len(per)) if i != worst]
clean = (sum(c["nll"][i] for i in keep) - sum(base["nll"][i] for i in keep)) / sum(base["ntok"][i] for i in keep)
full = (sum(c["nll"]) - sum(base["nll"])) / sum(base["ntok"])
check("911 note: job 8 reads 0.061; one text far above the rest; 0.0073 without it",
      f"{full:.3f}" == "0.061" and per[worst] > 5 and max(per[i] for i in keep) < 0.5 and f"{clean:.4f}" == "0.0073",
      f"text {worst}: +{per[worst]:.2f} nats/token; next highest {max(per[i] for i in keep):+.3f}; without it {clean:.5f}")

print("\nALL FIGURE NUMBERS MATCH" if not fails else f"\n{len(fails)} MISMATCHES")
