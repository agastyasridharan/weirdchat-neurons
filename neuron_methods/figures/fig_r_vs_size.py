"""Held-out behavior rate vs. neurons zeroed (figures/r_vs_size.png).
R and its 95% interval for every set, and R for each matched random 100-neuron set, from
results/heldout/heldout_summary.json ("R" and "rand"), rounded to 2 decimals. check_figures.py checks them.
Colors come from a colorblind-checked categorical palette; each line also has a direct label."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIZES = [3, 5, 10, 20, 50, 100]
R = {  # method: [(R, lo, hi) per size], from the job 6 output
    "M1": [(0.78, 0.62, 0.95), (0.55, 0.42, 0.69), (0.32, 0.21, 0.49), (0.35, 0.24, 0.49), (0.19, 0.08, 0.31), (0.15, 0.07, 0.24)],
    "M2": [(0.87, 0.65, 1.06), (0.63, 0.50, 0.77), (0.47, 0.36, 0.60), (0.41, 0.29, 0.57), (0.28, 0.13, 0.47), (0.18, 0.08, 0.30)],
    "M3": [(0.78, 0.62, 0.95), (0.63, 0.48, 0.78), (0.48, 0.32, 0.64), (0.34, 0.22, 0.46), (0.30, 0.16, 0.45), (0.24, 0.14, 0.34)],
    "M4": [(0.73, 0.57, 0.89), (0.64, 0.50, 0.83), (0.55, 0.42, 0.90), (0.48, 0.32, 0.60), (0.34, 0.24, 0.45), (0.21, 0.11, 0.34)],
}
RAND = {"M1": 0.97, "M2": 1.06, "M3": 1.12, "M4": 0.96}  # each method's matched random 100-neuron set
NAME = {"M1": "M1  grad×act", "M2": "M2  grad×act, layers < 48", "M3": "M3  iterative grad×act", "M4": "M4  RelP"}
COLOR = {"M1": "#2a78d6", "M2": "#eb6834", "M3": "#1baf7a", "M4": "#eda100"}
MARK = {"M1": "o", "M2": "s", "M3": "^", "M4": "D"}
DODGE = {"M1": 0.95, "M2": 0.983, "M3": 1.017, "M4": 1.05}
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(figsize=(10.5, 6.0), dpi=200)
fig.patch.set_facecolor(SURF)
ax.set_facecolor(SURF)
ax.set_xscale("log")
ax.axhline(1.0, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
ax.text(6.2, 1.015, "unedited", color=INK2, fontsize=9.5, va="bottom", ha="left")
for m in R:
    xs = [s * DODGE[m] for s in SIZES]
    ys = [v[0] for v in R[m]]
    lo = [v[0] - v[1] for v in R[m]]
    hi = [v[2] - v[0] for v in R[m]]
    first = 0
    if m == "M3":  # M3 starts from M1's top 3, so its 3-neuron point is M1's point: draw it once
        xs[0], first = SIZES[0] * DODGE["M1"], 1
    ax.errorbar(xs[first:], ys[first:], yerr=[lo[first:], hi[first:]], fmt="none", ecolor=COLOR[m], elinewidth=1.1,
                alpha=0.55, capsize=0, zorder=2)
    ax.plot(xs, ys, color=COLOR[m], lw=2, marker=MARK[m], ms=7, mec=SURF, mew=1.2, zorder=3, label=NAME[m],
            markevery=list(range(first, len(xs))))
    # the matched random set at the same size, hollow, directly above the method's own 100-neuron point
    ax.plot([100 * DODGE[m]], [RAND[m]], marker=MARK[m], ms=8.5, mfc="none", mec=COLOR[m], mew=1.8, ls="none", zorder=3)
ax.text(118, 1.10, "random 100-neuron sets\n(hollow, one per method)", color=INK2, fontsize=9.5, va="center", ha="left")
# direct labels at the right end, spread so they don't collide; a thin leader joins each label to its line end
order = sorted(R, key=lambda m: R[m][-1][0])
for j, m in enumerate(order):
    y_end, y_lab = R[m][-1][0], 0.06 + 0.085 * j
    ax.plot([100 * DODGE[m] * 1.04, 182], [y_end, y_lab], color="#b5b3ad", lw=0.8, zorder=2)
    ax.text(186, y_lab, f"{m}  {y_end:.2f}", color=INK, fontsize=10, va="center", ha="left")
ax.set_xlim(2.6, 300)
ax.set_ylim(0, 1.38)
ax.set_xticks(SIZES)
ax.set_xticklabels([str(s) for s in SIZES])
ax.minorticks_off()
ax.set_xlabel("Neurons zeroed per behavior", color=INK2)
ax.set_ylabel("Behavior rate / unedited rate\n(mean over 4 behaviors)", color=INK2)
ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(GRID)
ax.tick_params(colors=INK2, length=0)
ax.legend(loc="upper center", bbox_to_anchor=(0.40, 1.0), frameon=True, framealpha=1, facecolor=SURF, edgecolor="none", fontsize=9.5,
          labelcolor=INK, ncol=2, columnspacing=1.6)
fig.text(0.075, 0.958, "Held-out behavior rate vs. neurons zeroed", fontsize=14, fontweight="bold", color=INK, ha="left")
fig.text(0.075, 0.918, "38 held-out prompts (911, code execution, false embodiment, laser), 16 samples each.",
         fontsize=10, color=INK2, ha="left")
fig.text(0.075, 0.885, "Bars: 95% bootstrap CIs per method. Methods are compared paired, on the same prompts, "
         "so overlapping bars can still differ.", fontsize=10, color=INK2, ha="left")
fig.subplots_adjust(left=0.10, right=0.97, top=0.855, bottom=0.11)
fig.savefig(Path(__file__).with_name("r_vs_size.png"), facecolor=SURF)
print("saved")
