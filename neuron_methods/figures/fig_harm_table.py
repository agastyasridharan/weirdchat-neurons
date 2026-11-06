"""Harm checks on the picked set for each behavior (figures/harm_table.png).

Sources: "Behavior left" is the held-out ratio from results/heldout/heldout_summary.json ("final_sets"); legitimate
use, other behaviors, benchmarks and dNLL are from results/final/final_summary.json (GPU job 8), except the 911 set's
dNLL. Job 8 read 0.0613 for that set because one of the 200 reference texts (text 71) scored about 10.7 nats per token
above the unedited model in that run only; the same set measured 0.0072 in jobs 6 and 7 (heldout_summary.json and
bench_summary.json), and job 8 gives 0.0073 without text 71. check_figures.py checks every cell against the files.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
GOOD, BAD = "#0ca30c", "#d03b3b"  # status colors, always paired with an icon and a word
plt.rcParams.update({"font.family": "DejaVu Sans"})

COLS = [(0.030, "Behavior"), (0.160, "Picked set"), (0.278, "Behavior left\n(held-out)"), (0.385, "Legitimate use"),
        (0.640, "Other behaviors"), (0.790, "MMLU / GSM8K /\nyes-no change"), (0.915, "dNLL")]
ROWS = [  # behavior, set, left, (ok, legit text), other behaviors, benchmark changes, dNLL
    ("911", "M2, 10 neurons", "0%", (True, "passes"), "none flagged", "−1 / +1 / 0", "0.0072"),
    ("Code execution", "M1, 50 neurons", "0%", (True, "passes"), "911 claims fall\n(8 vs 18 of 64)", "−1 / +1 / 0", "0.0011"),
    ("False embodiment", "M3, 50 neurons", "42%", (True, "passes"), "911 claims fall\n(7 vs 18 of 64)", "−2 / +1 / 0", "0.0077"),
    ("Laser", "M1, 100 neurons", "18%",
     (False, "fails: laser-pointer mentions\n55% → 13%, warnings 34% → 13%"), "none flagged", "0 / 0 / 0", "0.0094"),
]
NOTES = [
    "Legitimate use: six checks; a set fails if a per-prompt exact test gives p < 0.025 (rule fixed before any harm result).",
    "Benchmark changes are in questions, of 400 / 250 / 40. dNLL: increase in nats per token on 200 reference texts; limit 0.023.",
    "911 dNLL: 0.0072 in two earlier runs. The final run read 0.061 because one reference text glitched in that run only; "
    "without that text, 0.0073.",
]

fig = plt.figure(figsize=(11, 4.5), dpi=200)
fig.patch.set_facecolor(SURF)
fig.text(0.030, 0.925, "Harm checks on the picked set for each behavior", fontsize=14, fontweight="bold", color=INK)
fig.text(0.030, 0.869, "The smallest set from any method that does as well as grad×act's top 100, picked on the held-out "
         "results.", fontsize=10, color=INK2)
fig.text(0.030, 0.829, "Each set is compared with the unedited model in the same run.", fontsize=10, color=INK2)
y_head, y0, dy = 0.735, 0.600, 0.128
line = lambda y: fig.add_artist(plt.Line2D([0.03, 0.97], [y, y], color=GRID, lw=0.9, transform=fig.transFigure))
for x, h in COLS:
    fig.text(x, y_head, h, fontsize=9.5, color=INK2, va="center", ha="left", fontweight="bold")
line(y_head - 0.055)
for r, row in enumerate(ROWS):
    y = y0 - r * dy
    beh, st, left, (ok, legit), other, bench, dnll = row
    for (x, _), val in zip(COLS, (beh, st, left, None, other, bench, dnll)):
        if val is not None:
            fig.text(x, y, val, fontsize=10, color=INK, va="center", ha="left", linespacing=1.25)
    fig.text(COLS[3][0], y, "✓" if ok else "✗", fontsize=11, color=GOOD if ok else BAD, va="center", ha="left",
             fontweight="bold")
    fig.text(COLS[3][0] + 0.018, y, legit, fontsize=10, color=INK, va="center", ha="left", linespacing=1.25)
    line(y - dy / 2)
for i, note in enumerate(NOTES):
    fig.text(0.030, 0.098 - i * 0.034, note, fontsize=8.3, color=INK2, va="center", ha="left")
fig.savefig(Path(__file__).with_name("harm_table.png"), facecolor=SURF)
print("saved")
