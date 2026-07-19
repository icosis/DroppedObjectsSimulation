"""
plot_distribution.py
====================
Poster figure: distribution of experimental horizontal displacement per drop
angle — box plot + jittered scatter of every valid trial. No simulation
overlay (see plot_comparison.py for the experiment-vs-simulation figure).

Run from the ExperimentalVideos folder:
    python C:/Users/micha/DroppedSim/tools/plot_distribution.py
"""

import csv
import statistics
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RESULTS_CSV = "results_com.csv"
OUT_PATH    = "displacement_scatter.png"

COLORS = {30: "#2F6FB8", 45: "#C77F1B", 60: "#B5384B"}
ANGLES = [30, 45, 60]

# ── Load and clean (same filters as plot_comparison.py) ───────────────────────
valid = {a: [] for a in ANGLES}
with open(RESULTS_CSV) as f:
    for row in csv.DictReader(f):
        if row["displacement_com_cm"] == "ERROR":
            continue
        angle = int(row["angle_deg"])
        disp  = float(row["displacement_com_cm"])
        raw   = float(row["displacement_raw_cm"])
        xe    = int(row["x_entry_px"])
        xc    = int(row["x_contact_px"])
        if xc < xe or raw > 25 or xe < 920:
            continue
        valid[angle].append(disp)

print("Valid counts:", {a: len(valid[a]) for a in ANGLES})

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))

box_data   = [valid[a] for a in ANGLES]
box_labels = [f"{a}°\n(n={len(valid[a])})" for a in ANGLES]

bp = ax.boxplot(box_data, patch_artist=True, tick_labels=box_labels,
                widths=0.5,
                medianprops={"color": "black", "linewidth": 2},
                whiskerprops={"linewidth": 1.2},
                capprops={"linewidth": 1.2},
                flierprops={"marker": ""})          # fliers drawn by the scatter
for patch, a in zip(bp["boxes"], ANGLES):
    patch.set_facecolor(COLORS[a])
    patch.set_alpha(0.35)
    patch.set_linewidth(1.2)

# Jittered scatter of every trial on top
rng = np.random.default_rng(42)
for i, a in enumerate(ANGLES, 1):
    vals   = valid[a]
    jitter = rng.uniform(-0.16, 0.16, len(vals))
    ax.scatter(i + jitter, vals, s=34, color=COLORS[a],
               alpha=0.75, zorder=3, edgecolors="white", linewidths=0.6)

# Mean labels beside each group
for i, a in enumerate(ANGLES, 1):
    m = statistics.mean(valid[a])
    ax.text(i + 0.32, m, f"mean {m:.1f}", va="center", fontsize=10,
            color="#444444")

ax.set_xlabel("Drop Angle", fontsize=13)
ax.set_ylabel("CoM Horizontal Displacement (cm)", fontsize=13)
ax.set_title("Centre-of-Mass Displacement per Drop Angle — all valid trials",
             fontsize=13)
ax.set_ylim(0, 20)
ax.grid(axis="y", alpha=0.3, linestyle=":")
ax.tick_params(labelsize=12)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
print(f"Saved -> {OUT_PATH}")
