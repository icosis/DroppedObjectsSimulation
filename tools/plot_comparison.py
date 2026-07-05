"""
plot_comparison.py
==================
Generates two comparison plots:
  1. Bar chart — experimental mean vs simulation prediction per angle
  2. Box + scatter — full distribution, split by entry group, with simulation line

Run from the ExperimentalVideos folder:
    python C:/Users/micha/DroppedSim/tools/plot_comparison.py
"""

import csv
import sys
import warnings
import statistics
import numpy as np

# Set display backend BEFORE pyplot is imported, so DropTest's Agg call won't override it
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import the physics simulation from DropTest
sys.path.insert(0, r"C:\Users\micha\DroppedSim")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")          # suppress DropTest's Agg override warning
    from DropTest import simulate_drop

RESULTS_CSV = "results.csv"

# Run the physics model fresh — no experimental data used here
SIM = {a: simulate_drop(a)["displacement_cm"] for a in [30, 45, 60]}
COLORS = {30: "#2F6FB8", 45: "#C77F1B", 60: "#B5384B"}
ANGLES = [30, 45, 60]

# ── Load and clean data ────────────────────────────────────────────────────────
# Since the entry-detection fix (min/max entry-x window in analyze_videos.py),
# all entries land in the valid range x≈980–1130.  The filters below remain as
# safety nets against future detection regressions.
valid = {a: [] for a in ANGLES}

with open(RESULTS_CSV) as f:
    for row in csv.DictReader(f):
        if row["displacement_cm"] == "ERROR":
            continue
        angle = int(row["angle_deg"])
        disp  = float(row["displacement_cm"])
        xe    = int(row["x_entry_px"])
        xc    = int(row["x_contact_px"])
        if xc < xe:    # pipe moved backward — bad read
            continue
        if disp > 25:  # outlier (C0055 — contact detection failure)
            continue
        if xe < 920:   # safety net: left-wall ripple false positive
            continue
        valid[angle].append(disp)

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Horizontal Displacement of Steel Pipe vs Drop Angle",
    fontsize=14, fontweight="bold"
)

# ── Plot 1: Bar chart ──────────────────────────────────────────────────────────
print("Valid counts:", {a: len(valid[a]) for a in ANGLES})
idx      = np.arange(len(ANGLES))
bw       = 0.35
exp_mean = [statistics.mean(valid[a]) for a in ANGLES]
exp_sd   = [statistics.stdev(valid[a]) if len(valid[a]) > 1 else 0 for a in ANGLES]
sim_vals = [SIM[a] for a in ANGLES]

bars_exp = ax1.bar(idx - bw/2, exp_mean, bw,
                   color=[COLORS[a] for a in ANGLES], alpha=0.85,
                   yerr=exp_sd, capsize=5,
                   error_kw={"linewidth": 1.5, "ecolor": "black"})
bars_sim = ax1.bar(idx + bw/2, sim_vals, bw,
                   color=[COLORS[a] for a in ANGLES], alpha=0.35,
                   hatch="//")

# Value labels
for bar, val in zip(bars_exp, exp_mean):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
             f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
for bar, val in zip(bars_sim, sim_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
             f"{val:.1f}", ha="center", va="bottom", fontsize=9, color="#555555")

ax1.set_xticks(idx)
ax1.set_xticklabels(["30°", "45°", "60°"], fontsize=12)
ax1.set_xlabel("Drop Angle", fontsize=12)
ax1.set_ylabel("Horizontal Displacement (cm)", fontsize=12)
ax1.set_title("Mean Experimental vs Simulation", fontsize=12)
ax1.set_ylim(0, 30)
ax1.grid(axis="y", alpha=0.3)
ax1.legend(
    handles=[
        mpatches.Patch(color="grey", alpha=0.85, label="Experimental (mean ± SD)"),
        mpatches.Patch(color="grey", alpha=0.35, hatch="//", label="Simulation"),
    ]
)

# ── Plot 2: Box + scatter ──────────────────────────────────────────────────────
box_data   = [valid[a] for a in ANGLES]
box_labels = [f"{a}°\n(n={len(valid[a])})" for a in ANGLES]
box_colors = [COLORS[a] for a in ANGLES]

bp = ax2.boxplot(box_data, patch_artist=True, tick_labels=box_labels,
                 medianprops={"color": "black", "linewidth": 2})
for patch, color in zip(bp["boxes"], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.55)

# Scatter individual points with jitter
rng = np.random.default_rng(42)
for i, (vals, color) in enumerate(zip(box_data, box_colors), 1):
    jitter = rng.uniform(-0.18, 0.18, len(vals))
    ax2.scatter(i + jitter, vals, s=22, color=color, alpha=0.6, zorder=3)

# Simulation dashed lines
for i, a in enumerate(ANGLES, 1):
    ax2.hlines(SIM[a], i - 0.4, i + 0.4, colors=COLORS[a], linewidths=2.5,
               linestyles="--", zorder=4)
    ax2.text(i + 0.45, SIM[a], f"sim {a}°", va="center",
             fontsize=8, color=COLORS[a])

ax2.set_ylabel("Horizontal Displacement (cm)", fontsize=12)
ax2.set_title("Distribution per Angle — verified detections only\n(dashed = simulation)", fontsize=12)
ax2.set_ylim(0, 30)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
out_path = "displacement_comparison.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved → {out_path}")
