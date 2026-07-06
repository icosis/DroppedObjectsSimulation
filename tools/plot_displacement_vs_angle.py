"""
plot_displacement_vs_angle.py
=============================
Scatter plot of horizontal displacement (cm) vs MEASURED water-entry angle
(degrees, from entry_angles.csv) — one point per video, coloured by ramp group.

Displacement is recomputed with the validated trajectory tracker
(trace_trajectory.trace_video) so it uses the global calibration and the same
exclusion rules as trajectories_all.png (floor bounces, sparse tracks).

Run from the video folder:
    python C:/.../tools/plot_displacement_vs_angle.py --angles entry_angles.csv

Outputs:
    displacement_vs_entry_angle.png   the figure
    displacement_vs_entry_angle.csv   the joined per-video data
"""

import os
import sys
import csv
import json
import argparse
import io
import contextlib

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(__file__))
from trace_trajectory import (
    trace_video, load_calibration, bounce_rebound_cm, BOUNCE_LIMIT_CM,
)

LABELS_FILE = "labels.json"
ANGLE_COLOR = {30: "tab:red", 45: "tab:green", 60: "tab:blue"}
RESID_CUTOFF = 12.0   # entry-angle rows further than this from prediction are bad detections


def main():
    parser = argparse.ArgumentParser(
        description="Plot displacement vs measured entry angle.")
    parser.add_argument("--angles", default="entry_angles.csv", metavar="CSV",
                        help="entry_angles.csv from batch_entry_angles.py")
    parser.add_argument("--out", default="displacement_vs_entry_angle",
                        help="output basename (png + csv)")
    args = parser.parse_args()

    if not os.path.exists(args.angles):
        sys.exit(f"{args.angles} not found — run batch_entry_angles.py first, "
                 f"or pass --angles PATH")

    calib = load_calibration()

    with open(args.angles) as f:
        angle_rows = {r["filename"]: r for r in csv.DictReader(f)}
    with open(LABELS_FILE) as f:
        labels = json.load(f)

    rows = []
    videos = sorted(angle_rows.keys())
    print(f"Joining {len(videos)} videos ...")
    for i, name in enumerate(videos):
        r = angle_rows[name]
        ramp = labels.get(name)
        if ramp is None or not os.path.exists(name):
            continue
        # Entry angle must be a clean detection
        try:
            angle = float(r["measured_angle_deg"])
            resid = float(r["residual_deg"])
        except (ValueError, KeyError):
            print(f"  {name}: skip (no clean entry angle)")
            continue
        if abs(resid) > RESID_CUTOFF:
            print(f"  {name}: skip (entry-angle outlier, residual {resid:+.1f}°)")
            continue

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            trace = trace_video(name, calib)
        if trace is None or len(trace["path_px"]) < 4:
            print(f"  {name}: skip (no reliable trajectory)")
            continue
        depth_frac = ((trace["path_px"][-1][1] - trace["y_surface_px"])
                      / (trace["y_bottom_px"] - trace["y_surface_px"]))
        if depth_frac < 0.5:
            print(f"  {name}: skip (trajectory ends at {depth_frac:.0%} depth)")
            continue
        if bounce_rebound_cm(trace) > BOUNCE_LIMIT_CM:
            print(f"  {name}: skip (floor bounce)")
            continue

        p = trace["path_px"]
        disp = abs(p[-1][0] - p[0][0]) / trace["px_per_cm"]
        rows.append({"filename": name, "ramp_deg": ramp,
                     "entry_angle_deg": angle, "displacement_cm": round(disp, 2)})
        print(f"  [{i+1:3d}/{len(videos)}] {name}: {angle:.1f}° → {disp:.2f} cm")

    if len(rows) < 3:
        sys.exit("Not enough joined data points.")

    # ── Write joined CSV ─────────────────────────────────────────────────────
    csv_path = args.out + ".csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "ramp_deg",
                                          "entry_angle_deg", "displacement_cm"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved joined data → {csv_path}  ({len(rows)} videos)")

    # ── Figure ───────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(7, 5.5))

    for ramp in sorted(ANGLE_COLOR):
        grp = [r for r in rows if r["ramp_deg"] == ramp]
        if not grp:
            continue
        xs = [r["entry_angle_deg"] for r in grp]
        ys = [r["displacement_cm"] for r in grp]
        ax.scatter(xs, ys, s=36, color=ANGLE_COLOR[ramp], alpha=0.75,
                   edgecolors="white", linewidths=0.6,
                   label=f"{ramp}° ramp  (n={len(grp)})")

    # Overall linear fit across all points
    ax_all = np.array([r["entry_angle_deg"] for r in rows])
    ay_all = np.array([r["displacement_cm"] for r in rows])
    b, a = np.polyfit(ax_all, ay_all, 1)          # y = b·x + a
    xfit = np.linspace(ax_all.min() - 2, ax_all.max() + 2, 50)
    ax.plot(xfit, b * xfit + a, "--", color="gray", lw=1.5, zorder=1)
    r_coef = np.corrcoef(ax_all, ay_all)[0, 1]
    ax.annotate(f"fit: {b:.2f} cm/°   r = {r_coef:.2f}",
                xy=(0.03, 0.05), xycoords="axes fraction",
                fontsize=9, color="dimgray")

    ax.set_xlabel("measured entry angle (° below horizontal)")
    ax.set_ylabel("horizontal displacement (cm)")
    ax.set_title("Horizontal displacement vs water-entry angle")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    png_path = args.out + ".png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure → {png_path}")

    # ── Group stats ──────────────────────────────────────────────────────────
    print("\nGroup summary (by ramp angle):")
    for ramp in sorted(ANGLE_COLOR):
        grp = [r for r in rows if r["ramp_deg"] == ramp]
        if not grp:
            continue
        angs = [r["entry_angle_deg"] for r in grp]
        disp = [r["displacement_cm"] for r in grp]
        print(f"  {ramp}° ramp: entry {sum(angs)/len(angs):5.1f}°  "
              f"displacement {sum(disp)/len(disp):5.2f} cm  (n={len(grp)})")


if __name__ == "__main__":
    main()
