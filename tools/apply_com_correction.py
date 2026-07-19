"""
apply_com_correction.py
=======================
Convert raw displacements (nose-pierce -> centroid) to centre-of-mass
displacements (CoM -> CoM), matching the simulation's reference frame.

At the instant the nose touches the water, the pipe's CoM sits (L/2)*cos(theta)
up-ramp of the pierce point (L = 7.62 cm, theta = measured entry angle).  The
simulation's displacement origin is the CoM at that instant, so the raw
measurement understates CoM displacement by exactly that offset:

    disp_com = disp_raw + (L/2) * cos(theta_entry)

Per-video measured entry angles come from entry_angles.csv; videos whose angle
measurement failed or was an outlier (|residual| > 12 deg) fall back to their
group's mean measured angle.

Run from the ExperimentalVideos folder (after analyze_videos.py and
batch_entry_angles.py):
    python C:/Users/micha/DroppedSim/tools/apply_com_correction.py

Writes results_com.csv.
"""

import csv
import sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RESULTS_CSV = "results.csv"
ANGLES_CSV  = "entry_angles.csv"
OUT_CSV     = "results_com.csv"

HALF_LENGTH_CM = 7.62 / 2.0   # 3 inch pipe
RESID_CUTOFF   = 12.0         # deg — beyond this the angle measurement failed


def main():
    # Measured entry angles per video, plus group means of the good ones
    angles, good_by_group = {}, {}
    with open(ANGLES_CSV) as f:
        for r in csv.DictReader(f):
            if r["measured_angle_deg"] in ("ERROR", ""):
                continue
            meas = float(r["measured_angle_deg"])
            pred = float(r["predicted_angle_deg"])
            ramp = int(r["ramp_angle_deg"])
            ok   = abs(meas - pred) <= RESID_CUTOFF
            angles[r["filename"]] = (meas, ok)
            if ok:
                good_by_group.setdefault(ramp, []).append(meas)

    group_mean = {a: sum(v) / len(v) for a, v in good_by_group.items()}
    print("Group mean measured entry angles:",
          {a: round(m, 1) for a, m in group_mean.items()})

    rows_out = []
    with open(RESULTS_CSV) as f:
        for r in csv.DictReader(f):
            row = {
                "filename":  r["filename"],
                "angle_deg": r["angle_deg"],
                "displacement_raw_cm": r["displacement_cm"],
                "entry_angle_used_deg": "",
                "com_offset_cm": "",
                "displacement_com_cm": "ERROR",
                "x_entry_px":   r["x_entry_px"],
                "x_contact_px": r["x_contact_px"],
            }
            if r["displacement_cm"] != "ERROR":
                ramp = int(r["angle_deg"])
                meas, ok = angles.get(r["filename"], (None, False))
                theta = meas if (meas is not None and ok) else group_mean[ramp]
                offset = HALF_LENGTH_CM * np.cos(np.radians(theta))
                row["entry_angle_used_deg"] = round(theta, 1)
                row["com_offset_cm"]        = round(offset, 2)
                row["displacement_com_cm"]  = round(
                    float(r["displacement_cm"]) + offset, 2)
            rows_out.append(row)

    fields = ["filename", "angle_deg", "displacement_raw_cm",
              "entry_angle_used_deg", "com_offset_cm", "displacement_com_cm",
              "x_entry_px", "x_contact_px"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)

    ok_n = sum(1 for r in rows_out if r["displacement_com_cm"] != "ERROR")
    print(f"Wrote {OUT_CSV}: {ok_n}/{len(rows_out)} corrected rows.")


if __name__ == "__main__":
    main()
