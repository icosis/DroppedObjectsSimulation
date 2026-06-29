"""
batch_entry_angles.py
=====================
Runs entry-angle detection on every labelled video and saves results to
entry_angles.csv in the current directory.

Run from the Data folder (where calibration.json lives):
    python "C:\\Users\\Michael\\Dropped\\DroppedObjectsSimulation\\tools\\batch_entry_angles.py"
"""

import sys
import os
import json
import csv

sys.path.insert(0, os.path.dirname(__file__))
from measure_entry_angle import (
    load_calibration, detect_surface, measure_entry_angle, theoretical_pitch_down
)

VIDEO_FOLDER     = "Dropped Object Folder"
LABELS_FILE      = os.path.join(VIDEO_FOLDER, "labels.json")
CALIBRATION_FILE = "calibration.json"
OUTPUT_CSV       = "entry_angles.csv"


def main():
    # Load calibration
    calib = load_calibration()
    if calib is None:
        sys.exit("calibration.json not found. Run analyze_videos.py --calibrate first.")
    print(f"Calibration loaded: {calib['px_per_cm']:.2f} px/cm")

    # Load labels
    if not os.path.exists(LABELS_FILE):
        sys.exit(f"{LABELS_FILE} not found. Run analyze_videos.py --label first.")
    with open(LABELS_FILE) as f:
        labels = json.load(f)

    videos = sorted(labels.keys())
    print(f"Found {len(videos)} labelled videos\n")

    rows = []
    for i, filename in enumerate(videos):
        ramp_angle = labels[filename]
        video_path = os.path.join(VIDEO_FOLDER, filename)

        print(f"[{i+1:3d}/{len(videos)}]  {filename}  ramp={ramp_angle}°", end="  ")
        sys.stdout.flush()

        if not os.path.exists(video_path):
            print("SKIP — file not found")
            rows.append({
                "filename":         filename,
                "ramp_angle_deg":   ramp_angle,
                "measured_angle_deg": "ERROR",
                "predicted_angle_deg": "",
                "residual_deg":     "",
                "frame":            "",
            })
            continue

        result = measure_entry_angle(video_path, calib, ramp_deg=ramp_angle, debug=False)

        if result is None:
            print("FAIL — no detection")
            rows.append({
                "filename":           filename,
                "ramp_angle_deg":     ramp_angle,
                "measured_angle_deg": "ERROR",
                "predicted_angle_deg": "",
                "residual_deg":       "",
                "frame":              "",
            })
            continue

        delta, _ = theoretical_pitch_down(ramp_angle)
        predicted = round(ramp_angle + delta, 1)
        residual  = round(result["angle_deg"] - predicted, 1)

        print(f"measured={result['angle_deg']}°  predicted={predicted}°  residual={residual:+.1f}°")
        rows.append({
            "filename":            filename,
            "ramp_angle_deg":      ramp_angle,
            "measured_angle_deg":  result["angle_deg"],
            "predicted_angle_deg": predicted,
            "residual_deg":        residual,
            "frame":               result["frame"],
        })

    # Write CSV
    fieldnames = ["filename", "ramp_angle_deg", "measured_angle_deg",
                  "predicted_angle_deg", "residual_deg", "frame"]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok    = sum(1 for r in rows if r["measured_angle_deg"] not in ("ERROR", ""))
    fails = len(rows) - ok
    print(f"\n{'─'*60}")
    print(f"Done: {ok} succeeded, {fails} failed")
    print(f"Results saved to: {OUTPUT_CSV}")

    # Print summary by angle group
    print(f"\n── Mean measured entry angle by ramp angle ──────────────")
    for ramp in [30, 45, 60]:
        vals = [r["measured_angle_deg"] for r in rows
                if r["ramp_angle_deg"] == ramp
                and isinstance(r["measured_angle_deg"], float)]
        if vals:
            mean = round(sum(vals) / len(vals), 1)
            delta, _ = theoretical_pitch_down(ramp)
            predicted = round(ramp + delta, 1)
            print(f"  {ramp}°  →  mean measured {mean}°  "
                  f"(predicted {predicted}°,  n={len(vals)})")


if __name__ == "__main__":
    main()
