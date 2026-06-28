"""
analyze_videos.py
==================
Measures horizontal displacement of a pipe from water entry to first floor
contact across a batch of side-view videos.

Works even when:
  - The pipe is visible on the ramp from frame 1 (no empty background needed)
  - There are reflections in the tank (static reflections are ignored)
  - The pipe slows down near the floor
  - Videos are unnamed (use --label mode to tag them first)

Workflow:
  1. Label videos with their angle (run once, saves labels.json):
       python C:\\Users\\micha\\DroppedSim\\tools\\analyze_videos.py --label

  2. Calibrate using a reference frame (run once, saves calibration.json):
       python C:\\Users\\micha\\DroppedSim\\tools\\analyze_videos.py --calibrate

  3. Process all videos and write results.csv:
       python C:\\Users\\micha\\DroppedSim\\tools\\analyze_videos.py

  4. Test detection on one video:
       python C:\\Users\\micha\\DroppedSim\\tools\\analyze_videos.py --debug --video clip.mp4

All three steps remember their results in json files, so you only redo them
if something changes.

Assumptions:
  - Horizontal px/cm ≈ vertical px/cm (square pixels, camera roughly centred).
    Small camera angle introduces <5% error — note as a limitation.
  - Pipe is the largest moving object in frame.
"""

import cv2
import numpy as np
import json
import os
import glob
import csv
import argparse
import sys
from collections import deque

CALIBRATION_FILE = "calibration.json"
LABELS_FILE      = "labels.json"
OUTPUT_CSV       = "results.csv"

# Detection tuning
DIFF_WINDOW  = 8      # compare frame t to frame (t - DIFF_WINDOW); higher = more sensitive to slow motion
THRESHOLD    = 12     # pixel brightness change to count as motion
MIN_AREA_PX  = 250    # minimum contour area to be the pipe (filters ripple noise; pipe blob is much larger)
CONTACT_TOL_CM = 3.0  # centroid counts as "at floor" within this many cm


# ── Labeling ──────────────────────────────────────────────────────────────────

def label_videos(videos):
    """
    Shows a frame from each unlabelled video. Press:
      4 → 45°    6 → 60°    7 → 75°
      S → skip (unsure, come back later)
      Q → quit and save progress

    Progress is saved to labels.json after every keypress so you can stop
    and resume at any time without losing work.
    """
    # Load existing labels
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE) as f:
            labels = json.load(f)
    else:
        labels = {}

    remaining = [v for v in videos if os.path.basename(v) not in labels]
    if not remaining:
        print("All videos are already labelled.")
        return labels

    print(f"\n=== LABELLING ({len(remaining)} remaining) ===")
    print("Keys:  3 = 30°   4 = 45°   6 = 60°   S = skip   Q = quit & save\n")

    win = "Label  —  3=30°  4=45°  6=60°  S=skip  Q=quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    for i, v in enumerate(remaining):
        name = os.path.basename(v)

        # Grab a frame from 30% into the video (pipe usually visible)
        cap = cv2.VideoCapture(v)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(total * 0.30)))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"  Cannot read {name}, skipping.")
            continue

        # Overlay instructions
        disp = frame.copy()
        cv2.putText(disp, f"{i+1}/{len(remaining)}  {name}",
                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(disp, "3=30  4=45  6=60  S=skip  Q=quit",
                    (8, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow(win, disp)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord("3"):
                labels[name] = 30
                print(f"  {name} → 30°")
                break
            elif key == ord("4"):
                labels[name] = 45
                print(f"  {name} → 45°")
                break
            elif key == ord("6"):
                labels[name] = 60
                print(f"  {name} → 60°")
                break
            elif key in (ord("s"), ord("S")):
                print(f"  {name} → skipped")
                break
            elif key in (ord("q"), ord("Q")):
                cv2.destroyAllWindows()
                with open(LABELS_FILE, "w") as f:
                    json.dump(labels, f, indent=2)
                print(f"\nProgress saved to {LABELS_FILE}. "
                      f"{len(labels)}/{len(videos)} labelled.")
                return labels

        # Save after every video so progress is never lost
        with open(LABELS_FILE, "w") as f:
            json.dump(labels, f, indent=2)

    cv2.destroyAllWindows()
    labelled = sum(1 for v in labels.values() if v in (30, 45, 60))
    print(f"\nDone. {labelled}/{len(videos)} videos labelled → {LABELS_FILE}")
    return labels


# ── Calibration ────────────────────────────────────────────────────────────────

_clicks = []

def _on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        _clicks.append((x, y))

def _wait_for_n_clicks(window, frame, n, prompt, prev_points=()):
    """Display frame, wait for n clicks, return them."""
    # Drain any queued click events from the previous step
    _clicks.clear()
    for _ in range(15):
        cv2.waitKey(20)
    _clicks.clear()
    print(f"\n  {prompt}")
    while len(_clicks) < n:
        disp = frame.copy()
        for p in list(prev_points) + _clicks:
            cv2.circle(disp, p, 6, (0, 255, 0), -1)
        cv2.putText(disp, prompt, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(window, disp)
        cv2.waitKey(20)
    return list(_clicks)


def calibrate(reference_video):
    """
    One-time interactive calibration. Opens the first frame of a video,
    asks you to click reference points, then saves calibration.json.
    """
    print("\n=== CALIBRATION ===")
    cap = cv2.VideoCapture(reference_video)
    # Seek to a frame where the tank is clearly visible (10% into video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 10))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        sys.exit(f"Cannot read frame from {reference_video}")

    win = "CALIBRATION  —  follow the prompts below"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, _on_click)

    # ── Step 1: two ruler marks ──────────────────────────────────────────────
    pts = _wait_for_n_clicks(
        win, frame, 2,
        "Step 1/3: Click TWO ruler marks (white lines) — top mark first, then lower mark"
    )
    real_cm   = float(input("  How many cm apart are those two marks? ").strip())
    dy_px     = abs(pts[1][1] - pts[0][1])
    px_per_cm = dy_px / real_cm
    print(f"  Scale: {px_per_cm:.2f} px / cm")

    # ── Step 2: left and right edges of tank ─────────────────────────────────
    left_pts = _wait_for_n_clicks(
        win, frame, 1,
        "Step 2/5: Click the LEFT EDGE of the tank glass",
        prev_points=pts
    )
    x_tank_left = left_pts[0][0]
    print(f"  Tank left edge:  x = {x_tank_left} px")

    right_pts = _wait_for_n_clicks(
        win, frame, 1,
        "Step 3/5: Click the RIGHT EDGE of the tank glass",
        prev_points=pts + left_pts
    )
    x_tank_right = right_pts[0][0]
    print(f"  Tank right edge: x = {x_tank_right} px")

    # ── Step 4: water surface ────────────────────────────────────────────────
    surf = _wait_for_n_clicks(
        win, frame, 1,
        "Step 4/5: Click a point ON the water surface (inside the tank)",
        prev_points=pts + left_pts + right_pts
    )
    y_surface = surf[0][1]
    print(f"  Water surface: y = {y_surface} px")

    # ── Step 5: tank floor ───────────────────────────────────────────────────
    floor_pts = _wait_for_n_clicks(
        win, frame, 1,
        "Step 5/5: Click a point ON the tank floor",
        prev_points=pts + left_pts + right_pts + surf
    )
    y_bottom = floor_pts[0][1]
    print(f"  Tank floor:    y = {y_bottom} px")

    cv2.destroyAllWindows()

    calib = {
        "px_per_cm":    round(px_per_cm, 4),
        "x_tank_left":  x_tank_left,
        "x_tank_right": x_tank_right,
        "y_surface_px": y_surface,
        "y_bottom_px":  y_bottom,
        "note": "horizontal scale assumed equal to vertical scale (ruler)"
    }
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"\nCalibration saved to {CALIBRATION_FILE}")
    return calib


# ── Water surface auto-detection ──────────────────────────────────────────────

def detect_water_surface(frame, x_tank_left, x_tank_right, y_hint, search_range_px=120):
    """
    Find the water surface y-coordinate in a frame by locating the row with the
    strongest horizontal edge (air-water interface) near the calibrated y_hint.
    """
    h, w = frame.shape[:2]
    x_right  = x_tank_right if x_tank_right else w
    y_start  = max(0, y_hint - search_range_px)
    y_end    = min(h, y_hint + search_range_px)

    roi  = frame[y_start:y_end, x_tank_left:x_right]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Row-wise horizontal edge strength
    sobel     = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=15)
    row_edge  = np.abs(sobel).mean(axis=1)

    best_row = int(np.argmax(row_edge))
    return y_start + best_row


# ── Moving-object detection ────────────────────────────────────────────────────

def find_nose(current_frame, old_frame, y_surface, y_bottom, x_tank_left=0, x_tank_right=None):
    """
    Compares current_frame to old_frame (DIFF_WINDOW frames ago) to find
    what is moving. Returns (x, y) of the lowest moving point between the
    water surface and the floor, or None if nothing found.

    Using temporal diff means:
      - Static reflections → ignored (same in both frames)
      - Moving pipe → detected
      - Water ripples → small contours, filtered by MIN_AREA_PX
    """
    diff = cv2.absdiff(current_frame, old_frame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY)

    # Clean up noise
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)

    # Only look inside the tank (bounded by surface/floor and left/right glass walls)
    x_right = x_tank_right if x_tank_right is not None else mask.shape[1]
    roi = np.zeros_like(mask)
    roi[y_surface:y_bottom, x_tank_left:x_right] = mask[y_surface:y_bottom, x_tank_left:x_right]

    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    # Largest contour = pipe body (reflections are usually smaller or fragmented)
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_AREA_PX:
        return None, None

    # Use centroid — more robust than lowest point, which gets pulled down by reflections
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None, None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    left_x = int(c[:, 0, 0].min())   # leftmost x of contour = pipe nose at entry
    return (cx, cy), left_x


def find_pipe_above_water(current_frame, bg_frame, y_top, y_surface, x_left, x_right):
    """
    Looks for the pipe on the ramp ABOVE the water surface.
    Returns the leftmost x of the largest moving contour, or None.
    Used to anchor entry detection — if the pipe is heading toward x=1000 above
    water, a false ripple blob at x=850 will be rejected.
    """
    diff = cv2.absdiff(current_frame, bg_frame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    roi = np.zeros_like(mask)
    roi[y_top:y_surface, x_left:x_right] = mask[y_top:y_surface, x_left:x_right]
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_AREA_PX:
        return None
    return int(c[:, 0, 0].min())   # leftmost x = pipe nose on ramp


# ── Single-video processing ────────────────────────────────────────────────────

def process_video(video_path, calib, debug=False):
    """
    Returns dict {displacement_cm, x_entry_px, x_contact_px} or None on failure.

    Entry  = first frame the nose is detected below the water surface.
    Contact = first frame the nose is within CONTACT_TOL_CM of the floor,
              OR the last frame the nose was detected if it disappears near floor.
    """
    px_per_cm   = calib["px_per_cm"]
    x_tank_left  = calib.get("x_tank_left", 0)
    x_tank_right = calib.get("x_tank_right", None)
    y_surface    = calib["y_surface_px"]
    y_bottom     = calib["y_bottom_px"]
    tol_px       = int(CONTACT_TOL_CM * px_per_cm)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  SKIP: cannot open")
        return None

    # Use first frame as static background — tank is empty while pipe is still on ramp
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return None

    # Auto-detect water surface per video (overrides calibrated value)
    y_surface = detect_water_surface(first_frame, x_tank_left, x_tank_right,
                                     y_hint=y_surface, search_range_px=120)

    # Skip a few frames so first_frame is captured cleanly before processing
    START_FRAME = 10
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

    x_entry           = None
    x_contact         = None
    last_nose         = None
    y_history         = []    # recent centroid y-positions for velocity detection
    in_water           = False # True once pipe is confirmed below surface
    entry_shown        = False # True once we have paused on the detection frame
    surface_zone_hits  = 0    # consecutive frames with valid nose in surface zone

    # Pipe always enters in the right portion of the tank — require entry x to be
    # at least 40% of tank width from the left wall.  This blocks water-ripple
    # false positives at the left wall (x≈850) while keeping all real entries (x≈1000).
    tank_w_full = (x_tank_right if x_tank_right else 1920) - x_tank_left
    min_entry_x = x_tank_left + int(tank_w_full * 0.20)
    max_entry_x = x_tank_left + int(tank_w_full * 0.40)

    if debug:
        cv2.namedWindow("Debug", cv2.WINDOW_NORMAL)

    tank_h = y_bottom - y_surface   # tank height in pixels

    frame_idx = START_FRAME
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Before entry restrict search to left 65% of tank to ignore right-side disturbances
        if x_entry is None:
            tank_w   = (x_tank_right if x_tank_right else frame.shape[1]) - x_tank_left
            x_search = x_tank_left + int(tank_w * 0.65)
        else:
            x_search = x_tank_right
        nose, nose_left_x = find_nose(frame, first_frame, y_surface, y_bottom, x_tank_left, x_search)

        should_break = False

        if nose:
            nx, ny = nose

            # ── Entry: prefer detection near surface; fall back to first in-water detection
            if x_entry is None and ny >= y_surface:
                candidate_x = nose_left_x + 20
                # Reject detections in the left 40% of tank — those are ripple
                # false positives, not the pipe (pipe always enters right of centre)
                if candidate_x < min_entry_x or candidate_x > max_entry_x:
                    surface_zone_hits = 0
                    pass  # out of valid entry range
                elif y_surface <= ny <= y_surface + 120:
                    # Require 2 consecutive frames to rule out single-frame splashes
                    surface_zone_hits += 1
                    if surface_zone_hits >= 2:
                        x_entry = candidate_x
                        if debug:
                            print(f"  f{frame_idx}: ACCEPTED (surface zone) cand={candidate_x} depth={ny - y_surface}px")
                elif last_nose is None or ny >= last_nose[1] + 12:
                    # Fallback: fast-moving detection deeper in tank
                    surface_zone_hits = 0
                    x_entry = candidate_x
                    if debug:
                        print(f"  f{frame_idx}: ACCEPTED (fallback) cand={candidate_x} depth={ny - y_surface}px")

            last_nose = nose
            y_history.append(ny)
            if len(y_history) > 12:
                y_history.pop(0)

            if x_entry is not None and x_contact is None:
                # Method 1: centroid within tolerance of floor
                if ny >= y_bottom - tol_px:
                    x_contact = nx
                    should_break = True

                # Method 2: pipe has stopped moving downward (velocity ≈ 0)
                # Only trigger in bottom 40% of tank to avoid mid-water false positives
                if (not should_break
                        and len(y_history) == 12
                        and ny > y_surface + tank_h * 0.6):
                    dy = y_history[-1] - y_history[0]
                    if abs(dy) < 4:
                        x_contact = nx
                        should_break = True

        else:
            surface_zone_hits = 0
            # Detection lost — if last position was near floor, use it
            if (x_entry is not None
                    and x_contact is None
                    and last_nose is not None):
                lx, ly = last_nose
                if ly >= y_bottom - int(6 * px_per_cm):
                    x_contact = lx
                    should_break = True

        # ── Debug overlay (after processing so x_entry is current) ─────────
        if debug:
            disp = frame.copy()
            h, w = disp.shape[:2]
            cv2.line(disp, (0, y_surface), (w, y_surface), (255, 200, 0), 1)
            cv2.line(disp, (0, y_bottom),  (w, y_bottom),  (0, 80, 255),  1)
            cv2.line(disp, (0, y_bottom - tol_px), (w, y_bottom - tol_px),
                     (0, 80, 255), 1)
            if nose:
                cv2.circle(disp, nose, 7, (0, 255, 0), -1)
            if x_entry is not None:
                cv2.circle(disp, (x_entry, y_surface), 9, (0, 165, 255), 2)
            if last_nose:
                cv2.circle(disp, last_nose, 5, (200, 200, 0), 1)
            status = f"frame {frame_idx}  entry={'set' if x_entry else '--'}"
            cv2.putText(disp, status, (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.imshow("Debug", disp)
            if x_entry is not None and not entry_shown:
                cv2.putText(disp, "ENTRY DETECTED -- press any key to continue",
                            (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                while True:
                    cv2.imshow("Debug", disp)
                    if cv2.waitKey(30) & 0xFF != 255:
                        break
                entry_shown = True
            elif x_entry is not None:
                if cv2.waitKey(80) & 0xFF == ord("q"):
                    break
            else:
                cv2.waitKey(1)   # fast-forward before entry

        if should_break:
            break

    cap.release()
    if debug:
        cv2.putText(disp, "END — press any key to close", (8, disp.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Debug", disp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if x_entry is None:
        print(f"  WARN: water entry not detected")
        return None
    if x_contact is None:
        # Use last known nose position as best guess
        if last_nose is not None:
            x_contact = last_nose[0]
            print(f"  WARN: floor contact not confirmed — using last known position")
        else:
            print(f"  WARN: floor contact not detected")
            return None

    displacement_cm = abs(x_contact - x_entry) / px_per_cm
    return {
        "displacement_cm": round(displacement_cm, 2),
        "x_entry_px":      x_entry,
        "x_contact_px":    x_contact,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def find_videos():
    seen, videos = set(), []
    for ext in ["*.mp4", "*.MP4", "*.mov", "*.MOV", "*.avi", "*.AVI"]:
        for v in glob.glob(ext):
            key = v.lower()
            if key not in seen:
                seen.add(key)
                videos.append(v)
    return sorted(videos)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label",       action="store_true",
                        help="Tag each video with its ramp angle (run this first)")
    parser.add_argument("--calibrate",   action="store_true",
                        help="Run calibration and exit")
    parser.add_argument("--recalibrate", action="store_true",
                        help="Redo calibration even if calibration.json exists")
    parser.add_argument("--debug",       action="store_true",
                        help="Show frame-by-frame detection window")
    parser.add_argument("--video",       metavar="FILE",
                        help="Process only this one video (for testing)")
    args = parser.parse_args()

    videos = find_videos()
    if not videos:
        sys.exit("No video files found in current directory.\n"
                 "cd into your video folder first, then run this script.")

    print(f"Found {len(videos)} video(s) in current directory.")

    # ── Label mode ────────────────────────────────────────────────────────────
    if args.label:
        label_videos(videos)
        return

    # ── Load or create calibration ────────────────────────────────────────────
    if os.path.exists(CALIBRATION_FILE) and not args.recalibrate and not args.calibrate:
        with open(CALIBRATION_FILE) as f:
            calib = json.load(f)
        print(f"Loaded calibration: {calib['px_per_cm']:.2f} px/cm  |  "
              f"surface y={calib['y_surface_px']}  |  floor y={calib['y_bottom_px']}")
    else:
        calib = calibrate(videos[0])

    if args.calibrate:
        return  # calibrate-only run, don't process videos

    # ── Load labels ───────────────────────────────────────────────────────────
    labels = {}
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE) as f:
            labels = json.load(f)
        labelled = sum(1 for v in labels.values() if v in (30, 45, 60))
        print(f"Loaded {labelled}/{len(videos)} angle labels from {LABELS_FILE}")
    else:
        print(f"No {LABELS_FILE} found — angle_deg column will be empty.")
        print(f"Run with --label first to tag each video.")

    # ── Process videos ────────────────────────────────────────────────────────
    targets = [args.video] if args.video else videos
    rows    = []

    for i, v in enumerate(targets):
        name  = os.path.basename(v)
        angle = labels.get(name, "")
        print(f"[{i+1:3d}/{len(targets)}] {name}  angle={angle or '?'}", end="  ")
        sys.stdout.flush()
        result = process_video(v, calib, debug=args.debug)
        if result:
            print(f"→ {result['displacement_cm']:.2f} cm")
        else:
            print("→ ERROR")
        rows.append({
            "filename":        name,
            "angle_deg":       angle,
            "displacement_cm": result["displacement_cm"] if result else "ERROR",
            "x_entry_px":      result["x_entry_px"]   if result else "",
            "x_contact_px":    result["x_contact_px"] if result else "",
        })

    # ── Write CSV (skipped for single-video debug runs) ───────────────────────
    ok = sum(1 for r in rows if r["displacement_cm"] not in ("ERROR", ""))
    print(f"\n{'─'*50}")
    print(f"Processed {ok}/{len(rows)} videos successfully.")
    if args.video:
        print("Single-video debug run — results.csv not modified.")
    else:
        fieldnames = ["filename", "angle_deg", "displacement_cm",
                      "x_entry_px", "x_contact_px"]
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Results saved to: {OUTPUT_CSV}")
    print(f"\nNext steps:")
    print(f"  1. Open {OUTPUT_CSV}")
    print(f"  2. Fill in the 'angle_deg' column for each video (45, 60, or 75)")
    print(f"  3. Any rows showing ERROR — review with: "
          f"python tools\\analyze_videos.py --debug --video <filename>")
    print(f"\nNote: horizontal scale = vertical scale (from ruler).")
    print(f"Camera angle introduces a small systematic error — mention as a limitation.")


if __name__ == "__main__":
    main()
