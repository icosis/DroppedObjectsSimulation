"""
measure_entry_angle.py
======================
Detects the axis angle of the pipe at the exact frame its nose touches
the water surface, then compares it to the theoretical ramp angle plus
the pitch-down prediction.

Usage (run from the folder containing calibration.json):
    python tools/measure_entry_angle.py --video "Dropped Object Folder\C0051.MP4" --ramp 30 --debug
    python tools/measure_entry_angle.py --video "Dropped Object Folder\C0051.MP4" --ramp 30

Requires calibration.json in the current directory.
"""

import cv2
import numpy as np
import argparse
import sys
import os
import json

# ── Tuning ────────────────────────────────────────────────────────────────────
THRESHOLD        = 12    # pixel brightness change to count as motion
MIN_AREA_PX      = 300   # minimum contour area to be the pipe
MIN_ASPECT       = 2.5   # pipe must be ≥2.5× longer than wide (filters background blobs)
DIFF_WINDOW      = 5     # rolling buffer size — compare frame t to frame t-N for contour accuracy
CONSEC_REQ       = 2     # consecutive frames with pipe-like detection before accepting
SURFACE_ZONE_PX  = 60    # only lock in last_above when nose is within this many px of surface
MIN_DOWNWARD_PX  = 3     # centroid must move at least this many px downward per frame
TRIGGER_MARGIN   = 20    # fire entry trigger when blob nose is within this many px of surface
ANGLE_TOL        = 25    # blob angle must be within this many degrees of the ramp angle

CALIBRATION_FILE = "calibration.json"


# ── Physics: theoretical pitch-down angle ─────────────────────────────────────
def theoretical_pitch_down(ramp_deg, mu=0.20, release_height=0.10,
                            L=0.0762, g=9.81):
    """
    Δφ = g * L * cos(θ) / (4 * v²)
    Returns (delta_phi_deg, v_exit_m_s).
    """
    theta  = np.radians(ramp_deg)
    a_ramp = g * (np.sin(theta) - mu * np.cos(theta))
    if a_ramp <= 0:
        return 0.0, 0.0
    h        = release_height - (L / 2.0) * np.sin(theta)
    ramp_len = h / np.sin(theta)
    v        = np.sqrt(2.0 * a_ramp * ramp_len)
    delta    = np.degrees(g * L * np.cos(theta) / (4.0 * v**2))
    return delta, v


# ── Utilities ─────────────────────────────────────────────────────────────────
def load_calibration():
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE) as f:
            return json.load(f)
    return None


def detect_surface(frame, x_left, x_right, y_hint, band=120):
    """Locate water surface row via peak horizontal edge strength."""
    h, w  = frame.shape[:2]
    xr    = x_right if x_right else w
    y0    = max(0, y_hint - band)
    y1    = min(h, y_hint + band)
    roi   = frame[y0:y1, x_left:xr]
    gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    sob   = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=15)
    row   = int(np.argmax(np.abs(sob).mean(axis=1)))
    return y0 + row


def get_motion_contour(frame, ref_frame, y_top, y_bot, x_left, x_right):
    """
    Returns the largest motion contour within the given region, or None.
    """
    diff  = cv2.absdiff(frame, ref_frame)
    gray  = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY)
    k     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)

    roi   = np.zeros_like(mask)
    roi[y_top:y_bot, x_left:x_right] = mask[y_top:y_bot, x_left:x_right]

    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    candidates = []
    for c in contours:
        if cv2.contourArea(c) < MIN_AREA_PX:
            continue
        _, (w, h), _ = cv2.minAreaRect(c)
        if w == 0 or h == 0:
            continue
        if max(w, h) / min(w, h) >= MIN_ASPECT:
            candidates.append((cv2.contourArea(c), c))

    if not candidates:
        return None
    _, best = max(candidates, key=lambda x: x[0])
    return best


def fit_angle(contour):
    """
    Fit a line through the contour. Returns angle below horizontal (degrees).
    """
    if contour is None or len(contour) < 5:
        return None
    vx, vy, _, _ = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
    return float(np.degrees(np.arctan2(abs(float(vy[0])), abs(float(vx[0])))))


def draw_overlay(frame, contour, angle, y_surface, frame_idx, label=""):
    """Draw detection overlay on a copy of frame. Returns the annotated copy."""
    disp = frame.copy()
    h, w = disp.shape[:2]

    # Water surface line
    cv2.line(disp, (0, y_surface), (w, y_surface), (0, 200, 255), 2)
    cv2.putText(disp, "water surface", (8, y_surface - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

    if contour is not None:
        cv2.drawContours(disp, [contour], -1, (0, 255, 80), 2)

    if angle is not None and contour is not None:
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            arm = 100
            dx  = int(arm * np.cos(np.radians(angle)))
            dy  = int(arm * np.sin(np.radians(angle)))
            tip_a = (cx + dx, cy + dy)
            tip_b = (cx - dx, cy - dy)
            nose_tip = tip_a if tip_a[1] >= tip_b[1] else tip_b
            tail_tip = tip_b if nose_tip == tip_a else tip_a
            cv2.line(disp, tail_tip, nose_tip, (0, 100, 255), 3)
            cv2.circle(disp, nose_tip, 9, (0, 0, 255), -1)
            cv2.putText(disp, f"{angle:.1f} deg", (cx + 10, cy - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)

    status = f"frame {frame_idx}  {label}"
    cv2.putText(disp, status, (8, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return disp


# ── Main ──────────────────────────────────────────────────────────────────────
def measure_entry_angle(video_path, calib, ramp_deg=None, debug=False):
    """
    Scan the video for the frame where the pipe nose first crosses
    the water surface. Returns the pipe axis angle at that frame.
    """
    x_left    = calib.get("x_tank_left",  0)
    x_right   = calib.get("x_tank_right", None)
    y_surface = calib["y_surface_px"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Cannot open: {video_path}")

    ret, bg = cap.read()
    if not ret:
        cap.release()
        return None

    h_frame, w_frame = bg.shape[:2]
    xr = x_right if x_right else w_frame

    # Refine surface y for this specific video
    y_surface = detect_surface(bg, x_left, xr, y_hint=y_surface)
    print(f"  Water surface detected at y = {y_surface} px")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 10)

    frame_buffer = []
    frame_idx    = 10
    entry_angle  = None
    entry_frame  = None

    last_above_frame   = None
    last_above_contour = None
    last_above_angle   = None
    above_consec       = 0    # consecutive frames with valid pipe-like blob above water
    prev_centroid_y    = None # centroid y from previous frame, for velocity check

    if debug:
        cv2.namedWindow("Entry Angle", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        frame_buffer.append(frame.copy())
        if len(frame_buffer) > DIFF_WINDOW:
            frame_buffer.pop(0)
        if len(frame_buffer) < 2:
            continue
        ref = frame_buffer[0]

        y_top = max(0, y_surface - 400)
        y_bot = y_surface + 10
        c = get_motion_contour(frame, ref, y_top, y_bot, x_left, xr)

        if c is not None:
            nose_y = int(c[:, 0, 1].max())

            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx_blob = int(M["m10"] / M["m00"])
            cy      = int(M["m01"] / M["m00"])

            # Reject if centroid is outside tank walls
            if not (x_left <= cx_blob <= xr):
                above_consec = 0
                if debug and cv2.waitKey(30) & 0xFF == ord("q"):
                    break
                continue

            # Reject if blob angle is implausible for the given ramp
            if ramp_deg is not None:
                blob_ang = fit_angle(c)
                if blob_ang is None or abs(blob_ang - ramp_deg) > ANGLE_TOL:
                    above_consec = 0
                    if debug and cv2.waitKey(30) & 0xFF == ord("q"):
                        break
                    continue

            if nose_y < y_surface:
                # Require blob to be moving downward toward water
                moving_down = (prev_centroid_y is None or cy >= prev_centroid_y + MIN_DOWNWARD_PX)
                if moving_down:
                    above_consec += 1
                else:
                    above_consec = 0
                prev_centroid_y = cy

                # Only lock in when close to surface AND moving AND sustained
                if above_consec >= CONSEC_REQ and nose_y >= y_surface - SURFACE_ZONE_PX:
                    last_above_frame   = frame.copy()
                    last_above_contour = c
                    last_above_angle   = fit_angle(c)
            else:
                above_consec    = 0
                prev_centroid_y = None

            if nose_y >= y_surface - TRIGGER_MARGIN:
                # Use confirmed above-water frame if available; fall back to current frame
                display_frame   = last_above_frame   if last_above_frame   is not None else frame
                display_contour = last_above_contour if last_above_contour is not None else c
                entry_angle     = last_above_angle   if last_above_angle   is not None else fit_angle(c)
                entry_frame     = frame_idx

                if debug:
                    lbl  = "ENTRY — press any key" if last_above_frame is not None else "ENTRY (fallback) — press any key"
                    disp = draw_overlay(display_frame, display_contour, entry_angle,
                                        y_surface, entry_frame,
                                        label=lbl)
                    base     = os.path.splitext(os.path.basename(video_path))[0]
                    out_path = f"entry_angle_{base}.png"
                    cv2.imwrite(out_path, disp)
                    print(f"  Saved entry frame -> {out_path}")
                    cv2.imshow("Entry Angle", disp)
                    cv2.waitKey(0)
                break

            if debug:
                dist  = y_surface - nose_y
                label = f"nose {dist}px above surf  consec={above_consec}"
                disp  = draw_overlay(frame, c, fit_angle(c), y_surface, frame_idx, label)
                cv2.imshow("Entry Angle", disp)
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
        else:
            above_consec = 0
            if debug:
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break

    cap.release()
    if debug:
        cv2.destroyAllWindows()

    if entry_angle is None:
        print("  Could not detect pipe at water surface.")
        return None

    return {
        "angle_deg":   round(entry_angle, 1),
        "frame":       entry_frame,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Measure pipe axis angle at water entry."
    )
    parser.add_argument("--video",  required=True, metavar="FILE")
    parser.add_argument("--ramp",   type=float, default=None, metavar="DEG",
                        help="Ramp angle for theoretical comparison")
    parser.add_argument("--mu",     type=float, default=0.20,
                        help="Ramp friction coefficient (default 0.20)")
    parser.add_argument("--debug",  action="store_true",
                        help="Show detection window, pause at entry frame")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        sys.exit(f"File not found: {args.video}")

    calib = load_calibration()
    if calib is None:
        sys.exit("calibration.json not found. Run analyze_videos.py --calibrate first.")

    print(f"\nAnalysing: {args.video}")

    result = measure_entry_angle(args.video, calib, ramp_deg=args.ramp, debug=args.debug)
    if result is None:
        sys.exit("Measurement failed.")

    print(f"\n── Entry angle ───────────────────────────────────────")
    print(f"  Measured:  {result['angle_deg']}°  (frame {result['frame']})")

    if args.ramp is not None:
        delta, v = theoretical_pitch_down(args.ramp, mu=args.mu)
        predicted = args.ramp + delta
        residual  = result["angle_deg"] - predicted
        print(f"\n── Theory (ramp = {args.ramp}°, μ = {args.mu}) ──────────")
        print(f"  Ramp angle:       {args.ramp:.1f}°")
        print(f"  Pitch-down Δφ:   +{delta:.1f}°")
        print(f"  Predicted entry:  {predicted:.1f}°")
        print(f"  Measured entry:   {result['angle_deg']}°")
        print(f"  Residual:         {residual:+.1f}°")
        print(f"  Exit speed:       {v * 100:.1f} cm/s")


if __name__ == "__main__":
    main()
