"""
trace_trajectory.py
===================
Records the full underwater path of the cylinder — from water entry to first
floor contact — and produces two figures per video:

  1. A traced still image  : the path drawn on the contact frame  (trace_<name>.png)
  2. A trajectory plot in cm: horizontal displacement vs depth      (traj_<name>.png)

It reuses the detection logic in analyze_videos.py (find_nose, water-surface
detection, entry acceptance) so the recorded entry point matches results.csv.

Run from the video folder (where calibration.json lives):

    # One video, both figures:
    python tools/trace_trajectory.py --video C0051.MP4

    # All labelled videos, overlaid onto one plot coloured by ramp angle:
    python tools/trace_trajectory.py

Requires calibration.json (and labels.json for batch mode) in the current
directory, plus matplotlib (pip install matplotlib).
"""

import cv2
import numpy as np
import json
import os
import sys
import glob
import argparse

# Force UTF-8 stdout so redirected output doesn't crash on arrows/box chars
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(__file__))
from analyze_videos import (
    detect_water_surface, find_nose, MIN_AREA_PX, THRESHOLD,
    CALIBRATION_FILE, LABELS_FILE,
)
from measure_entry_angle import measure_entry_angle

# Colour per ramp angle for the overlaid batch plot (matplotlib names)
ANGLE_COLOR = {30: "tab:red", 45: "tab:green", 60: "tab:blue"}
# BGR colours for the on-frame trace overlay
TRACE_BGR   = (0, 165, 255)   # orange path
ENTRY_BGR   = (0, 255, 255)   # yellow entry marker
CONTACT_BGR = (0, 80, 255)    # red contact marker
RULER_BGR   = (255, 255, 255) # ruler ticks and labels
HALF_LENGTH_CM = 7.62 / 2.0   # pipe half-length: CoM sits this far up-axis of the nose
COM_BGR     = (255, 120, 0)   # centre-of-mass marker (blue)
DEBUG_WIN   = "Trajectory (q = quit)"


def dashed_vline(disp, x, y0, y1, color=RULER_BGR, dash=7, gap=7, thickness=1):
    """Vertical dashed line from y0 down to y1."""
    for y in range(int(y0), int(y1), dash + gap):
        cv2.line(disp, (int(x), y), (int(x), min(y + dash, int(y1))),
                 color, thickness)


def draw_rulers(disp, x_left, x_right, y_surface, y_bottom, px_per_cm,
                x_origin=None):
    """
    Draw cm rulers on the tank: a vertical ruler down the left wall (depth
    below the surface) and — once the entry point is known — a horizontal
    ruler along the floor whose 0 sits at the entry x, so the label under the
    pipe's resting spot reads its horizontal displacement directly.
    Ticks every 1 cm, numbered every 5 cm.
    """
    h, w = disp.shape[:2]
    xr = x_right if x_right else w
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Vertical ruler: depth below surface, drawn just inside the left wall
    depth_cm = int((y_bottom - y_surface) / px_per_cm)
    cv2.line(disp, (x_left, y_surface), (x_left, y_bottom), RULER_BGR, 1)
    for d in range(0, depth_cm + 1):
        y = int(y_surface + d * px_per_cm)
        major = (d % 5 == 0)
        tick = 16 if major else 8
        cv2.line(disp, (x_left, y), (x_left + tick, y), RULER_BGR, 2 if major else 1)
        if major and d > 0:
            cv2.putText(disp, str(d), (x_left + tick + 4, y + 5),
                        font, 0.45, RULER_BGR, 1, cv2.LINE_AA)

    # Horizontal ruler: displacement from the ENTRY point, along the floor.
    # Only drawn once entry is known (x_origin set), so it appears in the
    # video at the moment the pipe enters the water.
    if x_origin is None:
        return
    width_cm = int((xr - x_origin) / px_per_cm)
    cv2.line(disp, (int(x_origin), y_bottom), (xr, y_bottom), RULER_BGR, 1)
    # Dashed drop-line from the entry point down to the ruler's zero
    dashed_vline(disp, x_origin, y_surface, y_bottom)
    # Ticks: 0.5 cm minor, 1 cm medium, 5 cm major with labels
    for half in range(0, width_cm * 2 + 1):
        d = half / 2.0
        x = int(x_origin + d * px_per_cm)
        if half % 10 == 0:            # 5 cm — major, labelled
            tick, thick = 16, 2
            cv2.putText(disp, str(int(d)), (x - 8, y_bottom - tick - 6),
                        font, 0.45, RULER_BGR, 1, cv2.LINE_AA)
        elif half % 2 == 0:           # 1 cm — medium
            tick, thick = 9, 1
        else:                         # 0.5 cm — minor
            tick, thick = 5, 1
        cv2.line(disp, (x, y_bottom - tick), (x, y_bottom), RULER_BGR, thick)


# The global/canonical calibration lives in the project's data/ folder, next to
# the tools/ directory: <repo>/tools/trace_trajectory.py  ->  <repo>/data/calibration.json
GLOBAL_CALIBRATION = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", CALIBRATION_FILE))


def find_calibration(explicit=None):
    """
    Locate the calibration file, preferring the *global* project one.

    Resolution order:
      1. An explicit --calib path if given.
      2. The global project calibration in <repo>/data/calibration.json.
      3. Walk up from the current directory as a fallback.
    """
    if explicit:
        if not os.path.exists(explicit):
            sys.exit(f"--calib file not found: {explicit}")
        return explicit

    if os.path.exists(GLOBAL_CALIBRATION):
        return GLOBAL_CALIBRATION

    d = os.path.abspath(os.getcwd())
    while True:
        cand = os.path.join(d, CALIBRATION_FILE)
        if os.path.exists(cand):
            return cand
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def smooth_path(points, passes=3):
    """
    Smooth a tracked polyline while staying as close as possible to the
    measured centroids.

    Stage 1 — a light 1/4·1/2·1/4 weighted average on interior points damps
    single-frame centroid tugs (wake/reflection pulling the blob centre for
    one frame). Endpoints (entry, contact) are pinned exactly.

    Stage 2 — Chaikin corner-cutting rounds the remaining slope kinks. Each
    pass replaces every segment with points at its 1/4 and 3/4 marks, so the
    curve always stays inside the corridor of the measured path — it cannot
    drift away from the data the way a heavy spline can.

    Works in any coordinate space (pixels or cm). Returns [(x, y), ...] floats.
    """
    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) < 3:
        return pts

    # Stage 0 — median-of-3 on interior points. A wake/splash can yank the
    # centroid sideways for a single frame; averaging would only shrink that
    # outlier, but the median REPLACES it with a neighbouring value, so the
    # curve never follows the bad point at all. Genuine trend is unaffected
    # (for monotone stretches the median is the point itself).
    med = [pts[0]]
    for i in range(1, len(pts) - 1):
        med.append((sorted((pts[i-1][0], pts[i][0], pts[i+1][0]))[1],
                    sorted((pts[i-1][1], pts[i][1], pts[i+1][1]))[1]))
    med.append(pts[-1])
    pts = med

    sm = [pts[0]]
    for i in range(1, len(pts) - 1):
        sm.append((0.25 * pts[i-1][0] + 0.5 * pts[i][0] + 0.25 * pts[i+1][0],
                   0.25 * pts[i-1][1] + 0.5 * pts[i][1] + 0.25 * pts[i+1][1]))
    sm.append(pts[-1])

    for _ in range(passes):
        out = [sm[0]]
        for (ax, ay), (bx, by) in zip(sm[:-1], sm[1:]):
            out.append((0.75 * ax + 0.25 * bx, 0.75 * ay + 0.25 * by))
            out.append((0.25 * ax + 0.75 * bx, 0.25 * ay + 0.75 * by))
        out.append(sm[-1])
        sm = out
    return sm


def smooth_px(points, passes=3):
    """smooth_path, rounded to int32 pixel coordinates ready for cv2.polylines."""
    return np.array(smooth_path(points, passes), dtype=np.int32)


def load_calibration(explicit=None):
    path = find_calibration(explicit)
    if path is None:
        sys.exit(f"{CALIBRATION_FILE} not found. Run analyze_videos.py --calibrate first.")
    with open(path) as f:
        calib = json.load(f)
    print(f"Using calibration: {path}")
    return calib


# ── Core: record the underwater path ────────────────────────────────────────────

def dashed_segment(disp, p0, p1, color, dash=10, gap=8, thickness=1):
    """Dashed straight line between two arbitrary points."""
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    length = max(1.0, np.hypot(x1 - x0, y1 - y0))
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    s = 0.0
    while s < length:
        e = min(s + dash, length)
        cv2.line(disp, (int(x0 + ux * s), int(y0 + uy * s)),
                 (int(x0 + ux * e), int(y0 + uy * e)), color, thickness,
                 cv2.LINE_AA)
        s = e + gap


def draw_entry_angle(disp, x0, y_surface, angle_deg, y_bottom=None,
                     px_per_cm=None, x_ref=None):
    """
    Short indicator line at the measured pipe-axis angle, pivoted on entry.
    If y_bottom is given, the line is extended (dashed) to the tank floor and
    the straight-line landing point — where the pipe would land with no
    hydrodynamic deflection — is marked.
    """
    if angle_deg is None:
        return
    arm = 90
    dx = int(arm * np.cos(np.radians(angle_deg)))
    dy = int(arm * np.sin(np.radians(angle_deg)))
    # Line through the entry point along the pipe axis (nose down-right)
    cv2.line(disp, (x0 - dx, y_surface - dy), (x0 + dx, y_surface + dy),
             ENTRY_BGR, 2, cv2.LINE_AA)
    cv2.putText(disp, f"Entry {angle_deg:.1f} deg",
                (x0 + dx + 8, y_surface + dy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, ENTRY_BGR, 2, cv2.LINE_AA)


    # Straight-line projection to the floor: expected landing with no water
    if y_bottom is None or angle_deg <= 5:
        return
    x_land = int(x0 + (y_bottom - y_surface) / np.tan(np.radians(angle_deg)))
    if not (0 <= x_land < disp.shape[1]):
        return
    dashed_segment(disp, (x0 + dx, y_surface + dy), (x_land, y_bottom),
                   ENTRY_BGR, thickness=2)
    if px_per_cm:
        # The nose (front end) travels along this axis line; with no water it
        # reaches floor depth exactly at the line/floor intersection — the
        # expected front-end landing, comparable to the actual front end.
        cv2.circle(disp, (x_land, y_bottom), 10, COM_BGR, -1)
        cv2.circle(disp, (x_land, y_bottom), 10, (255, 255, 255), 2)
        ref_b = x_ref if x_ref is not None else x0
        cv2.putText(disp, f"Expected {(x_land - ref_b) / px_per_cm:.2f} cm",
                    (x_land + 16, y_bottom + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COM_BGR, 2, cv2.LINE_AA)


def front_end_point(frame, first_frame, y_surface, y_bottom, x_left, x_right):
    """
    Rightmost point of the largest moving contour — the pipe's LEADING (front)
    end in the direction of travel.  Used at the contact frame so displacement
    is measured front-end-to-front-end (nose pierce -> leading tip at floor).
    Returns (x, y) or None.
    """
    diff = cv2.absdiff(frame, first_frame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    xr = x_right if x_right else mask.shape[1]
    roi = np.zeros_like(mask)
    roi[y_surface:y_bottom, x_left:xr] = mask[y_surface:y_bottom, x_left:xr]
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_AREA_PX:
        return None
    idx = int(c[:, 0, 0].argmax())          # rightmost = leading edge
    return (int(c[idx, 0, 0]), int(c[idx, 0, 1]))


def trace_video(video_path, calib, debug=False, video_out=None,
                entry_angle=None, entry_frame=None, entry_anchor_x=None):
    """
    Walk the video, detect the cylinder centroid every frame, and record the
    path from water entry to floor contact.

    Returns a dict:
        path_px      : list of (x, y) centroid pixels, entry -> contact
        x_entry_px   : accepted entry x (nose at surface)
        y_surface_px : detected surface row for this video
        y_bottom_px  : tank floor row
        px_per_cm    : scale
        frame_img    : BGR frame at contact (for the trace overlay)
        x_contact_px : contact x
    or None if entry was never detected.
    """
    px_per_cm    = calib["px_per_cm"]
    x_tank_left  = calib.get("x_tank_left", 0)
    x_tank_right = calib.get("x_tank_right", None)
    y_surface    = calib["y_surface_px"]
    y_bottom     = calib["y_bottom_px"]
    tol_px       = int(3.0 * px_per_cm)   # CONTACT_TOL_CM in analyze_videos

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("  SKIP: cannot open")
        return None

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return None

    y_surface = detect_water_surface(first_frame, x_tank_left, x_tank_right,
                                     y_hint=y_surface, search_range_px=120)

    # Optional annotated video showing the trajectory trail grow frame-by-frame
    writer = None
    if video_out:
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30
        h_f, w_f = first_frame.shape[:2]
        writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"),
                                 max(1.0, fps / 3.0), (w_f, h_f))

    entry_paused = False
    if debug:
        cv2.namedWindow(DEBUG_WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(DEBUG_WIN, 1280, 720)

    START_FRAME = 10
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

    x_entry           = None
    x_contact         = None
    last_nose         = None
    y_history         = []
    surface_zone_hits = 0
    prev_zone_y       = None
    path_px           = []          # committed trajectory (only drawn after entry)
    pending_px        = []          # in-water points seen before entry is confirmed
    contact_frame_img = first_frame # fallback if we never break

    tank_w_full = (x_tank_right if x_tank_right else 1920) - x_tank_left
    min_entry_x = x_tank_left + int(tank_w_full * 0.20)
    max_entry_x = x_tank_left + int(tank_w_full * 0.40)
    tank_h      = y_bottom - y_surface

    CONTACT_WALL_MARGIN_PX = 40

    # CoM entry origin = the pierce point.  The pipe slides along one straight
    # axis line that crosses the surface at a single x; the nose crosses there,
    # and later the CoM crosses the surface at that SAME x (pipe half-submerged).
    # So the CoM-entry reference is the pierce anchor itself — no offset.
    # Displacement is then CoM(surface) -> centroid(floor), matching the sim.
    com_offset_px = com_offset_cm = 0.0

    def plausible_contact(cx_):
        if x_entry is not None and cx_ > x_entry + int(25 * px_per_cm):
            return False
        if x_tank_right and cx_ >= x_tank_right - CONTACT_WALL_MARGIN_PX:
            return False
        return True

    frame_idx = START_FRAME
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if x_entry is None:
            tank_w   = (x_tank_right if x_tank_right else frame.shape[1]) - x_tank_left
            x_search = x_tank_left + int(tank_w * 0.65)
        else:
            x_search = x_tank_right
        nose, nose_left_x, nose_area = find_nose(
            frame, first_frame, y_surface, y_bottom, x_tank_left, x_search)

        should_break = False

        if nose:
            nx, ny = nose

            # ── Entry acceptance (mirrors analyze_videos.process_video) ────────
            if x_entry is None and ny >= y_surface:
                candidate_x = nose_left_x + 20
                if candidate_x < min_entry_x or candidate_x > max_entry_x:
                    surface_zone_hits = 0
                    prev_zone_y = None
                elif y_surface <= ny <= y_surface + 120:
                    if nose_area >= 3000 and ny >= y_surface + 30:
                        x_entry = candidate_x
                    elif prev_zone_y is None or ny < prev_zone_y - 5:
                        surface_zone_hits = 1
                        prev_zone_y = ny
                    elif ny >= prev_zone_y + 5:
                        surface_zone_hits += 1
                        prev_zone_y = ny
                        if surface_zone_hits >= 2:
                            x_entry = candidate_x
                elif last_nose is None or ny >= last_nose[1] + 12:
                    surface_zone_hits = 0
                    prev_zone_y = None
                    x_entry = candidate_x

            # ── Record the path, but only DRAW it once entry is confirmed ───────
            # The orange trail must never appear before the pipe enters the water.
            # So plausible in-water points are buffered in pending_px; the moment
            # entry is confirmed the buffer is committed to path_px (this keeps the
            # descent for slow 30° drops that confirm entry only once deeper), and
            # from then on points append straight to path_px.
            jump_px  = int(10 * px_per_cm)
            in_water = y_surface <= ny <= y_bottom
            if not path_px and not pending_px:
                # First point of the trail = the entry origin. The splash curtain
                # spreads LEFT of the pierce point at the surface; starting the
                # trail there fakes a horizontal "glide" and inflates displacement.
                # So the origin must lie inside the physical entry window.
                plausible_x = min_entry_x <= nx <= max_entry_x
            else:
                plausible_x = (nx >= min_entry_x - int(2 * px_per_cm)
                               and (x_tank_right is None or nx <= x_tank_right))

            # Commit the buffered descent as soon as entry is confirmed.
            # Splash spreads LEFT of the pierce point, and underwater the pipe
            # only travels down-ramp (rightward) — so buffered points left of
            # the entry x are splash, not pipe.
            if x_entry is not None and not path_px and pending_px:
                path_px = [p for p in pending_px
                           if p[0] >= x_entry - int(0.5 * px_per_cm)]
                contact_frame_img = frame.copy()

            # Anchor the trail at the surface: the pipe physically pierced
            # y_surface at the pierce anchor (fall back to x_entry), but splash
            # hides the first few cm of descent — without this anchor, curves
            # appear to start mid-water.  Using the pierce anchor keeps the
            # trail, entry dot, and angle line visually coincident.
            trail_x0 = entry_anchor_x if entry_anchor_x is not None else x_entry
            if x_entry is not None and (not path_px
                                        or path_px[0][1] > y_surface + 5):
                path_px.insert(0, (trail_x0, y_surface))

            if in_water and plausible_x:
                if x_entry is not None:
                    lastp = path_px[-1] if path_px else None
                    # Skip duplicated frames (the footage repeats frames; a repeat
                    # adds a zero-motion point that only distorts the smoothing)
                    dup = (lastp is not None
                           and abs(nx - lastp[0]) <= 2 and abs(ny - lastp[1]) <= 2)
                    # Underwater the pipe only drifts down-ramp (rightward) — its
                    # horizontal velocity decays but never reverses.  A sizeable
                    # left step mid-tank means the blob latched onto the bubble
                    # wake the pipe left behind, not the pipe itself.  Near the
                    # floor the pipe may tip and settle backwards, so no check there.
                    max_x    = max(p[0] for p in path_px) if path_px else nx
                    mid_tank = ny < y_surface + tank_h * 0.6
                    wake     = mid_tank and nx < max_x - int(0.5 * px_per_cm)
                    no_jump  = lastp is None or abs(nx - lastp[0]) < jump_px
                    if not dup and not wake and no_jump:
                        path_px.append((nx, ny))
                        contact_frame_img = frame.copy()
                else:
                    # Before entry: buffer only a coherent, descending track
                    if (not pending_px
                            or (ny >= pending_px[-1][1] - 5
                                and abs(nx - pending_px[-1][0]) < jump_px)):
                        pending_px.append((nx, ny))

            last_nose = nose
            y_history.append(ny)
            if len(y_history) > 12:
                y_history.pop(0)

            if x_entry is not None and x_contact is None:
                if ny >= y_bottom - tol_px and plausible_contact(nx):
                    x_contact = nx
                    should_break = True
                if (not should_break and len(y_history) == 12
                        and ny > y_surface + tank_h * 0.6):
                    dy = y_history[-1] - y_history[0]
                    if abs(dy) < 4 and plausible_contact(nx):
                        x_contact = nx
                        should_break = True
        else:
            surface_zone_hits = 0
            prev_zone_y = None
            if x_entry is None:
                pending_px = []   # drop stale pre-entry buffer when detection is lost
            if (x_entry is not None and x_contact is None and last_nose is not None):
                lx, ly = last_nose
                if ly >= y_bottom - int(6 * px_per_cm) and plausible_contact(lx):
                    x_contact = lx
                    should_break = True

        # ── Draw the growing trail on this frame (for the video and/or window) ──
        if writer is not None or debug:
            disp = frame.copy()
            hh, ww = disp.shape[:2]
            cv2.line(disp, (0, y_surface), (ww, y_surface), ENTRY_BGR, 1)
            at_pierce = (entry_frame is not None and frame_idx >= entry_frame)
            ruler_x0 = (path_px[0][0] if path_px
                        else (entry_anchor_x if at_pierce else None))
            if ruler_x0 is not None:
                ruler_x0 = int(ruler_x0 - com_offset_px)
            draw_rulers(disp, x_tank_left, x_tank_right, y_surface, y_bottom,
                        px_per_cm, x_origin=ruler_x0)
            if len(path_px) >= 2:
                pts = smooth_px(path_px).reshape(-1, 1, 2)
                cv2.polylines(disp, [pts], False, TRACE_BGR, 2, cv2.LINE_AA)
            # The angle line stays pinned at the pierce anchor for the whole
            # video — one line only, where the nose actually crossed the surface
            angle_anchor = entry_anchor_x
            if path_px:
                if angle_anchor is None:
                    angle_anchor = path_px[0][0]
                # Entry dot pivots on the same anchor as the angle line
                cv2.circle(disp, (angle_anchor, y_surface), 8, ENTRY_BGR, 2)
                cv2.circle(disp, tuple(path_px[-1]), 7, TRACE_BGR, -1)
                # Drop-line from the current centroid to the floor ruler so
                # the running displacement can be read directly off the ticks
                dashed_vline(disp, path_px[-1][0], path_px[-1][1], y_bottom,
                             color=TRACE_BGR)
                draw_entry_angle(disp, angle_anchor, y_surface, entry_angle, y_bottom,
                                 px_per_cm, x_ref=angle_anchor - com_offset_px)
            elif at_pierce and entry_anchor_x is not None:
                # Pipe is piercing the surface but the trail hasn't started —
                # anchor the axis line where the angle pass tracked the pipe
                # (the raw underwater nose is unreliable this early: it may
                # still be latched onto slosh elsewhere in the tank)
                cv2.circle(disp, (entry_anchor_x, y_surface), 8, ENTRY_BGR, 2)
                draw_entry_angle(disp, entry_anchor_x, y_surface, entry_angle,
                                 y_bottom, px_per_cm,
                                 x_ref=entry_anchor_x - com_offset_px)
            if nose:
                cv2.circle(disp, nose, 5, (0, 255, 0), 1)   # raw detection
            cv2.putText(disp, f"frame {frame_idx}  pts={len(path_px)}", (8, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if writer is not None:
                writer.write(disp)
            if debug:
                cv2.imshow(DEBUG_WIN, disp)
                # Pause on the surface-pierce frame (known from the entry-angle
                # pass) so the centroid sits exactly on the surface line with no
                # trail yet; fall back to the entry-confirmation frame if the
                # angle pass failed.
                if entry_frame is not None:
                    pause_now = at_pierce and not entry_paused
                else:
                    pause_now = x_entry is not None and not entry_paused
                if pause_now:
                    entry_paused = True
                if cv2.waitKey(0 if pause_now else 40) & 0xFF == ord("q"):
                    break

        if should_break:
            break

    cap.release()

    # Front (leading) end of the pipe at contact — the measurement endpoint.
    front_end = None
    if path_px:
        fe = front_end_point(contact_frame_img, first_frame, y_surface, y_bottom,
                             x_tank_left, x_tank_right)
        front_end = fe if fe is not None else tuple(path_px[-1])

    def _draw_final(disp):
        """Draw the completed front-to-front annotation onto disp."""
        pts = smooth_px(path_px).reshape(-1, 1, 2)
        cv2.polylines(disp, [pts], False, TRACE_BGR, 2, cv2.LINE_AA)
        cv2.line(disp, (0, y_surface), (disp.shape[1], y_surface), ENTRY_BGR, 1)
        anchor = entry_anchor_x if entry_anchor_x is not None else path_px[0][0]
        draw_rulers(disp, x_tank_left, x_tank_right, y_surface, y_bottom,
                    px_per_cm, x_origin=int(anchor))
        cv2.circle(disp, (anchor, y_surface), 8, ENTRY_BGR, 2)
        # Red contact marker on the FLOOR axis at the leading-tip x, so it lines
        # up with the blue expected marker (both compare horizontal distance).
        land_pt = (front_end[0], y_bottom)
        cv2.line(disp, tuple(path_px[-1]), land_pt, TRACE_BGR, 2, cv2.LINE_AA)
        cv2.circle(disp, land_pt, 9, CONTACT_BGR, -1)
        draw_entry_angle(disp, anchor, y_surface, entry_angle, y_bottom,
                         px_per_cm, x_ref=anchor)
        disp_cm = abs(front_end[0] - anchor) / px_per_cm
        cv2.putText(disp, f"{disp_cm:.2f} cm", (front_end[0] + 16, y_bottom - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, TRACE_BGR, 2, cv2.LINE_AA)

    if writer is not None:
        if path_px:
            disp = contact_frame_img.copy()
            _draw_final(disp)
            for _ in range(int(max(1.0, fps))):     # hold the final frame ~1s
                writer.write(disp)
        writer.release()

    if debug:
        if path_px:
            disp = contact_frame_img.copy()
            _draw_final(disp)
            cv2.putText(disp, f"DONE  {len(path_px)} points  press any key",
                        (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(DEBUG_WIN, disp)
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    if x_entry is None:
        print("  WARN: water entry not detected")
        return None
    if x_contact is None and last_nose is not None and plausible_contact(last_nose[0]):
        x_contact = last_nose[0]

    return {
        "path_px":      path_px,
        "x_entry_px":   x_entry,
        "y_surface_px": y_surface,
        "y_bottom_px":  y_bottom,
        "px_per_cm":    px_per_cm,
        "frame_img":    contact_frame_img,
        "x_contact_px": x_contact if x_contact is not None else (path_px[-1][0] if path_px else x_entry),
        "entry_angle":  entry_angle,
        "entry_anchor_x": entry_anchor_x,
        "front_end_px": front_end,
    }


def bounce_rebound_cm(trace):
    """
    How far the pipe rebounded backwards (cm) after reaching its rightmost
    point.  A hard floor impact at steep angles kicks the pipe back left —
    those runs shouldn't be pooled with clean slide-to-rest trajectories.
    """
    path = trace["path_px"]
    if len(path) < 3:
        return 0.0
    xs = [p[0] for p in path]
    i_max = xs.index(max(xs))
    if i_max == len(xs) - 1:
        return 0.0
    return (xs[i_max] - min(xs[i_max:])) / trace["px_per_cm"]


BOUNCE_LIMIT_CM = 2.0   # exclude batch trajectories that rebound more than this


# ── Output 1: trace drawn on the frame ──────────────────────────────────────────

def save_trace_image(trace, out_path, calib=None):
    disp = trace["frame_img"].copy()
    h, w = disp.shape[:2]
    y_surf = trace["y_surface_px"]

    # Water surface line
    cv2.line(disp, (0, y_surf), (w, y_surf), ENTRY_BGR, 1)

    # cm rulers: depth on the left wall, CoM displacement on the floor.
    # Ruler zero = pierce anchor = where the CoM crosses the surface (no offset).
    if calib is not None:
        origin = trace["path_px"][0][0] if trace["path_px"] else trace["x_entry_px"]
        ang = trace.get("entry_angle")
        anchor = trace.get("entry_anchor_x")
        pivot  = anchor if anchor is not None else origin
        draw_rulers(disp, calib.get("x_tank_left", 0), calib.get("x_tank_right"),
                    y_surf, trace["y_bottom_px"], trace["px_per_cm"],
                    x_origin=int(pivot))
        draw_entry_angle(disp, pivot, y_surf, ang,
                         y_bottom=trace["y_bottom_px"],
                         px_per_cm=trace["px_per_cm"],
                         x_ref=pivot)
    # Front (leading) end of the pipe at contact = measurement endpoint
    front_end = trace.get("front_end_px")
    if front_end is None and trace["path_px"]:
        front_end = tuple(trace["path_px"][-1])

    y_bot = trace["y_bottom_px"]
    land_pt = (front_end[0], y_bot)
    if len(trace["path_px"]) >= 2:
        pivot = trace.get("entry_anchor_x") or trace["x_entry_px"]
        disp_cm = abs(front_end[0] - pivot) / trace["px_per_cm"]
        cv2.putText(disp, f"{disp_cm:.2f} cm", (front_end[0] + 16, y_bot - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, TRACE_BGR, 2, cv2.LINE_AA)

    # Smoothed curve as the path; raw measured centroids kept as small dots
    if len(trace["path_px"]) >= 2:
        pts = smooth_px(trace["path_px"]).reshape(-1, 1, 2)
        cv2.polylines(disp, [pts], False, TRACE_BGR, 2, cv2.LINE_AA)
    for p in trace["path_px"]:
        cv2.circle(disp, tuple(p), 2, TRACE_BGR, -1)

    # Entry (surface crossing) and contact markers — dot on the angle-line anchor
    entry_x = trace.get("entry_anchor_x")
    if entry_x is None:
        entry_x = trace["path_px"][0][0] if trace["path_px"] else trace["x_entry_px"]
    cv2.circle(disp, (entry_x, y_surf), 8, ENTRY_BGR, 2)
    if trace["path_px"]:
        # Red contact marker on the floor axis at the leading-tip x
        cv2.line(disp, tuple(trace["path_px"][-1]), land_pt, TRACE_BGR, 2, cv2.LINE_AA)
        cv2.circle(disp, land_pt, 9, CONTACT_BGR, -1)

    cv2.putText(disp, "Entry", (entry_x + 10, y_surf - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, ENTRY_BGR, 2)
    cv2.imwrite(out_path, disp)
    print(f"  Saved trace image → {out_path}")


# ── Output 2: trajectory plot in cm ─────────────────────────────────────────────

def path_to_cm(trace):
    """Convert pixel path to cm relative to the surface crossing (x) and surface (depth).

    Origin x is the first (shallowest) recorded point — the horizontal position where
    the nose crossed the surface — which is more robust than the formally-confirmed
    entry x for slow drops that confirm entry only once deep.
    """
    x0     = trace["path_px"][0][0]
    y_surf = trace["y_surface_px"]
    s      = trace["px_per_cm"]
    xs = [(x - x0) / s for (x, y) in trace["path_px"]]
    ys = [(y - y_surf) / s for (x, y) in trace["path_px"]]
    return xs, ys


def save_single_plot(trace, out_path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs, ys = path_to_cm(trace)
    sm = smooth_path(list(zip(xs, ys)))
    sx = [p[0] for p in sm]
    sy = [p[1] for p in sm]
    fig, ax = plt.subplots(figsize=(4, 6))
    ax.plot(xs, ys, "o", color="tab:blue", ms=4, alpha=0.45, label="measured")
    ax.plot(sx, sy, "-", color="tab:blue", lw=2, label="smoothed path")
    ax.plot(0, 0, "s", color="black", ms=8, label="entry")
    ax.plot(xs[-1], ys[-1], "v", color="tab:red", ms=9, label="floor contact")

    ax.invert_yaxis()                 # depth increases downward
    ax.set_aspect("equal", "box")
    ax.axhline(0, color="tab:cyan", lw=1)
    ax.set_xlabel("horizontal displacement (cm)")
    ax.set_ylabel("depth below surface (cm)")
    ax.set_title(title or "Underwater trajectory")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved trajectory plot → {out_path}")


def save_overlay_plot(traces_by_angle, out_path):
    """traces_by_angle: dict angle -> list of trace dicts. One overlaid figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 7))
    seen = set()
    for angle in sorted(traces_by_angle):
        color = ANGLE_COLOR.get(angle, "gray")
        for trace in traces_by_angle[angle]:
            xs, ys = path_to_cm(trace)
            sm = smooth_path(list(zip(xs, ys)))
            label = f"{angle}°" if angle not in seen else None
            ax.plot([p[0] for p in sm], [p[1] for p in sm], "-",
                    color=color, lw=1.2, alpha=0.6, label=label)
            seen.add(angle)

    ax.plot(0, 0, "s", color="black", ms=8, label="entry")
    ax.invert_yaxis()
    ax.set_aspect("equal", "box")
    ax.axhline(0, color="tab:cyan", lw=1)
    ax.set_xlabel("horizontal displacement (cm)")
    ax.set_ylabel("depth below surface (cm)")
    ax.set_title("Underwater trajectories by ramp angle")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved overlaid trajectory plot → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Trace the cylinder's underwater path.")
    parser.add_argument("--video", metavar="FILE", help="Process only this one video")
    parser.add_argument("--ramp",  type=int, default=None, help="Ramp angle (for the title)")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip the annotated trail video (single-video mode)")
    parser.add_argument("--debug", action="store_true",
                        help="Show a live window of the trajectory; save nothing")
    parser.add_argument("--calib", metavar="PATH", default=None,
                        help="Explicit calibration.json path (overrides auto-resolution)")
    args = parser.parse_args()

    calib = load_calibration(args.calib)
    print(f"Scale: {calib['px_per_cm']:.2f} px/cm")

    # ── Single-video mode: both figures ─────────────────────────────────────────
    if args.video:
        if not os.path.exists(args.video):
            sys.exit(f"File not found: {args.video}")
        print(f"\nTracing {args.video} ...")
        base = os.path.splitext(os.path.basename(args.video))[0]

        # Measure the pipe-axis entry angle first so it can be overlaid
        angle_res = measure_entry_angle(args.video, calib, ramp_deg=args.ramp)
        entry_angle  = angle_res["angle_deg"]    if angle_res else None
        entry_frame  = angle_res["frame"]        if angle_res else None
        entry_anchor = angle_res.get("x_px")     if angle_res else None
        if entry_angle is not None:
            print(f"  Entry angle: {entry_angle:.1f} deg (frame {entry_frame})")

        # ── Debug: live window only, save nothing ──────────────────────────────
        if args.debug:
            trace = trace_video(args.video, calib, debug=True,
                                entry_angle=entry_angle, entry_frame=entry_frame,
                                entry_anchor_x=entry_anchor)
            if trace is None:
                sys.exit("Could not trace this video.")
            p = trace["path_px"]
            disp_cm = abs(p[-1][0] - trace["x_entry_px"]) / trace["px_per_cm"] if len(p) >= 2 else 0.0
            print(f"  Path points: {len(p)}   displacement: {disp_cm:.2f} cm")
            return

        video_out = None if args.no_video else f"trace_{base}.mp4"
        trace = trace_video(args.video, calib, video_out=video_out,
                            entry_angle=entry_angle, entry_frame=entry_frame,
                            entry_anchor_x=entry_anchor)
        if trace is None:
            sys.exit("Could not trace this video.")
        p = trace["path_px"]
        disp_cm = abs(p[-1][0] - trace["x_entry_px"]) / trace["px_per_cm"] if len(p) >= 2 else 0.0
        depth_cm = (trace["y_bottom_px"] - trace["y_surface_px"]) / trace["px_per_cm"]
        print(f"  Path points: {len(trace['path_px'])}   "
              f"displacement: {disp_cm:.2f} cm   tank depth: {depth_cm:.1f} cm")
        save_trace_image(trace, f"trace_{base}.png", calib=calib)
        title = f"{base}" + (f"  (ramp {args.ramp}°)" if args.ramp else "")
        save_single_plot(trace, f"traj_{base}.png", title=title)
        if video_out and os.path.exists(video_out):
            print(f"  Saved trail video → {video_out}")
        return

    # ── Batch mode: overlaid plot coloured by ramp angle ────────────────────────
    if not os.path.exists(LABELS_FILE):
        sys.exit(f"{LABELS_FILE} not found. Run with --video FILE, or label videos first.")
    with open(LABELS_FILE) as f:
        labels = json.load(f)

    videos = sorted(labels.keys())
    print(f"Tracing {len(videos)} labelled videos ...\n")

    traces_by_angle = {}
    for i, name in enumerate(videos):
        if not os.path.exists(name):
            print(f"[{i+1:3d}/{len(videos)}] {name}  SKIP (file not found)")
            continue
        angle = labels[name]
        print(f"[{i+1:3d}/{len(videos)}] {name}  ramp={angle}°", end="  ")
        sys.stdout.flush()
        trace = trace_video(name, calib)
        if trace is None or len(trace["path_px"]) < 2:
            print("→ no path")
            continue
        # Need enough of the descent to be a meaningful trajectory
        depth_frac = ((trace["path_px"][-1][1] - trace["y_surface_px"])
                      / (trace["y_bottom_px"] - trace["y_surface_px"]))
        if len(trace["path_px"]) < 4 or depth_frac < 0.5:
            print(f"→ skipped (sparse: {len(trace['path_px'])} pts, "
                  f"{depth_frac:.0%} of depth)")
            continue
        rebound = bounce_rebound_cm(trace)
        if rebound > BOUNCE_LIMIT_CM:
            print(f"→ skipped (floor bounce: {rebound:.1f} cm rebound)")
            continue
        print(f"→ {len(trace['path_px'])} points")
        traces_by_angle.setdefault(angle, []).append(trace)

    if not traces_by_angle:
        sys.exit("No trajectories recorded.")
    save_overlay_plot(traces_by_angle, "trajectories_all.png")


if __name__ == "__main__":
    main()
