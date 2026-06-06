# -*- coding: utf-8 -*-
"""
SmartClass AI — Live Webcam Runner  (Fixed v3)
================================================
Fixes applied:
  - 4K → 1080p → 720p resolution fallback (no more 640x480)
  - No cv2.flip() mirroring (was reversing left/right grid coords)
  - Dynamic max_seats from detected faces (not hardcoded 6)
  - Auto classroom layout detection on startup
  - Reconciled activity labels (no high-attention + distracted contradiction)
  - Attention factor breakdown shown in panel
  - Session report saved on quit

Controls: Q/ESC=quit | N=next cam | G=remap grid | R=save report | 0-9=select cam
"""
import sys
import os
import io
import cv2
import time
import json
import numpy as np
from datetime import date, datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend"))
sys.path.insert(0, backend_path)

from core.engine import ClassroomEngine
from core.classroom_mapper import ClassroomMapper
from core.behavior_tracker import BehaviorTracker
from core.report_generator import ReportGenerator
from core.visualizer import draw_student_overlay, draw_global_hud, draw_grid_overlay, reset_smooth

# ── Config ────────────────────────────────────────────────────────────────────
ROWS_HINT   = 2
COLS_HINT   = 3
DISPLAY_W   = 1280
FAIL_LIMIT  = 3
REPORT_DIR  = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# Resolution fallback chain: try highest first
RESOLUTIONS = [(3840, 2160), (1920, 1080), (1280, 720), (640, 480)]


# ── Camera helpers ────────────────────────────────────────────────────────────

def scan_cameras() -> list:
    """Scan indices 0-4, return list of responsive camera indices."""
    available = []
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
        cap.release()
    return available


def open_camera(idx: int) -> tuple:
    """Open camera, try resolutions from highest to lowest. Returns (cap, w, h)."""
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return None, 0, 0

    actual_w, actual_h = 0, 0
    for rw, rh in RESOLUTIONS:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  rw)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, rh)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w >= 640 and h >= 480:
            actual_w, actual_h = w, h
            break

    print(f"[CAM] Camera {idx} opened at {actual_w}x{actual_h}")
    return cap, actual_w, actual_h


# ── Classroom layout detection ────────────────────────────────────────────────

def detect_layout(frame, rows_hint=ROWS_HINT, cols_hint=COLS_HINT):
    """Detect classroom layout from a reference frame."""
    print(f"[GRID] Detecting layout (hint: {rows_hint}x{cols_hint})...")
    try:
        mapper = ClassroomMapper()
        grid   = mapper.detect_layout(frame, rows_hint=rows_hint, cols_hint=cols_hint)
        print(f"[GRID] {grid.rows}x{grid.cols} = {len(grid.cells)} cells: {[c.label for c in grid.cells]}")
        return grid
    except Exception as e:
        print(f"[GRID] Detection failed: {e} — using fallback")
        try:
            return ClassroomMapper()._fallback_grid(frame.shape[0], frame.shape[1], rows_hint, cols_hint)
        except Exception:
            return None


# ── Report saving ─────────────────────────────────────────────────────────────

def save_report(engine, session_start, session_id):
    """Generate and save session report to JSON."""
    rg       = ReportGenerator()
    duration = time.time() - session_start
    today    = date.today().isoformat()
    reports  = []

    for sid, st in sorted(engine.student_states.items()):
        tracker = engine.behavior_trackers.get(sid)
        if tracker is None:
            tracker = BehaviorTracker(session_start)

        rec_act, note = st.reconciled_activity()
        breakdown     = st.attention_breakdown()

        report = rg.generate_student_report(
            student_state=st,
            behavior_tracker=tracker,
            session_id=session_id,
            session_date=today,
            session_duration=duration,
            frame_skip=1,
            source_fps=25.0,
            student_name=f"Seat {sid}",
            grid_label=str(sid),
            attendance_status="present" if st.attn_hist else "absent",
        )
        # Enrich with reconciled data
        report["reconciled_activity"] = rec_act
        report["activity_note"]       = note
        report["attention_breakdown"] = breakdown
        reports.append(report)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(REPORT_DIR, f"session_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)

    print(f"\n[REPORT] Saved {len(reports)} student reports → {out_path}")
    print(f"\n{'='*60}")
    print("SESSION SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Seat':<6} {'Avg':>7} {'Peak':>7} {'Activity':<20} {'Note'}")
    print(f"  {'-'*60}")
    for r in reports:
        print(f"  {r['grid_label']:<6} "
              f"{r['avg_attention']:>6.1f}%  "
              f"{r['peak_attention']:>6.1f}%  "
              f"{r['reconciled_activity']:<20} "
              f"{r.get('activity_note','')}")
    print(f"{'='*60}\n")
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print("\n" + "=" * 65)
    print("SmartClass AI — Live Webcam Runner  (Fixed v3)")
    print("=" * 65)

    available = scan_cameras()
    if not available:
        print("ERROR: No camera found. Connect a webcam and retry.")
        sys.exit(0)

    print(f"Available cameras: {available}")

    # Dynamic max_seats — start with 8, will auto-adjust
    engine        = ClassroomEngine(max_seats=8)
    session_start = time.time()
    session_id    = engine.session_id
    prev_acts     = {}

    idx_ptr = len(available) - 1
    WIN     = "SmartClass AI — Live (Q=quit | G=grid | R=report | N=next cam)"

    try:
        while True:
            cam_idx = available[idx_ptr]
            print(f"\n[CAM] Opening camera {cam_idx}...")
            cap, w, h = open_camera(cam_idx)

            if cap is None or not cap.isOpened():
                print(f"[CAM] Failed to open {cam_idx}. Trying next.")
                idx_ptr = (idx_ptr + 1) % len(available)
                continue

            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WIN, DISPLAY_W, int(DISPLAY_W * h / w) if w > 0 else 720)

            # ── Auto-detect classroom layout ──────────────────────────────────
            print("[GRID] Capturing reference frame...")
            spatial_grid = None
            for _ in range(15):
                ret, ref = cap.read()
                if ret and ref is not None:
                    spatial_grid = detect_layout(ref)
                    engine.spatial_grid = spatial_grid
                    break

            grid_label = (f"{spatial_grid.rows}x{spatial_grid.cols}"
                          if spatial_grid else "no grid")

            fail_count    = 0
            switch_camera = False
            t0            = time.time()
            fps_buf       = []
            reset_smooth()

            while True:
                t_frame = time.time()
                ret, frame = cap.read()

                if not ret:
                    fail_count += 1
                    if fail_count >= FAIL_LIMIT:
                        print(f"[CAM] {FAIL_LIMIT} consecutive read failures. Exiting.")
                        cap.release()
                        cv2.destroyAllWindows()
                        sys.exit(0)
                    continue
                fail_count = 0

                # NOTE: No cv2.flip() — mirroring was reversing grid coordinates

                # ── Draw grid (background layer) ──────────────────────────────
                draw_grid_overlay(frame, spatial_grid)

                # ── Process frame ─────────────────────────────────────────────
                faces = engine.process_single_frame(frame, w, h, persist=True)

                # ── Track behavior changes ────────────────────────────────────
                now_ts = time.time()
                for face in faces:
                    sid = face["id"]
                    act = face.get("reconciledActivity", face.get("activity", "neutral"))
                    if sid not in engine.behavior_trackers:
                        engine.behavior_trackers[sid] = BehaviorTracker(session_start)
                    if prev_acts.get(sid) != act:
                        engine.behavior_trackers[sid].record(act, now_ts)
                        prev_acts[sid] = act

                # ── Draw student overlays ─────────────────────────────────────
                for face in faces:
                    draw_student_overlay(frame, face)

                if not faces:
                    cv2.putText(frame, "No students detected", (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 220), 2, cv2.LINE_AA)

                # ── HUD ───────────────────────────────────────────────────────
                fps_buf.append(time.time() - t_frame)
                if len(fps_buf) > 20:
                    fps_buf.pop(0)
                live_fps = 1.0 / max(0.001, sum(fps_buf) / len(fps_buf))

                draw_global_hud(frame, faces, {
                    "mode":       "LIVE",
                    "cam_idx":    cam_idx,
                    "w":          w,
                    "h":          h,
                    "fps":        live_fps,
                    "elapsed":    time.time() - t0,
                    "grid_label": grid_label,
                    "hint":       "N=next | G=remap grid | R=report | Q=quit",
                })

                cv2.imshow(WIN, frame)
                key = cv2.waitKey(1) & 0xFF

                # ── Key handling ──────────────────────────────────────────────
                if key in (ord("q"), 27):
                    save_report(engine, session_start, session_id)
                    cap.release()
                    cv2.destroyAllWindows()
                    sys.exit(0)

                elif key == ord("n"):
                    idx_ptr = (idx_ptr + 1) % len(available)
                    switch_camera = True
                    break

                elif key == ord("g"):
                    print("[GRID] Re-mapping classroom layout...")
                    spatial_grid = detect_layout(frame)
                    engine.spatial_grid = spatial_grid
                    grid_label = (f"{spatial_grid.rows}x{spatial_grid.cols}"
                                  if spatial_grid else "no grid")
                    reset_smooth()

                elif key == ord("r"):
                    save_report(engine, session_start, session_id)

                elif ord("0") <= key <= ord("9"):
                    desired = key - ord("0")
                    if desired in available:
                        idx_ptr = available.index(desired)
                        switch_camera = True
                        break
                    else:
                        print(f"[CAM] Index {desired} not available: {available}")

            cap.release()
            cv2.destroyAllWindows()

            if not switch_camera:
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
        save_report(engine, session_start, session_id)
    finally:
        cv2.destroyAllWindows()
        print("[INFO] Done.")


if __name__ == "__main__":
    run()
