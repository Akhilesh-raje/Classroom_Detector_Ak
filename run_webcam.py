# -*- coding: utf-8 -*-
"""
SmartClass AI — Live Webcam Feed Runner (Clean Orchestrator)
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

backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "backend"))
sys.path.insert(0, backend_path)

from core.config import YOLO_CONF, PERSIST, SMALL_BBOX_FRAC
from core.engine import ClassroomEngine
from core.classroom_mapper import ClassroomMapper
from core.behavior_tracker import BehaviorTracker
from core.report_generator import ReportGenerator
from core.visualizer import draw_student_overlay, draw_global_hud, draw_grid_lines, C

# Configs
MAX_SEATS = 6
DISPLAY_W = 1280
ROWS_HINT = 2
COLS_HINT = 3
REPORT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "reports"))
os.makedirs(REPORT_DIR, exist_ok=True)

def open_camera_optimized(idx):
    """
    Open camera indexing DSHOW, attempting highest-res fallback
    to prevent default low-resolution (640x480) crops.
    """
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    
    # 1st attempt: 4K UHD
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 2nd attempt: Full HD fallback (1080p)
    if w < 1920 or h < 1080:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    # 3rd attempt: HD Ready fallback (720p)
    if w < 1280 or h < 720:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    print(f"[CAM] Active resolution opened: {w}x{h}")
    return cap, w, h

def scan_cameras():
    available = []
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
        cap.release()
    return available

def save_session_report(states, behavior_trackers, session_start, session_id):
    rg = ReportGenerator()
    duration = time.time() - session_start
    today = date.today().isoformat()
    reports = []

    for sid, st in sorted(states.items()):
        tracker = behavior_trackers.get(sid) or BehaviorTracker(session_start)
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
        reports.append(report)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(REPORT_DIR, f"session_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)

    print(f"\n[REPORT] Saved session report → {out_path}")
    return out_path

def run():
    print("=" * 65)
    print("SmartClass AI — Live Webcam feed orchestrator")
    print("=" * 65)
    
    available = scan_cameras()
    if not available:
        print("ERROR: No webcam detected. Please connect your camera and retry.")
        return

    engine = ClassroomEngine(max_seats=MAX_SEATS)
    session_start = time.time()
    session_id = engine.session_id
    behavior_trackers = {}
    
    # Auto-prioritize external cameras
    idx_ptr = len(available) - 1
    WIN = "SmartClass AI — Live Classroom (Q=quit | G=remap)"
    
    while True:
        cam_idx = available[idx_ptr]
        cap, w, h = open_camera_optimized(cam_idx)
        if not cap.isOpened():
            idx_ptr = (idx_ptr + 1) % len(available)
            continue
            
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, DISPLAY_W, int(DISPLAY_W * h / w))
        
        # 1st GOOD frame layout scan
        spatial_grid = None
        for _ in range(10):
            ret, ref_frame = cap.read()
            if ret:
                # No mirroring on reference layout! Matches raw coordinates perfectly.
                spatial_grid = ClassroomMapper().detect_layout(ref_frame, rows_hint=ROWS_HINT, cols_hint=COLS_HINT)
                engine.spatial_grid = spatial_grid
                break
                
        grid_label = f"{spatial_grid.rows}x{spatial_grid.cols} grid" if spatial_grid else "no grid"
        fail_count = 0
        switch_camera = False
        t0 = time.time()
        fps_buf = []
        prev_acts = {}
        
        while True:
            t_frame = time.time()
            ret, frame = cap.read()
            if not ret:
                fail_count += 1
                if fail_count >= 3:
                    break
                continue
            fail_count = 0
            
            # CRITICAL FIX: No cv2.flip horizontal mirroring during live processing!
            # Keeps camera coordinates standard and readable, preventing R/L reversals.
            
            draw_grid_lines(frame, spatial_grid)
            
            faces = engine.process_single_frame(frame, w, h, persist=True)
            
            now_ts = time.time()
            for face in faces:
                sid = face["id"]
                act = face["activity"]
                if sid not in behavior_trackers:
                    behavior_trackers[sid] = BehaviorTracker(session_start)
                if prev_acts.get(sid) != act:
                    behavior_trackers[sid].record(act, now_ts)
                    prev_acts[sid] = act
                    
            for face in faces:
                draw_student_overlay(frame, face)
                
            if not faces:
                cv2.putText(frame, "Scanning classroom...", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                            
            fps_buf.append(time.time() - t_frame)
            if len(fps_buf) > 20:
                fps_buf.pop(0)
            live_fps = 1.0 / max(0.001, sum(fps_buf) / len(fps_buf))
            elapsed = time.time() - t0
            
            draw_global_hud(frame, faces, 0, elapsed, live_fps, grid_label, cam_idx=cam_idx)
            
            cv2.imshow(WIN, frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key in (ord("q"), 27):
                save_session_report(engine.student_states, behavior_trackers, session_start, session_id)
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord("n"):
                idx_ptr = (idx_ptr + 1) % len(available)
                switch_camera = True
                break
            elif key == ord("g"):
                print("[GRID] Re-mapping classroom layout...")
                spatial_grid = ClassroomMapper().detect_layout(frame, rows_hint=ROWS_HINT, cols_hint=COLS_HINT)
                engine.spatial_grid = spatial_grid
                grid_label = f"{spatial_grid.rows}x{spatial_grid.cols} grid" if spatial_grid else "no grid"
                
        cap.release()
        cv2.destroyAllWindows()
        if not switch_camera:
            break

if __name__ == "__main__":
    run()
