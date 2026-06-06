# -*- coding: utf-8 -*-
"""
SmartClass AI — Video File Runner (Clean Orchestrator)
"""
import sys
import os
import io
import time
import cv2
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "backend"))
sys.path.insert(0, backend_path)

from core.config import YOLO_CONF, PERSIST, SMALL_BBOX_FRAC
from core.engine import ClassroomEngine
from core.student_state import StudentState
from core.visualizer import draw_student_overlay, draw_global_hud, C

FRAME_SKIP = 3
MAX_SEATS = 6
VIDEO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "test video"))
OUTPUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "annotated_output.mp4"))
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
DISPLAY_W = 1280

def find_video(directory):
    if not os.path.isdir(directory):
        return None
    for f in sorted(os.listdir(directory)):
        if f.lower().endswith(VIDEO_EXTS):
            return os.path.join(directory, f)
    return None

def run():
    print("=" * 60)
    print("SmartClass AI — Video File Runner (Clean Orchestrator)")
    print("=" * 60)

    # Allow passing a specific video path as a command-line argument
    if len(sys.argv) > 1:
        video_path = os.path.abspath(sys.argv[1])
        if not os.path.isfile(video_path):
            print(f"ERROR: File not found: '{video_path}'")
            return
    else:
        video_path = find_video(VIDEO_DIR)
        if not video_path:
            print(f"ERROR: No video found in '{VIDEO_DIR}'")
            return

    print(f"Video  : {os.path.basename(video_path)}")
    print(f"Output : {OUTPUT_PATH}")
    print("=" * 60)

    cap = cv2.VideoCapture(video_path)
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_src, (W, H))

    WIN = "SmartClass AI — Video Processing Preview"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, DISPLAY_W, int(DISPLAY_W * H / W))

    engine = ClassroomEngine(max_seats=MAX_SEATS)
    frame_idx = 0
    processed = 0
    detections = 0
    last_faces = []
    t0 = time.time()
    fps_buf = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        t_frame = time.time()

        if frame_idx % FRAME_SKIP == 0:
            processed += 1
            faces = engine.process_single_frame(frame, W, H, persist=True)
            last_faces = faces
            detections += len(faces)

        for face in last_faces:
            draw_student_overlay(frame, face)

        elapsed = time.time() - t0
        fps_buf.append(time.time() - t_frame)
        if len(fps_buf) > 20:
            fps_buf.pop(0)
        live_fps = 1.0 / max(0.001, sum(fps_buf) / len(fps_buf))

        draw_global_hud(frame, last_faces, {
            "total_frames": total_fr,
            "elapsed": elapsed,
            "fps": live_fps,
            "mode": "video run",
            "frame_idx": frame_idx,
        })

        out.write(frame)
        cv2.imshow(WIN, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quit by user.")
            break

    cv2.destroyAllWindows()
    cap.release()
    out.release()

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Frames processed   : {processed}")
    print(f"  Unique seat IDs    : {sorted(engine.student_states.keys())}")
    print(f"  Total time         : {time.time() - t0:.2f}s")
    print(f"  Annotated video    : {OUTPUT_PATH}")
    if engine.student_states:
        print()
        print(f"  {'Seat':<6} {'Avg Attn':>9} {'Peak':>7} {'Activity':<28}")
        print("  " + "-" * 50)
        for sid, st in sorted(engine.student_states.items()):
            rec_act, note = st.reconciled_activity()
            display = f"{rec_act}" + (f" ({note})" if note else "")
            print(f"  {sid:<6} {st.avg:>8.1f}%  {st.peak:>6.1f}%  {display:<28}")
    print("=" * 60)

if __name__ == "__main__":
    run()
