# -*- coding: utf-8 -*-
"""Quick diagnostic test - just 10 frames"""
import sys, os, time, cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.dirname(HERE)
BACKEND_PATH = os.path.join(ROOT_PATH, "backend")
sys.path.insert(0, BACKEND_PATH)

VIDEO = os.path.join(ROOT_PATH, "test video", "test video - real classrom.mp4")

import core.config as cfg
cfg.YOLO_CONF = 0.30

from core.engine import ClassroomEngine

print("Quick Diagnostic Test")
print("=" * 60)

cap = cv2.VideoCapture(VIDEO)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Video: {W}x{H} @ {fps}fps")
print("Processing 10 frames...\n")

engine = ClassroomEngine(max_seats=15)

for i in range(10):
    ret, frame = cap.read()
    if not ret:
        break
    
    t0 = time.perf_counter()
    faces = engine.process_single_frame(frame, W, H, persist=True)
    dt = (time.perf_counter() - t0) * 1000
    
    print(f"Frame {i+1}: {len(faces)} faces detected in {dt:.0f}ms")
    
    for j, face in enumerate(faces[:3]):  # Show first 3
        sid = face.get("id", "?")
        act = face.get("activity", "?")
        attn = face.get("attentionScore", 0)
        print(f"  S{sid}: {act} {attn:.0f}%")

cap.release()

print("\nStudent States:")
for sid, st in sorted(engine.student_states.items()):
    act, note = st.reconciled_activity()
    print(f"  S{sid}: avg={st.avg:.1f}%, peak={st.peak:.1f}%, {act}")

print("\n✓ Test complete")
