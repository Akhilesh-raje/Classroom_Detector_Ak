# -*- coding: utf-8 -*-
"""
SmartClass AI - Standalone CLI Test
Tests the full YOLO pipeline using the core.engine.
Run: python scratch/test_pipeline.py
"""
import sys
import os
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

import cv2
from core.engine import ClassroomEngine

VIDEO_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'test video')
OUTPUT_DIR = os.path.dirname(__file__)

def find_video():
    for f in os.listdir(VIDEO_DIR):
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            return os.path.join(VIDEO_DIR, f)
    return None

def main():
    video_path = find_video()
    if not video_path:
        print(f"ERROR: No video files found in {VIDEO_DIR}")
        return

    output_path = os.path.join(OUTPUT_DIR, 'annotated_output.mp4')

    print("=" * 60)
    print("SmartClass AI - Backend Pipeline Test")
    print("=" * 60)
    print(f"Video : {video_path}")
    print(f"Output: {output_path}")
    print("=" * 60)

    # Open video for annotated output
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    engine = ClassroomEngine(max_seats=6)

    # Run the pipeline (frame_skip=3 for speed)
    events = list(engine.process_video_file(video_path, frame_skip=3))

    # Now create annotated video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_data = {e["frameIndex"]: e for e in events if e.get("type") == "frame"}

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx in frame_data:
            for face in frame_data[frame_idx].get("faces", []):
                x1, y1, x2, y2 = face["bbox"]
                sid = face.get("id")
                activity = face.get("activity", "?")
                attn = face.get("attentionScore", 0)

                # Color based on attention
                if attn >= 75:
                    color = (0, 255, 0)     # green
                elif attn >= 55:
                    color = (0, 255, 255)   # yellow
                else:
                    color = (0, 0, 255)     # red

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                label = f"Seat {sid} | {activity} | {attn:.0f}%"
                cv2.putText(frame, label, (x1, max(25, y1 - 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        out.write(frame)

    cap.release()
    out.release()

    # Print summary
    complete = [e for e in events if e.get("type") == "frame"]
    unique_ids = set()
    for e in complete:
        for face in e.get("faces", []):
            if face.get("id") is not None:
                unique_ids.add(face["id"])

    final_event = [e for e in events if e.get("type") == "complete"]
    if final_event:
        c = final_event[0]
        print()
        print("=" * 60)
        print("FINAL TRACKING RESULTS")
        print("=" * 60)
        print(f"  Frames processed   : {c['framesProcessed']}")
        print(f"  Total detections   : {c['totalFaceDetections']}")
        print(f"  Unique Students    : {len(unique_ids)}  (IDs: {sorted(list(unique_ids))})")
        print(f"  Elapsed time       : {c['elapsedSeconds']}s")
        print(f"  Annotated video    : {output_path}")
        print("=" * 60)

        if len(unique_ids) > 6:
            print(f"\n!! NOTE: Found {len(unique_ids)} unique IDs, but class has only 6.")
            print("   Some IDs may be false positives or tracking switches.")
        elif len(unique_ids) == 6:
            print("\n>> SUCCESS: Perfectly tracked all 6 students!")
    else:
        print("\n!! ERROR: Pipeline did not return a 'complete' event.")

if __name__ == "__main__":
    main()
