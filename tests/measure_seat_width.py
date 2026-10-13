# Measure actual per-seat pixel width from detected person positions
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend"))
import core.config as cfg; cfg.YOLO_CONF = 0.22
import cv2, statistics
from core.detector import track_persons

ROOT  = os.path.dirname(os.path.dirname(__file__))
VIDEO = os.path.join(ROOT, "test video", "test video - real classrom.mp4")

cap = cv2.VideoCapture(VIDEO)
W = int(cap.get(3)); H = int(cap.get(4))

# Use a single stable frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 5)
ret, frame = cap.read(); cap.release()

dets = sorted(track_persons(frame, persist=False), key=lambda d: (d["bbox"][1]+d["bbox"][3])/2)
print(f"W={W}  {len(dets)} persons")

# Group into rows by bbox bottom
bottoms = [d["bbox"][3] for d in dets]
bots_sorted = sorted(bottoms)
# The split at the largest gap
gaps = [bots_sorted[i+1]-bots_sorted[i] for i in range(len(bots_sorted)-1)]
split_idx = gaps.index(max(gaps))
row1_bottoms = bots_sorted[:split_idx+1]
row2_bottoms = bots_sorted[split_idx+1:]
print(f"Row 1 ({len(row1_bottoms)} persons): bottoms {row1_bottoms}")
print(f"Row 2 ({len(row2_bottoms)} persons): bottoms {row2_bottoms}")

# For each row measure X-center spacing between adjacent persons
row1_dets = sorted([d for d in dets if d["bbox"][3] in row1_bottoms],
                   key=lambda d: (d["bbox"][0]+d["bbox"][2])/2)
row2_dets = sorted([d for d in dets if d["bbox"][3] in row2_bottoms],
                   key=lambda d: (d["bbox"][0]+d["bbox"][2])/2)

print("\nRow 1 X-centers:")
cx1 = [(d["bbox"][0]+d["bbox"][2])/2 for d in row1_dets]
print(f"  {[round(x) for x in cx1]}")
if len(cx1) >= 2:
    spacings = [cx1[i+1]-cx1[i] for i in range(len(cx1)-1)]
    print(f"  Spacings: {[round(s) for s in spacings]}")
    print(f"  Median spacing: {statistics.median(spacings):.0f}px")
    print(f"  Total span: {cx1[-1]-cx1[0]:.0f}px / {len(cx1)} persons = {(cx1[-1]-cx1[0])/len(cx1):.0f}px/seat")

print("\nRow 2 X-centers:")
cx2 = [(d["bbox"][0]+d["bbox"][2])/2 for d in row2_dets]
print(f"  {[round(x) for x in cx2]}")
if len(cx2) >= 2:
    spacings = [cx2[i+1]-cx2[i] for i in range(len(cx2)-1)]
    print(f"  Spacings: {[round(s) for s in spacings]}")
    print(f"  Median spacing: {statistics.median(spacings):.0f}px")
    print(f"  Total span: {cx2[-1]-cx2[0]:.0f}px / {len(cx2)} persons = {(cx2[-1]-cx2[0])/len(cx2):.0f}px/seat")

print(f"\nHough line R1 span: 1450px  / actual {len(row1_dets)} persons = {1450/len(row1_dets):.0f}px/person")
print(f"Hough line R2 span: 1919px  / actual {len(row2_dets)} persons = {1919/len(row2_dets):.0f}px/person")
