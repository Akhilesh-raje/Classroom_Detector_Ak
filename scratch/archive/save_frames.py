import os
import cv2

ROOT = r"c:\Users\rajea\Documents\classroom"
v1_path = os.path.join(ROOT, "test video", "Student Sitting in the Class  Royalty Free Video  No Copyright Video_1080p.mp4")
v2_path = os.path.join(ROOT, "test video", "test video - real classrom.mp4")

out_dir = os.path.join(ROOT, "scratch")

cap1 = cv2.VideoCapture(v1_path)
ret1, f1 = cap1.read()
cap1.release()
if ret1:
    cv2.imwrite(os.path.join(out_dir, "v1_frame0.jpg"), f1)

cap2 = cv2.VideoCapture(v2_path)
ret2, f2 = cap2.read()
cap2.release()
if ret2:
    cv2.imwrite(os.path.join(out_dir, "v2_frame0.jpg"), f2)

print("Saved frames to scratch/v1_frame0.jpg and scratch/v2_frame0.jpg")
