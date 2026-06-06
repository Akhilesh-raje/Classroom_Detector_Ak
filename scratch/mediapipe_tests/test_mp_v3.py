import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

model_path = r'C:\Users\rajea\Documents\classroom\backend\face_landmarker.task'
video_path = r'C:\Users\rajea\Documents\classroom\test video\Student Sitting in the Class  Royalty Free Video  No Copyright Video_1080p.mp4'

# Initialize detector with extremely low confidence
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    num_faces=10,
    min_face_detection_confidence=0.01,
    min_face_presence_confidence=0.01,
    min_tracking_confidence=0.01
)
detector = vision.FaceLandmarker.create_from_options(options)

# Read frames and try to find ANYTHING
cap = cv2.VideoCapture(video_path)
found = False
for i in range(0, 500, 50):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if not ret: break
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    result = detector.detect(mp_image)
    n = len(result.face_landmarks) if result.face_landmarks else 0
    print(f"Frame {i}: Found {n} faces")
    if n > 0: found = True

cap.release()
detector.close()

if not found:
    print("CRITICAL: Even with 0.01 confidence, no faces were found.")
