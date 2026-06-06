import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

model_path = r'C:\Users\rajea\Documents\classroom\backend\face_landmarker.task'
video_path = r'C:\Users\rajea\Documents\classroom\test video\Student Sitting in the Class  Royalty Free Video  No Copyright Video_1080p.mp4'

# Initialize detector
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    num_faces=10,
    min_face_detection_confidence=0.1
)
detector = vision.FaceLandmarker.create_from_options(options)

# Read frame
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to read video")
    exit()

# Try different scales
for scale in [0.25, 0.5, 1.0]:
    w = int(frame.shape[1] * scale)
    h = int(frame.shape[0] * scale)
    small = cv2.resize(frame, (w, h))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    result = detector.detect(mp_image)
    n = len(result.face_landmarks) if result.face_landmarks else 0
    print(f"Scale {scale} ({w}x{h}): Found {n} faces")
    
    if n > 0:
        for i, lms in enumerate(result.face_landmarks):
            # Check bounding box
            xs = [lm.x for lm in lms]
            ys = [lm.y for lm in lms]
            print(f"  Face {i}: bbox norm [{min(xs):.3f}, {min(ys):.3f}, {max(xs):.3f}, {max(ys):.3f}]")

detector.close()
