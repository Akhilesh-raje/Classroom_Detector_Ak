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

# Read frame from middle
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to read video")
    exit()

cv2.imwrite(r'C:\Users\rajea\Documents\classroom\scratch\middle_frame.jpg', frame)

# Try scale 1.0
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

result = detector.detect(mp_image)
n = len(result.face_landmarks) if result.face_landmarks else 0
print(f"Frame 100: Found {n} faces")

detector.close()
