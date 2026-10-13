"""
Register test students by extracting faces from the classroom video
"""
import cv2
import sys
import requests
from pathlib import Path

API_URL = "http://localhost:8000/api"
VIDEO_PATH = r"C:\Users\rajea\Documents\classroom\test video\test video - real classrom.mp4"

# Test students to register
STUDENTS = [
    {"name": "Priya Sharma", "roll_no": "CS001", "section": "A"},
    {"name": "Ananya Gupta", "roll_no": "CS002", "section": "A"},
    {"name": "Sanya Verma", "roll_no": "CS003", "section": "A"},
    {"name": "Riya Patel", "roll_no": "CS004", "section": "A"},
    {"name": "Kavya Singh", "roll_no": "CS005", "section": "A"},
]

print("SmartClass — Test Student Registration")
print("=" * 80)
print(f"Video: {VIDEO_PATH}")
print(f"API: {API_URL}")
print()

# 1. Create students in database
print("Step 1: Creating student records...")
created_students = []
for student in STUDENTS:
    try:
        resp = requests.post(
            f"{API_URL}/students",
            data={
                "name": student["name"],
                "roll_no": student["roll_no"],
                "section": student["section"],
                "email": f"{student['roll_no'].lower()}@college.edu"
            }
        )
        if resp.status_code == 200:
            data = resp.json()
            created_students.append(data)
            print(f"  ✓ Created: {student['name']} (ID: {data['id']})")
        else:
            print(f"  ✗ Failed: {student['name']} - {resp.text}")
    except Exception as e:
        print(f"  ✗ Error: {student['name']} - {e}")

if not created_students:
    print("\n❌ No students created. Exiting.")
    sys.exit(1)

print(f"\n✓ Created {len(created_students)} students")

# 2. Extract faces from video
print("\nStep 2: Extracting face samples from video...")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ Could not open video")
    sys.exit(1)

# Skip to frame 100 (students should be visible and settled)
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)

face_samples = []
for i in range(10):  # Extract 10 frames
    ret, frame = cap.read()
    if not ret:
        break
    
    # Use a simple face detector to find faces
    # (In production, use the same YOLO detector as the main system)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        face_samples.append((frame, faces))
        print(f"  Frame {100+i}: Found {len(faces)} faces")
    
    # Skip a few frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + 5)

cap.release()

if not face_samples:
    print("\n❌ No faces detected in video. Try different frames.")
    sys.exit(1)

print(f"\n✓ Extracted {len(face_samples)} frames with faces")

# 3. Register faces for each student
print("\nStep 3: Registering faces...")
# Take first N students and assign them the largest face crops
frame, faces = face_samples[0]  # Use first frame
faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)  # Sort by area

for idx, student in enumerate(created_students[:len(faces_sorted)]):
    try:
        x, y, w, h = faces_sorted[idx]
        # Extract face crop with some padding
        pad = 20
        y1, y2 = max(0, y-pad), min(frame.shape[0], y+h+pad)
        x1, x2 = max(0, x-pad), min(frame.shape[1], x+w+pad)
        face_crop = frame[y1:y2, x1:x2]
        
        # Save temporarily
        temp_path = f"temp_face_{student['id']}.jpg"
        cv2.imwrite(temp_path, face_crop)
        
        # Upload to API
        with open(temp_path, "rb") as f:
            files = {"image": f}
            resp = requests.post(f"{API_URL}/register-face/{student['id']}", files=files)
            if resp.status_code == 200:
                print(f"  ✓ Registered face: {student['name']}")
            else:
                print(f"  ✗ Failed: {student['name']} - {resp.text}")
        
        # Clean up
        Path(temp_path).unlink(missing_ok=True)
    
    except Exception as e:
        print(f"  ✗ Error: {student['name']} - {e}")

print("\n" + "=" * 80)
print("✅ Registration complete!")
print(f"   {len(created_students)} students created")
print(f"   Faces registered for students")
print("\nNow play the video in the web UI — students should be recognized by name!")
print("=" * 80)
