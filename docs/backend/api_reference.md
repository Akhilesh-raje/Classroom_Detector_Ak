# 📡 Backend API Reference

This document provides a complete specification for all REST endpoints, streaming routes, and RPC actions implemented in [backend/main.py](file:///c:/Users/rajea/Documents/classroom/backend/main.py).

---

## 👥 Student Management Endpoints

### 1. `GET /api/students`
Retrieves all registered students with profile details, seating data, and enrollment flags.

- **Response `200 OK`**:
```json
[
  {
    "id": 1,
    "name": "Aarav Sharma",
    "rollNo": "CS2026-001",
    "section": "CSE-A",
    "email": "aarav.sharma@example.com",
    "hasFace": true,
    "faceImage": "data:image/jpeg;base64,...",
    "gridCell": "R1C1",
    "seatRow": 1,
    "seatBench": 1,
    "seatPos": 1,
    "registeredAt": "2026-10-12T08:30:00",
    "totalAttendance": 42,
    "avgAttentiveness": 84.5
  }
]
```

### 2. `POST /api/students`
Registers a new student record (with optional face image upload).

- **Form Parameters**:
  - `name` (string, required): Full name.
  - `roll_no` (string, required): Unique Roll Number.
  - `section` (string, optional): Section or class group.
  - `email` (string, optional): Email address.
  - `grid_cell` (string, optional): Assigned default seat (e.g. `R1C1`).
  - `face_image` (UploadFile, optional): Portrait photo for face embedding extraction.
- **Response `201 Created`**: Returns newly created student object.

### 3. `DELETE /api/students/{student_id}`
Removes a student and all associated biometric embeddings from the database.

- **Path Parameter**: `student_id` (integer).
- **Response `200 OK`**: `{"status": "deleted", "id": 1}`.

### 4. `POST /api/register-face/{student_id}` & `POST /api/enroll-face/{student_id}`
Uploads or updates the face photo for an existing student, triggering face alignment and 512-D embedding extraction.

- **Form Parameter**: `file` (UploadFile, required).
- **Response `200 OK`**: `{"status": "enrolled", "student_id": 1, "embedding_dim": 512}`.

### 5. `GET /api/students/{student_id}/profile`
Retrieves extended longitudinal data for a student, including historical attention curves, total session count, and behavioral flags.

---

## 📋 Attendance Engine Endpoints

### 1. `GET /api/attendance`
Retrieves current session attendance records, including confirmation tiers and temporal verification status.

- **Query Parameters**:
  - `date` (string, optional): ISO date filter (`YYYY-MM-DD`).
  - `session_id` (integer, optional): Specific session filter.
- **Response `200 OK`**:
```json
[
  {
    "id": 101,
    "student_id": 1,
    "student_name": "Aarav Sharma",
    "seat_label": "R1C1",
    "status": "PRESENT",
    "verification_tier": "TIER_1_VERIFIED",
    "confidence": 0.89,
    "first_seen": "2026-10-13T09:00:15",
    "last_seen": "2026-10-13T09:45:20",
    "duration_minutes": 45.1,
    "needs_review": false
  }
]
```

### 2. `POST /api/attendance/process-observation`
Ingests a single-frame observation from the detection pipeline into the continuous attendance state machine.

- **JSON Body**:
```json
{
  "student_id": 1,
  "seat_label": "R1C1",
  "confidence": 0.88,
  "attention_score": 82.0,
  "timestamp": 1728800000.0
}
```

### 3. `POST /api/attendance/review`
Resolves attendance flags or reviews flagged Tier-3 entries.

- **JSON Body**:
```json
{
  "record_id": 101,
  "action": "CONFIRM_PRESENT",
  "notes": "Verified by instructor"
}
```

### 4. `POST /api/attendance/manual`
Allows instructors to manually mark an absent or unverified student as present/absent.

### 5. `POST /api/attendance/reset`
Resets the active attendance buffer for a new lecture.

### 6. `GET /api/attendance/export-csv`
Downloads an official attendance ledger for the current session in CSV format.

---

## 📹 Video, Camera & Live Streaming Endpoints

### 1. `POST /api/camera/start`
Initializes USB camera hardware or RTSP stream.

- **JSON Body**: `{"device_index": 0}` (or RTSP URL).
- **Response `200 OK`**: `{"status": "camera_started"}`.

### 2. `POST /api/camera/stop`
Stops capturing and releases camera hardware lock.

### 3. `GET /api/camera/stream`
Continuous MJPEG video stream with computer vision overlays (bounding boxes, attention scores, HUD).

- **Content-Type**: `multipart/x-mixed-replace; boundary=frame`.

### 4. `GET /api/live-classroom/stream`
Stream endpoint for test video or recorded simulation.

### 5. `GET /api/live-classroom/telemetry`
Real-time JSON telemetry payload for dashboard widgets.

- **Response `200 OK`**:
```json
{
  "fps": 24.5,
  "timestamp": 45.2,
  "active_students": 12,
  "class_attention_avg": 79.4,
  "activity_counts": {
    "studying": 8,
    "attentive": 3,
    "distracted": 1,
    "drowsy": 0
  },
  "students": [
    {
      "seat_id": 1,
      "seat_label": "R1C1",
      "track_id": 3,
      "student_name": "Aarav Sharma",
      "attention_score": 85.0,
      "activity": "studying",
      "status": "OK"
    }
  ]
}
```

---

## 📊 Sessions & Analytics Endpoints

### 1. `POST /api/sessions/start`
Starts a recorded classroom monitoring session.

- **JSON Body**: `{"subject": "Computer Science 101", "instructor": "Prof. Rao", "section": "CSE-A"}`.

### 2. `POST /api/sessions/stop`
Concludes session, computes aggregate longitudinal statistics, and creates a `SessionReport`.

### 3. `GET /api/sessions`
Lists all historical recorded sessions.

### 4. `GET /api/reports/session/{session_id}`
Returns complete analysis report for a given session.

### 5. `GET /api/analytics/summary`
Returns global analytics: average attentiveness over time, daily attendance rates, and most frequent distractions.

---

## 🧠 Structure AI & Video Engine

### 1. `POST /api/analyze-video`
Uploads an MP4 video file and runs the full classroom perception pipeline.

### 2. `POST /api/analyze-video-path`
Runs analysis on a local video file path (e.g. `test video/test video - real classrom.mp4`).

### 3. `GET /api/structure-ai/latest`
Returns the latest synthesized classroom geometry, desk edge coordinates, and row topologies.

### 4. `POST /api/structure-ai/run`
Triggers monocular depth map calculation and Hough line fitting on the current scene frame.

---

## 🤖 Ollama Local LLM Integration

### 1. `POST /api/ollama/status`
Checks if the local Ollama daemon is reachable (`http://localhost:11434`) and lists installed models (e.g. `llava:latest`, `llama3:latest`).

### 2. `POST /api/ollama/label-seats`
Prompts the local multimodal LLM with classroom crops to provide natural-language seat behavior summaries.

### 3. `GET /api/ollama/analyze`
Generates an executive instructional summary for the instructor with pedagogical improvement tips.
