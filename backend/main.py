"""
SmartClass AI — FastAPI Backend
Handles: student registration, face processing, attendance, analytics, video analysis
"""
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.orm import Session
from datetime import datetime, date
from typing import Optional
import cv2
import numpy as np
import base64
import json
import asyncio
import threading
import tempfile
import os
import logging

import re
import uuid
import time as time_module

from core.database import (
    init_db, get_db, Student, AttendanceRecord, ActivityLog,
    Session as SessionModel, SessionReport, SessionLocal,
)
from core.face_engine import face_engine
from core.face_recognition_engine import face_rec_engine
from core.ollama_labeler import ollama_labeler
from core.engine import ClassroomEngine, get_logger
from core.report_generator import ReportGenerator

video_processor = ClassroomEngine()
report_gen = ReportGenerator()
log = get_logger("main")

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize DB and load faces
    init_db()
    db = SessionLocal()
    try:
        students = db.query(Student).filter(Student.face_encoding.isnot(None)).all()
        face_engine.load_registered_faces(students)
        # Also load into main engine for real face recognition
        video_processor.load_registered_faces(students)
        print(f"SmartClass AI started. {len(students)} faces loaded.")
    finally:
        db.close()
    yield
    # Shutdown: Clean up camera if needed
    global camera_capture, is_monitoring
    is_monitoring = False
    if camera_capture:
        camera_capture.release()

app = FastAPI(title="SmartClass AI", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles
# Mount test video directory for playback
test_video_dir = os.path.join(os.path.dirname(__file__), "..", "test video")
if os.path.exists(test_video_dir):
    app.mount("/test-videos", StaticFiles(directory=test_video_dir), name="test-videos")

# Global state for live streaming
camera_lock = threading.Lock()
camera_capture = None
is_monitoring = False
latest_detections = []

@app.get("/api/students")
def get_students(db: Session = Depends(get_db)):
    students = db.query(Student).all()
    return [
        {
            "id": s.id,
            "name": s.name,
            "rollNo": s.roll_no,
            "section": s.section,
            "email": s.email,
            "hasFace": s.face_encoding is not None,
            "faceImage": s.face_image,
            "registeredAt": s.registered_at.isoformat() if s.registered_at else None,
            "totalAttendance": s.total_attendance,
            "avgAttentiveness": s.avg_attentiveness,
        }
        for s in students
    ]


@app.post("/api/students")
def create_student(
    name: str = Form(...),
    rollNo: str = Form(...),
    section: str = Form(...),
    email: str = Form(""),
    faceImage: str = Form(""),
    faceEncoding: str = Form(""),
    db: Session = Depends(get_db),
):
    existing = db.query(Student).filter(Student.roll_no == rollNo).first()
    if existing:
        raise HTTPException(400, "Roll number already registered")

    encoding_bytes = None
    if faceEncoding:
        encoding_bytes = base64.b64decode(faceEncoding)

    student = Student(
        name=name,
        roll_no=rollNo,
        section=section,
        email=email if email else None,
        face_encoding=encoding_bytes,
        face_image=faceImage if faceImage else None,
    )
    db.add(student)
    db.commit()
    db.refresh(student)

    if encoding_bytes:
        face_engine.register_face(student.id, encoding_bytes)

    return {"id": student.id, "message": "Student registered successfully"}


@app.delete("/api/students/{student_id}")
def delete_student(student_id: int, db: Session = Depends(get_db)):
    student = db.query(Student).filter(Student.id == student_id).first()
    if not student:
        raise HTTPException(404, "Student not found")
    db.delete(student)
    db.commit()
    # Remove from engine
    face_engine.registered_encodings = [
        (sid, enc) for sid, enc in face_engine.registered_encodings if sid != student_id
    ]
    return {"message": "Student deleted"}


# ─── Face Processing ─────────────────────────────────────────

@app.post("/api/process-frame")
async def process_frame(image: UploadFile = File(...), db: Session = Depends(get_db)):
    """Process a single frame from the frontend camera."""
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(400, "Invalid image")

    faces = face_engine.process_frame(frame)
    today = date.today().isoformat()

    results = []
    for face in faces:
        student_name = None
        student_roll = None
        attendance_marked = False

        if face["matched_student_id"]:
            student = db.query(Student).filter(Student.id == face["matched_student_id"]).first()
            if student:
                student_name = student.name
                student_roll = student.roll_no

                # Auto-mark attendance
                existing = (
                    db.query(AttendanceRecord)
                    .filter(
                        AttendanceRecord.student_id == student.id,
                        AttendanceRecord.date == today,
                    )
                    .first()
                )
                if not existing:
                    record = AttendanceRecord(
                        student_id=student.id,
                        date=today,
                        confidence=face["match_confidence"],
                        captured_image=face["thumbnail"],
                    )
                    db.add(record)
                    student.total_attendance += 1
                    db.commit()
                    attendance_marked = True

        results.append({
            "bbox": face["bbox"],
            "headPose": face["head_pose"],
            "eyeOpenness": face["eye_openness"],
            "mouthOpen": face["mouth_open"],
            "attentionScore": face["attention_score"],
            "activity": face["activity"],
            "emotion": face["emotion"],
            "matchedStudentId": face["matched_student_id"],
            "studentName": student_name,
            "studentRoll": student_roll,
            "matchConfidence": face["match_confidence"],
            "attendanceMarked": attendance_marked,
            "thumbnail": face["thumbnail"],
        })

    return {"faces": results, "count": len(results), "timestamp": datetime.now().isoformat()}


@app.post("/api/register-face/{student_id}")
async def register_face(student_id: int, image: UploadFile = File(...), db: Session = Depends(get_db)):
    """Capture a face and register encoding for a student."""
    student = db.query(Student).filter(Student.id == student_id).first()
    if not student:
        raise HTTPException(404, "Student not found")

    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(400, "Invalid image")

    faces = face_engine.process_frame(frame)
    if not faces:
        raise HTTPException(400, "No face detected in image")

    # Use the largest face
    largest = max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))

    encoding_bytes = largest["encoding"].tobytes()
    student.face_encoding = encoding_bytes
    student.face_image = largest["thumbnail"]
    db.commit()

    face_engine.register_face(student_id, encoding_bytes)

    return {"message": "Face registered successfully", "thumbnail": largest["thumbnail"]}


# ─── Attendance ───────────────────────────────────────────────

@app.get("/api/attendance")
def get_attendance(target_date: Optional[str] = None, db: Session = Depends(get_db)):
    if not target_date:
        target_date = date.today().isoformat()

    records = db.query(AttendanceRecord).filter(AttendanceRecord.date == target_date).all()
    all_students = db.query(Student).all()
    student_map = {s.id: s for s in all_students}

    present_ids = {r.student_id for r in records}

    result = []
    for s in all_students:
        record = next((r for r in records if r.student_id == s.id), None)
        result.append({
            "studentId": s.id,
            "name": s.name,
            "rollNo": s.roll_no,
            "section": s.section,
            "faceImage": s.face_image,
            "status": "present" if s.id in present_ids else "absent",
            "markedAt": record.marked_at.isoformat() if record else None,
            "confidence": record.confidence if record else None,
            "method": record.method if record else None,
        })

    return {
        "date": target_date,
        "total": len(all_students),
        "present": len(present_ids),
        "absent": len(all_students) - len(present_ids),
        "records": result,
    }


@app.post("/api/attendance/manual")
def manual_attendance(student_id: int = Form(...), status: str = Form("present"), db: Session = Depends(get_db)):
    today = date.today().isoformat()
    existing = db.query(AttendanceRecord).filter(
        AttendanceRecord.student_id == student_id, AttendanceRecord.date == today
    ).first()

    if status == "present" and not existing:
        record = AttendanceRecord(student_id=student_id, date=today, confidence=1.0, method="manual")
        db.add(record)
        student = db.query(Student).filter(Student.id == student_id).first()
        if student:
            student.total_attendance += 1
        db.commit()
    elif status == "absent" and existing:
        db.delete(existing)
        db.commit()

    return {"message": "Attendance updated"}


# ─── Analytics ────────────────────────────────────────────────

@app.get("/api/analytics/summary")
def analytics_summary(db: Session = Depends(get_db)):
    total_students = db.query(Student).count()
    today = date.today().isoformat()
    present_today = db.query(AttendanceRecord).filter(AttendanceRecord.date == today).count()

    return {
        "totalStudents": total_students,
        "presentToday": present_today,
        "absentToday": total_students - present_today,
        "attendanceRate": round(present_today / max(1, total_students) * 100, 1),
    }


@app.get("/api/activity-logs")
def get_activity_logs(limit: int = 50, db: Session = Depends(get_db)):
    logs = db.query(ActivityLog).order_by(ActivityLog.timestamp.desc()).limit(limit).all()
    return [
        {
            "id": l.id,
            "studentId": l.student_id,
            "studentName": l.student_name,
            "activity": l.activity,
            "attentionScore": l.attention_score,
            "emotion": l.emotion,
            "timestamp": l.timestamp.isoformat(),
        }
        for l in logs
    ]


# ─── Camera Stream (Server-side camera) ──────────────────────

@app.post("/api/camera/start")
def start_camera():
    global camera_capture, is_monitoring
    if is_monitoring:
        return {"message": "Already monitoring"}
        
    # Scan available cameras to automatically pick the external USB webcam
    available = []
    for i in range(5):
        c = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if c.isOpened():
            ret, frame = c.read()
            if ret:
                available.append(i)
            c.release()
            
    if not available:
        raise HTTPException(500, "No active camera device detected. Please verify your webcam is connected.")
        
    # Auto-select: prioritize external USB cameras (higher index) over default integrated (index 0)
    selected_idx = available[-1] if len(available) > 1 else available[0]
    log.info(f"FastAPI Start Camera: Available indices {available}. Selecting index {selected_idx}.")
    
    camera_capture = cv2.VideoCapture(selected_idx)
    if not camera_capture.isOpened():
        raise HTTPException(500, f"Failed to open selected camera index {selected_idx}.")
        
    # Request 4K UHD resolution (3840x2160)
    camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    
    w = int(camera_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(camera_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info(f"FastAPI Camera started successfully: index {selected_idx} at {w}x{h} resolution.")
        
    is_monitoring = True
    return {"message": f"Camera started on index {selected_idx} at {w}x{h}"}


@app.post("/api/camera/stop")
def stop_camera():
    global camera_capture, is_monitoring
    is_monitoring = False
    if camera_capture:
        camera_capture.release()
        camera_capture = None
    return {"message": "Camera stopped"}


def generate_frames():
    global camera_capture, is_monitoring, latest_detections
    while is_monitoring and camera_capture and camera_capture.isOpened():
        ret, frame = camera_capture.read()
        if not ret:
            break

        # Process with AI
        faces = face_engine.process_frame(frame)
        latest_detections = faces
        h, w = frame.shape[:2]

        # Draw overlays
        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            score = face["attention_score"]
            activity = face["activity"]
            name = "Unknown"
            if face["matched_student_id"]:
                name = f"Seat {face['matched_student_id']}"

            # Box color based on attention: Green (Attentive), Yellow (Moderate), Red (Distracted)
            color = (0, 255, 0) if score >= 75 else (0, 255, 255) if score >= 55 else (0, 0, 255)

            # Draw box and core label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{name} | {activity.upper()} | {score:.0f}%"
            cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            # Detailed metrics overlays
            hp = face.get("head_pose", {"yaw": 0.0, "pitch": 0.0, "roll": 0.0})
            eye_op = face.get("eye_openness", 1.0)
            mouth_op = face.get("mouth_open", 0.0)
            emo = face.get("emotion", "neutral")
            phone = face.get("phone_detected", False)
            crop = face.get("crop_used", "none")

            # Draw details panel below bbox
            details_y = y2 + 15
            pose_str = f"Yaw:{hp['yaw']:+.1f} Pitch:{hp['pitch']:+.1f} ({crop})"
            cv2.putText(frame, pose_str, (x1, min(h - 10, details_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)

            details_y += 15
            metrics_str = f"Eye:{eye_op:.2f} Mouth:{mouth_op:.2f} [{emo}]"
            cv2.putText(frame, metrics_str, (x1, min(h - 10, details_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)

            # Draw alert header for phone usage
            if phone:
                cv2.rectangle(frame, (x1, y1 - 32), (x2, y1 - 5), (0, 0, 255), -1)
                cv2.putText(frame, "!!! PHONE !!!", (x1 + 5, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Highlight seat mapping on screen
            cv2.circle(frame, ((x1+x2)//2, (y1+y2)//2), 4, color, -1)

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


@app.get("/api/camera/stream")
def video_stream():
    if not is_monitoring:
        raise HTTPException(400, "Camera not started")
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/camera/detections")
def get_latest_detections():
    events = list(video_processor.pending_attendance_events)
    video_processor.pending_attendance_events.clear()
    return {
        "faces": [
            {
                "bbox": f["bbox"],
                "attentionScore": f["attention_score"],
                "activity": f["activity"],
                "emotion": f["emotion"],
                "matchedStudentId": f["matched_student_id"],
                "matchConfidence": f["match_confidence"],
                "headPose": f["head_pose"],
                "eyeOpenness": f["eye_openness"],
                "mouthOpen": f["mouth_open"],
                "phoneDetected": f.get("phone_detected", False),
                "laptopDetected": f.get("laptop_detected", False),
                "earphoneDetected": f.get("earphone_detected", False),
                "gridCell": f.get("grid_cell", ""),
                "cropUsed": f.get("crop_used", "none"),
            }
            for f in latest_detections
        ],
        "attendanceEvents": events,
    }


# ─── Session Management ───────────────────────────────────────

@app.post("/api/sessions/start")
def start_session(
    session_name: str = Form(""),
    db: Session = Depends(get_db),
):
    """Start a new monitoring session."""
    existing = db.query(SessionModel).filter(SessionModel.status == "active").first()
    if existing:
        raise HTTPException(409, "A session is already active. Stop it before starting a new one.")

    today = date.today().isoformat()
    name = session_name.strip() if session_name.strip() else f"Session {today}"
    sid  = str(uuid.uuid4())

    record = SessionModel(
        session_id=sid,
        session_name=name,
        start_time=datetime.utcnow(),
        status="active",
    )
    db.add(record)
    db.commit()

    video_processor.reset()
    log.info(f"[SESSION] Started: {sid} — {name}")
    return {"session_id": sid, "message": "Session started"}


@app.post("/api/sessions/stop")
def stop_session(db: Session = Depends(get_db)):
    """Stop the active session and generate reports."""
    import concurrent.futures, time as _time

    active = db.query(SessionModel).filter(SessionModel.status == "active").first()
    if not active:
        raise HTTPException(404, "No active session found.")

    active.status   = "complete"
    active.end_time = datetime.utcnow()

    # Compute session stats
    states = video_processor.student_states
    active.student_count = len(states)
    if states:
        avgs = [st.avg for st in states.values()]
        active.avg_class_attention = round(sum(avgs) / len(avgs), 1)

    db.commit()

    # Generate and persist reports (30-second timeout)
    session_duration = (active.end_time - active.start_time).total_seconds()
    session_date     = active.start_time.strftime("%Y-%m-%d")

    def _persist():
        _db = SessionLocal()
        try:
            reports = []
            for seat_id, st in states.items():
                tracker = video_processor.behavior_trackers.get(seat_id)
                if tracker is None:
                    from core.behavior_tracker import BehaviorTracker
                    tracker = BehaviorTracker(video_processor.session_start_time)
                report = report_gen.generate_student_report(
                    student_state=st,
                    behavior_tracker=tracker,
                    session_id=active.session_id,
                    session_date=session_date,
                    session_duration=session_duration,
                    frame_skip=3,
                    source_fps=25.0,
                    student_name=f"Seat {seat_id}",
                    grid_label=str(seat_id),
                    attendance_status="absent",
                )
                reports.append(report)
            report_gen.persist_reports(reports, active.session_id, _db)
        except Exception as e:
            log.warning(f"[SESSION] Report persistence error: {e}")
        finally:
            _db.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_persist)
        try:
            fut.result(timeout=30)
        except concurrent.futures.TimeoutError:
            log.warning("[SESSION] Report finalization exceeded 30s — returning response anyway.")

    log.info(f"[SESSION] Stopped: {active.session_id}")
    return {"session_id": active.session_id, "message": "Session stopped"}


@app.get("/api/sessions")
def list_sessions(db: Session = Depends(get_db)):
    """Return all sessions ordered by start_time descending."""
    sessions = db.query(SessionModel).order_by(SessionModel.start_time.desc()).all()
    return [
        {
            "session_id":          s.session_id,
            "session_name":        s.session_name,
            "start_time":          s.start_time.isoformat() if s.start_time else None,
            "end_time":            s.end_time.isoformat() if s.end_time else None,
            "status":              s.status,
            "student_count":       s.student_count,
            "avg_class_attention": s.avg_class_attention,
        }
        for s in sessions
    ]


# ─── Reports ─────────────────────────────────────────────────

@app.get("/api/reports/session/{session_id}")
def get_session_reports(session_id: str, db: Session = Depends(get_db)):
    """Return all student reports for a session."""
    rows = db.query(SessionReport).filter(SessionReport.session_id == session_id).all()
    if not rows:
        raise HTTPException(404, f"No reports found for session {session_id}")
    return [json.loads(r.report_json) for r in rows]


@app.get("/api/reports/student/{student_id}")
def get_student_reports(student_id: int, db: Session = Depends(get_db)):
    """Return all historical reports for a student."""
    student = db.query(Student).filter(Student.id == student_id).first()
    if not student:
        raise HTTPException(404, "Student not found")
    rows = (
        db.query(SessionReport)
        .filter(SessionReport.student_id == student_id)
        .order_by(SessionReport.created_at.desc())
        .all()
    )
    return [json.loads(r.report_json) for r in rows]


# ─── Seat Assignment ──────────────────────────────────────────

@app.post("/api/students/{student_id}/assign-seat")
def assign_seat(
    student_id: int,
    grid_cell: str = Form(...),
    db: Session = Depends(get_db),
):
    """Assign a student to a GridCell (e.g. R1C2)."""
    if not re.fullmatch(r"R\d+C\d+", grid_cell):
        raise HTTPException(422, f"Invalid grid_cell format '{grid_cell}'. Expected pattern R{{row}}C{{col}} e.g. R1C2")
    student = db.query(Student).filter(Student.id == student_id).first()
    if not student:
        raise HTTPException(404, "Student not found")
    student.grid_cell = grid_cell
    db.commit()
    return {"message": "Seat assigned"}


# ─── Video File Analysis ──────────────────────────────────────

@app.post("/api/analyze-video")
async def analyze_video_file(
    video: UploadFile = File(...),
    frameSkip: int = Form(3),
):
    """
    Upload a video file and receive a streaming NDJSON response.
    Each line is a JSON object: metadata | frame | complete.
    """
    # Save upload to temp file
    suffix = os.path.splitext(video.filename)[-1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await video.read()
        tmp.write(contents)
        tmp_path = tmp.name

    log.info(f"[UPLOAD] Received video '{video.filename}' ({len(contents)//1024} KB) -> {tmp_path}")

    def generate():
        try:
            for event in video_processor.process_video_file(tmp_path, frame_skip=frameSkip):
                yield json.dumps(event) + "\n"
        finally:
            try:
                os.unlink(tmp_path)
                log.info(f"[DELETE] Temp file removed: {tmp_path}")
            except Exception:
                pass

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"X-Content-Type-Options": "nosniff"},
    )


@app.get("/api/video-list")
def list_test_videos():
    """Return available test video files from the project folder."""
    test_dir = os.path.join(os.path.dirname(__file__), "..", "test video")
    test_dir = os.path.normpath(test_dir)
    videos = []
    if os.path.isdir(test_dir):
        for f in os.listdir(test_dir):
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                full = os.path.join(test_dir, f)
                videos.append({
                    "filename": f,
                    "sizeBytes": os.path.getsize(full),
                    "path": full,
                })
    log.info(f"[DIR] Listed {len(videos)} test video(s)")
    return {"videos": videos}


@app.post("/api/analyze-video-path")
async def analyze_video_by_path(
    video_path: str = Form(...),
    frameSkip: int = Form(3),
):
    """
    Analyze a video file by its server-side path (for local test videos).
    Streams NDJSON back.
    """
    if not os.path.isfile(video_path):
        raise HTTPException(404, f"File not found: {video_path}")

    log.info(f"[ANALYZE] Analyzing by path: {video_path} (frameSkip={frameSkip})")

    def generate():
        for event in video_processor.process_video_file(video_path, frame_skip=frameSkip):
            yield json.dumps(event) + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"X-Content-Type-Options": "nosniff"},
    )


# ─── Ollama Endpoints ─────────────────────────────────────────

@app.post("/api/ollama/status")
def ollama_status():
    """Check if Ollama is running and return available model."""
    available = ollama_labeler.is_available()
    model = ollama_labeler.get_model() if available else None
    return {"available": available, "model": model or ""}


@app.post("/api/ollama/label-seats")
def ollama_label_seats():
    """Run OllamaLabeler on current spatial grid, return seat→student mapping."""
    # Gather current grid cells from spatial grid
    grid_cells = []
    if video_processor.spatial_grid and video_processor.spatial_grid.cells:
        grid_cells = [c.label for c in video_processor.spatial_grid.cells]
    else:
        # Generate default grid cells based on student states
        seat_ids = list(video_processor.student_states.keys())
        if not seat_ids:
            grid_cells = [f"R{r}C{c}" for r in range(1, 3) for c in range(1, 4)]
        else:
            grid_cells = [f"S{sid}" for sid in seat_ids]

    # Build detected persons list
    detected_persons = []
    for sid, st in video_processor.student_states.items():
        detected_persons.append({
            "seat_id": sid,
            "grid_cell": f"S{sid}",
            "attention_score": st.avg,
            "activity": st.reconciled_activity()[0],
        })

    mapping = ollama_labeler.label_seats(grid_cells, detected_persons)
    return {"mapping": mapping, "grid_cells": grid_cells}


@app.get("/api/ollama/analyze")
def ollama_analyze_classroom(db: Session = Depends(get_db)):
    """Get Ollama natural language analysis of the entire classroom based on latest session reports."""
    # Gather latest session
    latest_session = (
        db.query(SessionModel)
        .order_by(SessionModel.start_time.desc())
        .first()
    )

    student_reports = []
    if latest_session:
        rows = db.query(SessionReport).filter(SessionReport.session_id == latest_session.session_id).all()
        for row in rows:
            try:
                student_reports.append(json.loads(row.report_json))
            except Exception:
                pass

    # Build summary data for Ollama
    if not student_reports:
        # Fall back to live student states
        for sid, st in video_processor.student_states.items():
            student_reports.append({
                "student_name": f"Seat {sid}",
                "avg_attention": st.avg,
                "activity": st.reconciled_activity()[0],
                "session_duration_minutes": (time_module.time() - video_processor.session_start_time) / 60,
            })

    if not student_reports:
        return {"summary": "No classroom data available yet. Start a session and let the AI observe the class."}

    summary = ollama_labeler.analyze_classroom(student_reports)
    return {"summary": summary, "student_count": len(student_reports)}


@app.post("/api/ollama/analyze-student/{seat_id}")
def ollama_analyze_student(seat_id: str, db: Session = Depends(get_db)):
    """Get Gemma natural language analysis of a student by seat_id or grid_cell."""
    # Try to find student by grid_cell
    student = db.query(Student).filter(Student.grid_cell == seat_id).first()
    student_name = student.name if student else f"Student at {seat_id}"

    # Get state from video processor
    state_data = {}
    for sid, st in video_processor.student_states.items():
        cell = getattr(st, "grid_cell", None) or str(sid)
        if cell == seat_id or str(sid) == seat_id:
            state_data = {
                "seat_id": sid,
                "student_name": student_name,
                "grid_cell": seat_id,
                "activity": st.reconciled_activity()[0],
                "attention_score": st.avg,
                "avg_attention": st.avg,
                "emotion": "neutral",
                "session_duration_minutes": (
                    (time_module.time() - video_processor.session_start_time) / 60
                ),
            }
            break

    if not state_data:
        state_data = {
            "seat_id": seat_id,
            "student_name": student_name,
            "grid_cell": seat_id,
            "activity": "unknown",
            "attention_score": 0,
            "avg_attention": 0,
            "session_duration_minutes": 0,
        }

    analysis = ollama_labeler.analyze_behavior(state_data)
    return {"seat_id": seat_id, "student_name": student_name, "analysis": analysis}


# ─── Face Recognition Status ──────────────────────────────────

@app.get("/api/face-recognition/status")
def face_recognition_status():
    """Check face recognition backend and return status."""
    from core.face_recognition_engine import face_rec_engine
    return {
        "loaded": True,
        "backend": face_rec_engine.backend,
        "is_reliable": face_rec_engine.is_reliable,
        "registered_count": len(video_processor.registered_faces),
        "warning": (
            "Using histogram fallback \u2014 identity matching is unreliable"
            if not face_rec_engine.is_reliable else ""
        ),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
