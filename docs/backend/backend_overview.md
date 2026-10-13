# 🖥️ Backend Architecture Overview

The SmartClass AI backend is a high-throughput, enterprise-ready Python service built on top of **FastAPI** ([backend/main.py](file:///c:/Users/rajea/Documents/classroom/backend/main.py)). It serves REST APIs, streams real-time MJPEG video with Computer Vision HUD overlays, pushes low-latency telemetry updates, and orchestrates offline background analytics.

---

## 🏛️ Server Structure & Lifespan

The server initializes critical components during the asynchronous application lifespan:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup:
    init_db()                           # Run SQLAlchemy table migrations
    db = SessionLocal()
    try:
        students = db.query(Student).filter(Student.face_encoding.isnot(None)).all()
        face_engine.load_registered_faces(students)       # Cache face embeddings
        video_processor.load_registered_faces(students)   # Warm engine cache
    finally:
        db.close()
    yield
    # Shutdown:
    global camera_capture, is_monitoring
    is_monitoring = False
    if camera_capture:
        camera_capture.release()        # Gracefully release camera hardware
```

---

## 📂 Backend Directory Structure

```text
backend/
├── core/                                # Enterprise Modular Domain Packages
│   ├── analytics/                       # Report generation & Ollama integration
│   │   ├── ollama_labeler.py            # Local multimodal VLM client
│   │   ├── report_generator.py          # Session aggregation & CSV/PDF exporter
│   │   └── __init__.py
│   ├── attendance/                      # Continuous attendance logic
│   │   ├── attendance_engine.py         # Temporal confirmation state machine
│   │   └── __init__.py
│   ├── behavior/                        # Student posture & attention tracking
│   │   ├── behavior_tracker.py          # MediaPipe Face Landmarker & PnP pose
│   │   ├── student_state.py             # Temporal attention state & smoothing
│   │   └── __init__.py
│   ├── detection/                       # Object detection & multi-target tracking
│   │   ├── detector.py                  # YOLO11s detector & ByteTrack integration
│   │   └── __init__.py
│   ├── pipeline/                        # Asynchronous pipeline stages
│   │   ├── analysis_stage.py            # Landmark analysis & identity checking
│   │   ├── association_stage.py         # Hungarian seat-to-person binding
│   │   ├── comprehensive_analyzer.py    # Offline artifact generator
│   │   ├── detection_stage.py           # Frame inference & track creation
│   │   ├── output_stage.py              # Scene state synthesis
│   │   ├── scheduler.py                 # Cadence manager (Fast/Med/Slow/Async)
│   │   └── __init__.py
│   ├── recognition/                     # Facial embeddings & identity matching
│   │   ├── face_engine.py               # In-memory face embedding index
│   │   ├── face_recognition_engine.py   # Cosine similarity & verification tiers
│   │   └── __init__.py
│   ├── scene/                           # Scene representations
│   │   ├── scene_state.py               # Universal SceneState & EntityState models
│   │   └── __init__.py
│   ├── spatial/                         # Seating geometry & coordinate mapping
│   │   ├── classroom_mapper.py          # Perspective projection & zone mapping
│   │   ├── seat_associator.py           # Multi-tier Hungarian seat associator
│   │   ├── seat_naming.py               # Standardized R1C1 / bench nomenclature
│   │   ├── spatial_engine.py            # Spatial Engine v1 & v2
│   │   ├── spatial_engine_v3.py         # Geometry-driven Spatial Engine v3
│   │   ├── stabilizer.py                # EMA coordinate stabilization
│   │   ├── topology.py                  # PhysicalSeat & ClassroomTopology models
│   │   └── __init__.py
│   ├── storage/                         # Database persistence
│   │   ├── database.py                  # SQLAlchemy models & SQLite setup
│   │   └── __init__.py
│   ├── structure/                       # 3D Depth & structural perception
│   │   ├── depth.py                     # Monocular depth map estimation
│   │   ├── detector.py                  # Structural feature detection
│   │   ├── geometry.py                  # 3D planes & ray projection
│   │   ├── segmenter.py                 # Bench & aisle semantic segmentation
│   │   ├── structure_engine.py          # Structural engine orchestrator
│   │   ├── topology_builder.py          # Physical seating synthesis
│   │   └── __init__.py
│   ├── visualization/                   # Video rendering & HUD
│   │   ├── visualizer.py                # High-contrast OpenCV visualizer
│   │   └── __init__.py
│   ├── config.py                        # Centralized hyperparameters & constants
│   ├── engine.py                        # ClassroomEngine orchestrator
│   └── __init__.py                      # Package entry point
├── experiments/                         # Diagnostic experiments & dual benchmarks
├── main.py                              # FastAPI entry point & API route handlers
├── migrate_attendance_v1_1.py           # Database schema migration script
└── requirements.txt                     # Python pip dependency requirements
```

---

## 📡 Streaming & Real-Time Media Architecture

### 1. MJPEG Streaming Endpoints
FastAPI streams live video using `multipart/x-mixed-replace; boundary=frame`:
- `/api/camera/stream`: Live camera video feed annotated with real-time bounding boxes, seat labels, attention pills, and FPS counter.
- `/api/live-classroom/stream`: Live simulated or test video feed for frontend preview.

### 2. Live Telemetry Endpoints
- `/api/live-classroom/telemetry`: High-frequency JSON endpoint polled by the frontend (or sent via WebSockets) to update seat occupancy cards, attention distribution bars, and activity counters.
- `/api/camera/detections`: Latest snapshot of detected person tracks and bounding coordinates.

---

## 🔒 Security & CORS Policies

- **CORS Configuration**: Configured with permissive headers for local LAN/development access (`allow_origins=["*"]`, `allow_credentials=True`).
- **Local Storage**: Biometric vectors are stored in local SQLite (`smartclass.db`). No external cloud storage or telemetry sync is performed.
