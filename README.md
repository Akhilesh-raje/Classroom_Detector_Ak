# SmartClass AI

An AI-powered classroom monitoring system that detects students in real-time, tracks attention levels, classifies behaviour, and generates session reports — all running locally with no cloud dependency.

---

## Table of Contents

1. [What it does](#what-it-does)
2. [Project structure](#project-structure)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Running the system](#running-the-system)
   - [Live webcam](#1-live-webcam-monitoring)
   - [Video file analysis](#2-video-file-analysis)
   - [Classroom scanner](#3-classroom-seat-scanner)
   - [Web dashboard](#4-web-dashboard)
6. [Key controls](#key-controls)
7. [Configuration options](#configuration-options)
8. [AI pipeline explained](#ai-pipeline-explained)
9. [Output files](#output-files)
10. [Performance notes](#performance-notes)
11. [Privacy](#privacy)

---

## What it does

- Detects every student in the frame using YOLO11s (person detection)
- Assigns a stable seat ID to each person that persists across the session
- Runs MediaPipe FaceLandmarker to extract 468 facial landmarks per student
- Computes an attention score (0–100%) per student per frame based on:
  - Head pose (yaw, pitch, roll)
  - Eye openness
  - Motion stability
- Classifies current behaviour: `studying`, `attentive`, `neutral`, `talking`, `distracted`, `drowsy`, `phone`, `laptop`, `music`
- Ghost-holds seats that temporarily disappear (occlusion, blink) so the count never flickers
- Interpolates bounding box positions between AI frames for smooth overlay
- Writes a per-seat summary report and an annotated video at the end

---

## Project structure

```
classroom/
│
├── run_webcam.py              # Live webcam monitoring
├── run_video.py               # Video file runner (general purpose)
├── test_real_classroom.py     # Real-time threaded video tester (main script)
├── scan_classroom.py          # Scan empty classroom to map seats
│
├── backend/
│   ├── main.py                # FastAPI server (REST API for web dashboard)
│   ├── requirements.txt       # Python dependencies
│   ├── smartclass.db          # SQLite database (auto-created)
│   └── core/
│       ├── config.py          # All tunable parameters
│       ├── engine.py          # Unified AI orchestrator
│       ├── detector.py        # YOLO tracking + MediaPipe crop generation
│       ├── stabilizer.py      # Seat ID assignment + ghost hold
│       ├── student_state.py   # Per-student attention + activity state machine
│       ├── behavior_tracker.py# Activity timeline recording
│       ├── face_recognition_engine.py  # Face ID (InsightFace / fallback)
│       ├── classroom_mapper.py# Desk layout detection
│       ├── visualizer.py      # Shared overlay rendering functions
│       ├── report_generator.py# Session report builder
│       ├── ollama_labeler.py  # Optional LLM seat naming via Ollama
│       └── database.py        # SQLAlchemy models + DB helpers
│
├── models/
│   ├── yolo11s.pt             # YOLO11 small — person detection weights
│   └── face_landmarker.task   # MediaPipe FaceLandmarker task file
│
├── src/                       # React frontend
│   ├── App.jsx
│   ├── api.js                 # API client
│   ├── main.jsx
│   ├── index.css
│   └── pages/
│       ├── Classroom.jsx      # Live monitoring view
│       ├── Students.jsx       # Student registration
│       ├── Sessions.jsx       # Session history
│       ├── Reports.jsx        # Analytics & charts
│       └── TestVideo.jsx      # Upload & test a video file
│
├── tests/
│   ├── brutal_test.py         # Full automated test suite (65 tests)
│   ├── test_all.py
│   ├── test_fixes.py
│   ├── test_scanner.py
│   └── test_webcam_mock.py
│
├── outputs/                   # Generated files (gitignored)
│   ├── annotated_output.mp4
│   ├── real_classroom_annotated.mp4
│   ├── real_classroom_results.txt
│   ├── classroom_layout.json
│   ├── classroom_layout.png
│   └── classroom_layout_grid.png
│
├── test video/                # Sample input videos
│   ├── test video - real classrom.mp4
│   └── Student Sitting in the Class ....mp4
│
├── scratch/                   # Experimental scripts (not production)
│
├── index.html                 # Vite entry point
├── vite.config.js
├── package.json
└── README.md
```

---

## Requirements

### Python
- Python 3.10 or newer
- pip

### Node.js (only needed for the web dashboard)
- Node.js 18 or newer

### Hardware
- CPU: Any modern CPU. Intel i5/i7 or AMD Ryzen 5/7 recommended.
- GPU: Optional but dramatically improves speed (NVIDIA CUDA or any GPU with ONNX support)
- RAM: 8 GB minimum, 16 GB recommended for full classroom (12+ students)
- Camera: Any USB webcam for live mode; 1080p recommended

---

## Installation

### 1. Clone / download the project

```bash
git clone <repo-url>
cd classroom
```

### 2. Install Python dependencies

```bash
pip install -r backend/requirements.txt
```

> If `insightface` or `face-recognition` fail to install (they require a C compiler),
> the system will fall back to a histogram-based identity method automatically.
> Everything else still works normally.

### 3. Install Node.js dependencies (web dashboard only)

```bash
npm install
```

### 4. Verify models are present

```
models/yolo11s.pt             (~22 MB)
models/face_landmarker.task   (~1 MB)
```

Both files must be present. They are not downloaded automatically.

---

## Running the system

### 1. Live webcam monitoring

```bash
python run_webcam.py
```

Opens your webcam, detects students in real time, and displays overlays.
Press `Q` to stop. A session report is saved to `outputs/` automatically.

---

### 2. Video file analysis

**Main script (real-time threaded pipeline):**

```bash
python test_real_classroom.py
```

Runs on the default video: `test video/test video - real classrom.mp4`

To use a different video:

```bash
python test_real_classroom.py "path/to/your/video.mp4"
```

**What happens:**
1. Both AI models are pre-loaded and warmed up (~15–50s, one-time)
2. A preview window opens showing the video at real video speed
3. AI runs in a background thread every 2 frames
4. Bounding boxes are interpolated between AI results — no snapping or jumping
5. When done (or you press Q): annotated video + text report saved to `outputs/`

**General video runner** (simpler, single-threaded):

```bash
python run_video.py
python run_video.py "path/to/video.mp4"
```

---

### 3. Classroom seat scanner

Used once to map the empty classroom before a session.

```bash
python scan_classroom.py
```

**What it does:**
- Opens your camera
- Detects all chairs, desks, and benches using YOLO + colour detection + edge detection
- Groups detections into rows (R1S1, R1S2, R2S1, etc.)
- Saves a `classroom_layout.json` with pixel coordinates of every seat
- Saves annotated images showing the detected layout

**Controls during scanning:**

| Key | Action |
|-----|--------|
| `SPACE` | Scan current frame |
| `S` | Save layout to JSON |
| `R` | Retry / clear detections |
| `N` | Switch to next camera |
| `0–9` | Select camera by index |
| `+` / `=` | Zoom in |
| `-` / `_` | Zoom out |
| `W A S D` | Pan when zoomed |
| `0` | Reset zoom to 1x |
| `Q` / `ESC` | Quit |

---

### 4. Web dashboard

Start the backend API server:

```bash
python backend/main.py
```

Start the frontend:

```bash
npm run dev
```

Open `http://localhost:3000` in your browser.

**Dashboard pages:**

| Page | What it shows |
|------|--------------|
| Classroom | Live camera feed with per-student overlays |
| Students | Register students with face photos |
| Sessions | History of past monitoring sessions |
| Reports | Attention charts, activity breakdowns, seat heatmaps |
| Test Video | Upload and analyse a video file via the browser |

---

## Key controls

These apply to `test_real_classroom.py` and `run_video.py` during playback:

| Key | Action |
|-----|--------|
| `D` | Toggle detail mode — switches between **full panel** (all metrics) and **compact** (just activity label + attention bar) |
| `SPACE` | Pause / resume playback |
| `+` or `=` | Speed up playback (up to 4x). AI continues at the same rate; display advances faster. |
| `-` or `_` | Slow down playback (minimum 0.25x) |
| `Q` or `ESC` | Quit and save report |

### Detail mode (D key)

**Full mode** shows per student:
- Seat ID and grid cell label
- Current activity icon (STUDY, FOCUS, IDLE, TALK, etc.)
- Attention score (current frame)
- Average attention (rolling 90-frame window)
- Detected emotion
- Head yaw and pitch angles
- Eye openness ratio
- Attention factor breakdown (head pose, motion stability)
- Activity reconciliation note (e.g. "momentarily distracted")
- Phone / laptop alert banners

**Compact mode** shows per student:
- Seat ID + activity label + attention % in a single badge
- One colour-coded attention bar below the bounding box
- Phone alert if detected

---

## Configuration options

All tunable parameters are in `backend/core/config.py`:

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `YOLO_MODEL_PATH` | `yolo11s.pt` | Path to YOLO weights file |
| `YOLO_CONF` | `0.18` | Minimum confidence for person detection. Lower = more detections but more false positives. Raise to 0.30–0.40 to reduce ghost boxes. |
| `PERSIST` | `True` | Whether YOLO uses tracking (keeps IDs stable between frames). Keep True. |
| `MAX_SEATS` | `6` | Maximum number of seat IDs to track. Set this to the actual number of students in the room. |
| `SEAT_PROXIMITY_THRES` | `0.25` | How close (as fraction of frame diagonal) a detection must be to an existing seat to be assigned to it. Lower = stricter matching. |
| `SMALL_BBOX_FRAC` | `0.02` | Face crops smaller than 2% of frame area use the upper-body crop instead of the tight face crop. |
| `FACE_MATCH_THRESHOLD` | `0.75` | Cosine similarity required to confirm a face identity match. |
| `FACE_MATCH_INTERVAL` | `30` | Run face recognition every N frames per seat (not every frame — expensive). |
| `HYSTERESIS_FRAMES` | `5` | A new activity label must be stable for 5 consecutive frames before being committed. Prevents rapid flickering. |
| `EMA_A` | `0.12` | Exponential moving average alpha for attention smoothing. Lower = smoother but slower to respond. |
| `HIST_LEN` | `90` | Number of frames kept in the attention history. 90 frames ≈ 3 seconds at 30fps. |
| `DASH_EVERY` | `10` | Print a console dashboard line every N processed frames. |

**Script-level overrides** (in `test_real_classroom.py`):

| Variable | Default | What it controls |
|----------|---------|-----------------|
| `MAX_SEATS` | `15` | Overrides config for this script |
| `YOLO_CONF_OV` | `0.30` | Overrides YOLO confidence threshold |
| `IOU_SUPPRESS` | `0.45` | IoU threshold for suppressing overlapping person boxes. Two boxes with overlap > 45% → keep only the higher-confidence one. |
| `AI_EVERY` | `2` | Run full AI (YOLO + MediaPipe) every N frames. Between AI frames, last result is reused with interpolated bboxes. Lower = more accurate but slower. |

---

## AI pipeline explained

```
Video frame
    │
    ▼
YOLO11s track()
    │  detects all persons with tracking IDs
    ▼
SeatStabilizer.update_and_map()
    │  assigns stable seat IDs (1..N)
    │  ghost-holds seats missing for up to 8 frames
    ▼
For each seat:
    │
    ├─ Phone / laptop detection (YOLO class 67, 63)
    │
    ├─ MediaPipe FaceLandmarker
    │    → 468 face landmarks
    │    → head pose (yaw, pitch, roll via solvePnP)
    │    → eye openness (blink blendshape)
    │    → emotion (smile, frown, brow blendshapes)
    │
    ├─ StudentState.compute_attn()
    │    → attention score 0–100%
    │    → factors: head_pose, eye, motion, face_visibility
    │
    ├─ StudentState.classify_act()
    │    → activity label with 5-frame hysteresis
    │
    └─ StudentState.reconciled_activity()
         → corrects label against session avg
           (e.g. peak 86% but labelled distracted → attentive)
    │
    ▼
SharedState.update()  (thread-safe)
    │
    ▼
Display thread — reads interpolated bboxes
    │  lerps box positions between AI frames → smooth overlay
    ▼
draw_full_overlay() or draw_compact_overlay()
    │
    ▼
cv2.imshow() at real video fps
```

### Attention score formula

```
attention = (
    motion_score    × 0.25 +
    head_pose_score × 0.60 +
    eye_score       × 0.15
) × 100 × eye_penalty
```

- **motion_score**: velocity, oscillation, area variance, drift (penalises fidgeting)
- **head_pose_score**: penalises yaw > 35° or pitch > 25° (natural writing/reading movement is not penalised)
- **eye_score**: raw eye openness from MediaPipe blink blendshape
- **eye_penalty**: multiplier that scales to 0 when eyes are nearly fully closed (< 0.3)

### Activity classification priority order

1. `phone` — phone detected in person's area
2. `drowsy` — eyes nearly closed for 5+ consecutive frames
3. `talking` — mouth open > 30% AND head oscillation detected
4. `distracted` — yaw > 35° AND attention < 65%
5. `laptop` — laptop detected nearby
6. `studying` — attention ≥ 72%, no electronics, low motion
7. `attentive` — attention ≥ 62%
8. `music` — earphone detected
9. `fidgeting` — high area variance, low velocity
10. `distracted` / `neutral` — kinematic fallback

---

## Output files

| File | Location | Contents |
|------|----------|----------|
| `real_classroom_annotated.mp4` | `outputs/` | Full video with all overlays burned in |
| `real_classroom_results.txt` | `outputs/` | Per-seat: avg attention, peak, low, final activity, class average |
| `classroom_layout.json` | `outputs/` | Seat map from scanner: pixel coords, row/col labels, seat names |
| `classroom_layout.png` | `outputs/` | Annotated photo showing detected seats |
| `classroom_layout_grid.png` | `outputs/` | Clean grid overlay on classroom photo |
| `smartclass.db` | `backend/` | SQLite database with students, sessions, attendance records |

---

## Performance notes

| Condition | Approx speed |
|-----------|-------------|
| CPU only, 12 students, 1080p | ~400–500ms per AI frame (~2–3 fps AI, 15 fps display) |
| CPU only, 6 students, 720p | ~200–250ms per AI frame (~4 fps AI, 25 fps display) |
| NVIDIA GPU (CUDA) | ~30–50ms per AI frame (~20+ fps AI, full 30 fps display) |

**To improve performance:**

1. Export YOLO to ONNX format:
   ```bash
   yolo export model=models/yolo11s.pt format=onnx
   ```
   Then change `YOLO_MODEL_PATH = "yolo11s.onnx"` in `config.py`.

2. Increase `AI_EVERY` in `test_real_classroom.py` (e.g. `AI_EVERY = 4`) to run AI less frequently. The display stays smooth due to interpolation.

3. Reduce resolution: add `frame = cv2.resize(frame, (1280, 720))` before submitting to the AI thread.

4. Use a GPU: install `onnxruntime-gpu` and the CUDA-enabled `torch` build for ultralytics.

---

## Privacy

All processing runs **entirely on your local machine**.

- No video, images, or biometric data is sent to any external server
- Face encodings are stored only in the local SQLite database (`backend/smartclass.db`)
- The database is never synced or uploaded anywhere
- To delete all stored data: `del backend\smartclass.db`
