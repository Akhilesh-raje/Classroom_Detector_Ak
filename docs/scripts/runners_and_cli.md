# 🚀 CLI & Execution Scripts

SmartClass AI provides a unified command-line tool along with standalone execution scripts for running headless batch analyses, camera pipelines, topology scanners, and database diagnostics.

---

## 🛠️ Master CLI (`cli.py`)

The unified entry point [cli.py](file:///c:/Users/rajea/Documents/classroom/cli.py) provides subcommands for all system workflows:

```bash
python cli.py <subcommand> [flags]
```

### Available Subcommands

| Subcommand | Description | Flags |
| :--- | :--- | :--- |
| `backend` | Runs the FastAPI production/development server. | `--port 8000`, `--reload` |
| `frontend` | Starts the Vite React development server. | None |
| `webcam` | Launches real-time USB webcam video capture. | None |
| `test-classroom` | Executes real classroom verification runner. | `--structure-ai`, `--video <path>` |
| `structure-ai` | Runs Structure AI Comprehensive Exporter. | `--video <path>` |

---

## 📹 Video File Runner (`run_video.py`)

Executes offline inference over an MP4 video file and outputs annotated MP4 video, forensic metrics, and benchmark logs:

```bash
python run_video.py "test video/test video - real classrom.mp4" --output annotated_output.mp4 --frame-skip 3
```

### Arguments:
- `video`: Path to MP4/AVI/MKV video file (defaults to first video in `test video/`).
- `--output`, `-o`: Filepath for saving annotated video with HUD overlays.
- `--headless`: Disables OpenCV GUI preview window for server/CI environments.
- `--max-seats`: Maximum number of seats to discover and track (default `12`).
- `--frame-skip`: Frame skip factor for AI processing acceleration (default `3`).

---

## 📷 Live Webcam Runner (`run_webcam.py`)

Connects to connected USB / RTSP cameras using DirectShow (`cv2.CAP_DSHOW`):

```bash
python run_webcam.py
```

### Key Behaviors:
- **Automatic Resolution Optimization**: Probes 4K UHD ($3840 \times 2160$), falling back gracefully to 1080p ($1920 \times 1080$) to avoid blurry default $640 \times 480$ captures.
- **Interactive Keyboard Controls**:
  - `SPACE`: Freeze frame and inspect detections.
  - `S`: Save snapshot and instant session report.
  - `R`: Reset spatial anchors.
  - `Q`: Graceful shutdown and camera hardware release.

---

## 📐 Classroom Scanner (`scan_classroom.py`)

Scans an empty classroom to discover permanent seating topology before students arrive:

```bash
python scan_classroom.py
```

### Outputs Generated:
- `classroom_layout.json`: Full seat map with bounding box pixel coordinates and physical names.
- `classroom_layout.png`: High-resolution annotated image highlighting discovered desks.
- `classroom_layout_grid.png`: Topological grid lines superimposed on the empty room.

---

## 🧪 Helper Scripts

### 1. `register_test_students.py`
Automated test fixture that extracts face crops from `test video/test video - real classrom.mp4` and enrolls sample students (`Priya Sharma`, `Ananya Gupta`, etc.) into the SQLite database with 512-D embeddings.

### 2. `check_db.py`
Console utility for inspecting SQLite table counts, student rosters, enrolled face encodings, and recent attendance records.

### 3. `test_backend_stream.py`
Tests and verifies HTTP MJPEG video streams from the running FastAPI backend without requiring a web browser.
