# ⚙️ Core Engine (`ClassroomEngine`)

The `ClassroomEngine` class ([backend/core/engine.py](file:///c:/Users/rajea/Documents/classroom/backend/core/engine.py)) serves as the master orchestrator in SmartClass AI. It integrates computer vision models, geometric topology, biometric verification, and temporal behavior analysis into a single, cohesive processing loop for both real-time cameras and offline video files.

---

## 🏛️ Class Overview

```python
class ClassroomEngine:
    def __init__(self, max_seats: int = 20): ...
```

### Key State & Properties
- `stabilizer`: Instance of `SeatStabilizer` managing continuous bounding box exponential smoothing.
- `associator`: Instance of `SeatAssociator` implementing Hungarian matching with hysteresis.
- `topology`: Active `ClassroomTopology` defining physical seat anchors and dimensions.
- `states`: Dictionary mapping `seat_id -> StudentState` tracking temporal attention, eye aspect ratios, and activity histograms.
- `trackers`: Dictionary mapping `seat_id -> BehaviorTracker` managing per-student MediaPipe pose solutions.
- `scheduler`: Instance of `PipelineScheduler` executing tiered frame cadences (Fast / Medium / Slow / Async).
- `stages`: Pre-initialized pipeline stages (`DetectionStage`, `AssociationStage`, `AnalysisStage`, `OutputStage`).

---

## 🔧 Core Methods & Lifecycles

### 1. `calibrate(frames: list, W_px: int, H_px: int, config: TopologyConfig | None = None)`
Takes a collection of $N$ calibration frames (typically 30–60 frames captured at startup or video start):
1. Runs person detection across the calibration set to identify candidate seat clusters.
2. Invokes `TopologyDiscovery.discover()` to synthesize the physical classroom seating matrix.
3. Automatically locks seat locations left-to-right and front-to-back.

### 2. `load_registered_faces(students: list)`
Synchronizes registered student records from the database into the face recognition engine's in-memory cosine cache:
- Iterates over enrolled students having valid `face_encoding` BLOBs.
- Builds an indexed numpy matrix of normalized 512-D vectors for vector similarity search.

### 3. `process_single_frame(frame, W_px, H_px, persist=True) -> list`
The primary real-time inference loop invoked by FastAPI's camera streaming worker:
1. **Detection Stage**: YOLO11s detects persons and devices (`phones`, `laptops`, `earphones`).
2. **Association Stage**: Maps detected person tracks to physical seat anchors via Hungarian matching.
3. **Analysis Stage**:
   - Computes head pose (Yaw/Pitch/Roll) and Eye Aspect Ratio (EAR).
   - Generates composite attention score (0–100%) and updates sliding window.
   - Triggers face recognition matching every $30$ frames (`FACE_MATCH_INTERVAL`).
4. **Output Stage**: Formats output list of student dictionaries and updates `SceneState`.

### 4. `process_video_file(video_path: str, frame_skip: int = 3)`
Batch processing loop for recorded MP4 video files:
- Opens `cv2.VideoCapture` and reads frames sequentially.
- Applies `frame_skip` (e.g. processing every 3rd frame) for accelerated offline analysis ($> 60$ effective FPS).
- Displays real-time terminal dashboard with colored ASCII attention progress bars.
- At completion, exports session summary report, student timelines, and benchmark CSVs.

---

## 📊 Console Dashboard Output

During CLI or video execution, `_print_dashboard()` renders a high-contrast ANSI terminal monitor:

```text
===========================================================================
  SmartClass AI   FPS: 28.4   Frame: 450/1400   Students: 12   Attn: 79.4%
===========================================================================
  Seat  Name           Roll No     Attn %   Status     Activity   Trend
  ─────────────────────────────────────────────────────────────────────────
  S1    Aarav S.       CS2026-001  84.5%    VERIFIED   studying    UP  
  S2    Diya P.        CS2026-002  91.0%    VERIFIED   attentive  STDY 
  S3    Rohan M.       CS2026-003  68.2%    VERIFIED   distracted  DN  
  S4    Sneha K.       CS2026-004  88.1%    VERIFIED   studying   STDY 
===========================================================================
```
