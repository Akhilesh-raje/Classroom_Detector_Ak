# ⚙️ Configuration Management (`config.py`)

Central system parameters and algorithmic hyperparameters are maintained in [backend/core/config.py](file:///c:/Users/rajea/Documents/classroom/backend/core/config.py). This document outlines each setting, its default value, and tuning guidelines.

---

## 📋 Parameter Reference

| Parameter Name | Default | Type | Description & Tuning Guide |
| :--- | :--- | :--- | :--- |
| `YOLO_MODEL_PATH` | `"yolo11s.pt"` | `str` | Path or model identifier for the Ultralytics YOLO model. Can be swapped with `yolo11n.pt` for ultra-low power devices or `yolo11m.pt` for high-resolution cameras. |
| `YOLO_CONF` | `0.18` | `float` | Minimum detection confidence threshold for persons and objects. Set conservatively ($0.18$) to detect partially occluded students in back rows. |
| `PERSIST` | `True` | `bool` | Enables temporal track persistence in YOLO ByteTrack (`persist=True`). |
| `MAX_SEATS` | `20` | `int` | Maximum physical seats supported in the default topology matrix. |
| `SEAT_PROXIMITY_THRES` | `0.25` | `float` | Maximum distance (as a fraction of frame diagonal) allowed when associating an unassigned track to a candidate seat anchor. |
| `SMALL_BBOX_FRAC` | `0.02` | `float` | Fraction of frame area below which a person bounding box is treated as distant/small. Used to adjust landmarking crop scales. |
| `FACE_MATCH_THRESHOLD` | `0.75` | `float` | Strict Cosine similarity threshold for authenticating face identity against registered database embeddings. |
| `FACE_MATCH_INTERVAL` | `30` | `int` | Number of frames between face recognition matches per seat. Throttling to 30 frames saves ~70% GPU compute without losing track fidelity. |
| `HYSTERESIS_FRAMES` | `5` | `int` | Consecutive frames required to confirm a transition between behavioral activities (e.g. from `attentive` to `distracted`). Prevents micro-flickering. |
| `EMA_A` | `0.12` | `float` | Exponential Moving Average smoothing factor $\alpha$ for student attention scores. Lower values yield smoother, noise-resistant trends. |
| `HIST_LEN` | `90` | `int` | Length of the sliding frame history deque for calculating student trends (`UP`, `DN`, `STDY`) and rolling averages. |
| `DASH_EVERY` | `10` | `int` | Console logging frequency (prints terminal dashboard every $N$ frames). |
| `TW` | `75` | `int` | Width in characters for terminal separators, banners, and summary tables. |

---

## 🛠️ Tuning for Different Environments

### 1. High-Density Classroom (40+ Students)
- Increase `MAX_SEATS = 60`.
- Decrease `SEAT_PROXIMITY_THRES = 0.15` to prevent adjacent students from crossing into neighboring slots.
- Set `YOLO_MODEL_PATH = "yolo11m.pt"` for finer feature resolution on distant desks.

### 2. Low-Resource Edge Device (CPU / Raspberry Pi 5)
- Set `YOLO_MODEL_PATH = "yolo11n.pt"`.
- Increase `FACE_MATCH_INTERVAL = 60`.
- Increase `EMA_A = 0.20`.
