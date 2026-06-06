"""
SmartClass AI — Core Configuration
"""
import os

# ── YOLO Detection Parameters ────────────────────────────────────────────────
YOLO_MODEL_PATH = "yolo11s.pt"
YOLO_CONF       = 0.18
PERSIST         = True

# ── Dynamic Seat Stabilizer Parameters ────────────────────────────────────────
MAX_SEATS       = 6
SEAT_PROXIMITY_THRES = 0.25  # Percentage of frame diagonal

SMALL_BBOX_FRAC      = 0.02   # Threshold for small-detection fallback (fraction of frame area)
FACE_MATCH_THRESHOLD = 0.75   # Cosine similarity threshold for face matching
FACE_MATCH_INTERVAL  = 30     # Frames between face-match attempts per seat
HYSTERESIS_FRAMES    = 5      # Frames required to confirm activity transition

# ── Temporal Tracking Parameters ──────────────────────────────────────────────
EMA_A           = 0.12   # Attention smoothing alpha (lower = smoother, more stable)
HIST_LEN        = 90     # Attention history frame length (longer = more stable avg)

# ── Dashboard & Display Parameters ────────────────────────────────────────────
DASH_EVERY      = 10     # Print console dashboard every N frames
TW              = 75     # Console width for separators and banners
