# -*- coding: utf-8 -*-
import sys, io
# Force UTF-8 output so Unicode chars work on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

"""
SmartClass AI - Standalone CLI Analysis Engine
Run this directly: python scratch/run_analysis.py
- NO FastAPI / uvicorn needed
- Full verbose logging with colors
- Fixes 1920x1080 detection by resizing frames
- Lower confidence thresholds for better recall
- Saves annotated output video
- Full blendshape dump for debugging
"""
import os, sys, time, math, base64
from io import BytesIO
from datetime import datetime
import cv2
import numpy as np

# ── Color helpers (works without colorama) ─────────────────────
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    C = {
        "cyan":    Fore.CYAN,
        "green":   Fore.GREEN,
        "yellow":  Fore.YELLOW,
        "red":     Fore.RED,
        "magenta": Fore.MAGENTA,
        "blue":    Fore.BLUE,
        "white":   Fore.WHITE,
        "reset":   Style.RESET_ALL,
        "bold":    Style.BRIGHT,
    }
except ImportError:
    C = {k: "" for k in ["cyan","green","yellow","red","magenta","blue","white","reset","bold"]}

def ts():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

def log(msg, level="INFO"):
    color = {"INFO": C["green"], "WARN": C["yellow"], "ERROR": C["red"],
             "FACE": C["cyan"], "DIAG": C["magenta"], "OK": C["green"]+""+C["bold"]}.get(level, "")
    print(f"{C['cyan']}[{ts()}]{C['reset']} {color}{level:<5}{C['reset']}  {msg}")

def separator(char="-", n=72):
    print(C["blue"] + char * n + C["reset"])

# ── MediaPipe Setup ────────────────────────────────────────────
log("Importing MediaPipe...", "INFO")
import os
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    log("MediaPipe imported successfully", "OK")
except Exception as e:
    log(f"FATAL: MediaPipe import failed — {e}", "ERROR")
    sys.exit(1)

# ── Paths ──────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
CLASSROOM_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
BACKEND_DIR  = os.path.join(CLASSROOM_ROOT, "backend")
MODEL_PATH   = os.path.abspath(os.path.join(CLASSROOM_ROOT, "models", "face_landmarker.task"))
VIDEO_DIR    = os.path.abspath(os.path.join(CLASSROOM_ROOT, "test video"))
OUTPUT_PATH  = os.path.join(SCRIPT_DIR, "annotated_output.mp4")

# Auto-pick first available video
video_files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith((".mp4",".avi",".mov",".mkv"))]
if not video_files:
    log(f"No video files found in: {VIDEO_DIR}", "ERROR")
    sys.exit(1)

VIDEO_PATH = os.path.join(VIDEO_DIR, video_files[0])

# ── Config ─────────────────────────────────────────────────────
FRAME_SKIP           = 1    # process every frame for full test
DETECT_WIDTH         = 960  # resize before detection (fixes 1080p issue)
MIN_FACE_CONFIDENCE  = 0.1  # lowered to detect more faces in varied lighting
MAX_FACES            = 10
DUMP_BLENDSHAPES     = True # set False to reduce output noise

# ── Init Detector ──────────────────────────────────────────────
separator()
log(f"Model  : {MODEL_PATH}", "INFO")
log(f"Video  : {VIDEO_PATH}", "INFO")
log(f"Output : {OUTPUT_PATH}", "INFO")
separator()

if not os.path.exists(MODEL_PATH):
    log(f"FATAL: face_landmarker.task not found at {MODEL_PATH}", "ERROR")
    sys.exit(1)

log("Initializing FaceLandmarker...", "INFO")
try:
    base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base_opts,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=MAX_FACES,
        min_face_detection_confidence=MIN_FACE_CONFIDENCE,
        min_face_presence_confidence=MIN_FACE_CONFIDENCE,
        min_tracking_confidence=MIN_FACE_CONFIDENCE,
    )
    detector = mp_vision.FaceLandmarker.create_from_options(opts)
    log(f"FaceLandmarker ready  (min_confidence={MIN_FACE_CONFIDENCE}, max_faces={MAX_FACES})", "OK")
except Exception as e:
    log(f"FATAL: FaceLandmarker init failed — {e}", "ERROR")
    sys.exit(1)

# ── Emotion Mapping ────────────────────────────────────────────
def blendshapes_to_emotions(blendshapes):
    s = {b.category_name: b.score for b in blendshapes}
    raw = {
        "happy":     (s.get("mouthSmileLeft",0)  + s.get("mouthSmileRight",0)) / 2,
        "sad":       (s.get("browInnerUp",0)      + s.get("mouthFrownLeft",0) + s.get("mouthFrownRight",0)) / 3,
        "angry":     (s.get("browDownLeft",0)     + s.get("browDownRight",0)) / 2,
        "surprised": (s.get("eyeWideLeft",0)      + s.get("eyeWideRight",0)) / 2,
        "fearful":   (s.get("browInnerUp",0)      + s.get("eyeWideLeft",0)) / 2,
        "disgusted": (s.get("noseSneerLeft",0)    + s.get("noseSneerRight",0)) / 2,
    }
    total = sum(raw.values())
    if total > 0:
        norm = {k: round(v/max(1.0,total),4) for k,v in raw.items()}
    else:
        norm = {k: 0.0 for k in raw}
    norm["neutral"] = max(0.0, round(1.0 - max(norm.values()), 4))
    return norm

def attention_score(blendshapes):
    s = {b.category_name: b.score for b in blendshapes}
    eye_open  = 1.0 - (s.get("eyeBlinkLeft",0) + s.get("eyeBlinkRight",0)) / 2
    mouth_op  = s.get("jawOpen", 0)
    look_away = (s.get("eyeLookInLeft",0) + s.get("eyeLookOutRight",0) +
                 s.get("eyeLookOutLeft",0) + s.get("eyeLookInRight",0)) / 2
    score = eye_open*0.4 + (1.0-mouth_op*0.5)*0.3 + (1.0-look_away)*0.3
    return round(score * 100, 1)

def classify_activity(attn, blendshapes):
    s = {b.category_name: b.score for b in blendshapes}
    blink = (s.get("eyeBlinkLeft",0) + s.get("eyeBlinkRight",0)) / 2
    if blink > 0.6:  return "drowsy"
    if s.get("jawOpen",0) > 0.4: return "talking"
    if attn < 40:    return "distracted"
    if attn > 70:    return "attentive"
    return "neutral"

EMOTION_COLOR = {
    "happy":"#10b981","sad":"#3b82f6","angry":"#ef4444",
    "surprised":"#f59e0b","fearful":"#8b5cf6","disgusted":"#14b8a6","neutral":"#94a3b8"
}
def hex_to_bgr(h):
    h = h.lstrip("#")
    r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return (b,g,r)

# ── Open Video ─────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    log(f"Cannot open video: {VIDEO_PATH}", "ERROR")
    sys.exit(1)

native_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
native_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
native_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration_s    = total_frames / native_fps

# Compute detection resize scale
detect_scale = DETECT_WIDTH / native_w
detect_h     = int(native_h * detect_scale)

log(f"Native resolution : {native_w}x{native_h}", "INFO")
log(f"Detection resize  : {DETECT_WIDTH}x{detect_h}  (scale={detect_scale:.3f})", "INFO")
log(f"FPS / Frames      : {native_fps:.1f} fps  |  {total_frames} total frames  |  {duration_s:.1f}s", "INFO")
log(f"Frame skip        : every {FRAME_SKIP} frame(s) → ~{total_frames//FRAME_SKIP} analyzed", "INFO")
separator()

# ── Output Video Writer ────────────────────────────────────────
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, native_fps, (native_w, native_h))

# ── Main Processing Loop ───────────────────────────────────────
frame_idx         = 0
processed_count   = 0
total_detections  = 0
frames_with_faces = 0
no_face_streak    = 0
t_start           = time.time()

# Stats accumulators
all_attention_scores = []
all_emotions         = {}
all_activities       = {}

separator("=")
log("STARTING FULL VIDEO ANALYSIS - ALL FRAMES", "OK")
separator("=")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    if frame_idx % FRAME_SKIP != 0:
        out.write(frame)
        continue

    processed_count += 1

    # Resize for detection
    small = cv2.resize(frame, (DETECT_WIDTH, detect_h))
    rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # Detect
    try:
        result = detector.detect(mp_img)
    except Exception as e:
        log(f"Frame {frame_idx:05d}: detect() exception — {e}", "ERROR")
        out.write(frame)
        continue

    n_faces = len(result.face_landmarks) if result.face_landmarks else 0

    if n_faces == 0:
        no_face_streak += 1
        # Only log every 10 consecutive no-face frames to reduce noise
        if no_face_streak == 1 or no_face_streak % 10 == 0:
            elapsed = time.time() - t_start
            pct = processed_count / max(1, total_frames // FRAME_SKIP) * 100
            log(f"Frame {frame_idx:05d} ({pct:.1f}%)  t={frame_idx/native_fps:.2f}s  │  "
                f"{C['yellow']}No faces detected  [streak={no_face_streak}]{C['reset']}", "WARN")
        out.write(frame)
        continue

    no_face_streak = 0
    frames_with_faces += 1
    total_detections  += n_faces

    elapsed = time.time() - t_start
    pct     = processed_count / max(1, total_frames // FRAME_SKIP) * 100
    separator("·")
    log(f"Frame {frame_idx:05d} ({pct:.1f}%)  t={frame_idx/native_fps:.2f}s  │  "
        f"{C['bold']}{C['green']}{n_faces} face(s) detected{C['reset']}", "FACE")

    annotated = frame.copy()

    for fi, (lms, blendshapes) in enumerate(zip(result.face_landmarks, result.face_blendshapes or [])):
        # Scale bbox back to native resolution
        xs = [lm.x * DETECT_WIDTH  / detect_scale for lm in lms]
        ys = [lm.y * detect_h       / detect_scale for lm in lms]
        x1, y1 = max(0, int(min(xs))-20), max(0, int(min(ys))-20)
        x2, y2 = min(native_w, int(max(xs))+20), min(native_h, int(max(ys))+20)

        # Analysis
        emotions   = blendshapes_to_emotions(blendshapes)
        emotion    = max(emotions, key=emotions.get)
        attn       = attention_score(blendshapes)
        activity   = classify_activity(attn, blendshapes)
        eye_open   = round(1.0 - ({b.category_name:b.score for b in blendshapes}.get("eyeBlinkLeft",0) +
                                   {b.category_name:b.score for b in blendshapes}.get("eyeBlinkRight",0))/2, 3)
        jaw_open   = round({b.category_name:b.score for b in blendshapes}.get("jawOpen",0), 3)

        # Accumulate stats
        all_attention_scores.append(attn)
        all_emotions[emotion]   = all_emotions.get(emotion, 0) + 1
        all_activities[activity]= all_activities.get(activity, 0) + 1

        color_bgr = hex_to_bgr(EMOTION_COLOR.get(emotion, "#ffffff"))
        cv2.rectangle(annotated, (x1,y1), (x2,y2), color_bgr, 3)
        lbl = f"Face#{fi+1} | {emotion} | {activity} | Attn:{attn}%"
        cv2.putText(annotated, lbl, (x1, max(25, y1-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2, cv2.LINE_AA)

        # Terminal output
        emotion_bar   = "".join(["█" if i/10 < emotions[emotion] else "░" for i in range(10)])
        attn_bar      = "".join(["█" if i*10 < attn else "░" for i in range(10)])
        attn_color    = C["green"] if attn > 70 else C["yellow"] if attn > 40 else C["red"]

        log(f"  Face #{fi+1}  bbox=({x1},{y1})→({x2},{y2})", "FACE")
        log(f"    Emotion   : {C['bold']}{emotion:<10}{C['reset']}  [{emotion_bar}]  ({emotions[emotion]*100:.1f}%)", "FACE")
        log(f"    Activity  : {C['bold']}{activity:<12}{C['reset']}", "FACE")
        log(f"    Attention : {attn_color}{attn_bar}{C['reset']}  {attn_color}{attn}%{C['reset']}", "FACE")
        log(f"    Eye open  : {eye_open:.3f}   Jaw open: {jaw_open:.3f}", "FACE")

        # Top emotion breakdown
        top_emos = sorted(emotions.items(), key=lambda x: -x[1])[:4]
        emo_str  = "  ".join([f"{k}={v*100:.1f}%" for k,v in top_emos])
        log(f"    Emotions  : {emo_str}", "FACE")

        # Full blendshape dump (top 10 active ones)
        if DUMP_BLENDSHAPES:
            bs_sorted = sorted(blendshapes, key=lambda b: -b.score)[:10]
            bs_str = "  ".join([f"{b.category_name}={b.score:.3f}" for b in bs_sorted])
            log(f"    Blendshapes(top10): {C['magenta']}{bs_str}{C['reset']}", "DIAG")

    # Release resources safely
    try:
        detector.close()
    except Exception as e:
        log(f"Warning: detector close failed — {e}", "WARN")
    cv2.destroyAllWindows()
    cap.release()
    out.release()
    elapsed_total = time.time() - t_start

# ── Final Report ───────────────────────────────────────────────
separator("=")
log("ANALYSIS COMPLETE - FINAL REPORT", "OK")
separator("=")
log(f"Total frames in video  : {total_frames}", "INFO")
log(f"Frames analyzed        : {processed_count}", "INFO")
log(f"Frames WITH faces      : {frames_with_faces}", "INFO")
log(f"Frames WITHOUT faces   : {processed_count - frames_with_faces}", "INFO")
log(f"Total face detections  : {total_detections}", "INFO")
log(f"Elapsed time           : {elapsed_total:.2f}s", "INFO")

if all_attention_scores:
    avg_attn = sum(all_attention_scores) / len(all_attention_scores)
    log(f"Avg attention score    : {avg_attn:.1f}%", "INFO")

if all_emotions:
    top_emotion = max(all_emotions, key=all_emotions.get)
    emo_str = "  ".join([f"{k}={v}" for k,v in sorted(all_emotions.items(), key=lambda x: -x[1])])
    log(f"Dominant emotion       : {C['bold']}{top_emotion}{C['reset']}", "INFO")
    log(f"Emotion counts         : {emo_str}", "INFO")

if all_activities:
    act_str = "  ".join([f"{k}={v}" for k,v in sorted(all_activities.items(), key=lambda x: -x[1])])
    log(f"Activity counts        : {act_str}", "INFO")

separator()
if frames_with_faces == 0:
    log("⚠ DIAGNOSTIC: Zero faces detected in entire video!", "ERROR")
    log("  Possible causes:", "ERROR")
    log("  1. Video has no clear faces (backs turned, too distant, etc.)", "ERROR")
    log("  2. Lighting is very poor", "ERROR")
    log("  3. min_face_detection_confidence might still be too high", "ERROR")
    log(f"     Current: {MIN_FACE_CONFIDENCE} — try lowering to 0.1", "ERROR")
else:
    detection_rate = frames_with_faces / processed_count * 100
    log(f"Face detection rate: {detection_rate:.1f}% of analyzed frames", "OK")
    log(f"Annotated video saved: {OUTPUT_PATH}", "OK")
separator("═")
