# -*- coding: utf-8 -*-
"""
SmartClass AI – Precision v3 Live Player
=========================================
Plays the test video with FULL real-time AI overlays:
  • Persistent student IDs (top-6 stable tracks)
  • EMA-smoothed attention scores per student
  • Emotion labels with temporal voting
  • Hysteresis activity states (no flickering)
  • Per-student 30-frame avg attention bar
  • Global classroom HUD (avg attention, face count, FPS)

Controls: Q = quit | SPACE = pause | +/- = speed
"""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'unwanted'))

import cv2
import numpy as np
import time

from yolo_face import detect_faces as yolo_detect
from video_processor import VideoProcessor

# ── Colour palette ────────────────────────────────────────────────────────────
WHITE   = (255, 255, 255)
BLACK   = (0, 0, 0)
CYAN    = (255, 255, 0)
GREEN   = (0, 230, 118)
YELLOW  = (0, 215, 255)
ORANGE  = (0, 140, 255)
RED     = (60, 60, 255)
MAGENTA = (200, 50, 200)
GRAY    = (130, 130, 130)
DARK_BG = (25, 25, 25)

EMO_COLORS = {
    "happy": (0, 230, 118), "sad": (255, 140, 50), "angry": (60, 60, 255),
    "surprised": (0, 200, 255), "fearful": (255, 80, 200),
    "disgusted": (0, 180, 180), "neutral": (170, 170, 170),
    "confused": (180, 130, 255), "unknown": (100, 100, 100),
}

ACT_COLORS = {
    "attentive": GREEN, "neutral": GRAY,
    "distracted": ORANGE, "drowsy": RED, "talking": YELLOW,
    "looking_down": ORANGE,
}

# ── Init ──────────────────────────────────────────────────────────────────────
proc = VideoProcessor()

VIDEO_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'test video')
vids = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(('.mp4','.avi','.mov','.mkv'))]
if not vids:
    print("!! No video files found in 'test video/' directory"); sys.exit(1)
VIDEO_PATH = os.path.join(VIDEO_DIR, vids[0])

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"!! Cannot open {VIDEO_PATH}"); sys.exit(1)

fps        = cap.get(cv2.CAP_PROP_FPS) or 25.0
total_fr   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
src_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
src_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
disp_w     = min(1440, src_w)
disp_h     = int(src_h * (disp_w / src_w))
scale      = disp_w / src_w

print(f"SmartClass AI - Precision v3 Live Player")
print(f"Video   : {VIDEO_PATH}")
print(f"Source  : {src_w}x{src_h} @ {fps:.0f}fps  ({total_fr} frames)")
print(f"Display : {disp_w}x{disp_h}")
print(f"Controls: Q=quit  SPACE=pause  +/-=speed")
print("=" * 60)

WIN = "SmartClass AI - Precision v3 Live"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, disp_w, disp_h)

# ── State ─────────────────────────────────────────────────────────────────────
paused      = False
frame       = None
frame_idx   = 0
speed_mult  = 1.0
last_faces  = []
analyze_n   = 3          # analyse every N-th frame
fps_buf     = []         # for rolling FPS calc

# ── Drawing helpers ───────────────────────────────────────────────────────────
def draw_rounded_rect(img, pt1, pt2, color, radius=8, thickness=-1, alpha=0.75):
    """Semi-transparent rounded rectangle overlay."""
    overlay = img.copy()
    x1, y1 = pt1; x2, y2 = pt2
    # Clamp
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    cv2.rectangle(overlay, (x1+radius, y1), (x2-radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1+radius), (x2, y2-radius), color, thickness)
    cv2.circle(overlay, (x1+radius, y1+radius), radius, color, thickness)
    cv2.circle(overlay, (x2-radius, y1+radius), radius, color, thickness)
    cv2.circle(overlay, (x1+radius, y2-radius), radius, color, thickness)
    cv2.circle(overlay, (x2-radius, y2-radius), radius, color, thickness)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

def put_text_bg(img, text, pos, scale_f=0.45, color=WHITE, bg=DARK_BG, thick=1, pad=4):
    """Text with a background box for readability."""
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale_f, thick)
    x, y = pos
    cv2.rectangle(img, (x-pad, y-th-pad), (x+tw+pad, y+bl+pad), bg, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale_f, color, thick, cv2.LINE_AA)

def attn_color(attn):
    if attn >= 80: return GREEN
    if attn >= 60: return YELLOW
    if attn >= 40: return ORANGE
    return RED

# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    t0 = time.perf_counter()

    if not paused:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            proc.track_manager = type(proc.track_manager)(max_students=20)
            proc.student_states.clear()
            continue
        frame_idx += 1

    if frame is None:
        continue

    display = frame.copy()

    # ── Analysis pass ─────────────────────────────────────────────────────
    if not paused and frame_idx % analyze_n == 0:
        dets = yolo_detect(frame, min_conf=0.20, persist=True)
        ids  = [d["id"] for d in dets if d["id"] is not None]
        proc.track_manager.update(ids)   # stats only

        faces = []
        for det in dets:
            tid = det["id"]
            if tid is None:
                continue
            st = proc._get_state(tid)

            # ── Cascading multi-crop: tight → wide → upper-body ──
            mp_res = None
            used_crop = "none"
            crop_candidates = det.get("crops", [])
            if not crop_candidates:
                crop_candidates = [{"name": "tight", "bbox": det["bbox"]}]

            for crop_info in crop_candidates:
                cx1, cy1, cx2, cy2 = crop_info["bbox"]
                crop = frame[max(0,cy1):max(0,cy2), max(0,cx1):max(0,cx2)]
                if crop.size == 0:
                    continue
                mp_res = proc._analyze_crop(crop)
                if mp_res:
                    used_crop = crop_info["name"]
                    break

            x1, y1, x2, y2 = det["bbox"]

            if mp_res:
                attn_s = st.smooth_attention(mp_res["attn_raw"])
                emo    = st.smooth_emotion(mp_res["probs"])
                from video_processor import _classify_activity
                act    = _classify_activity(attn_s, mp_res["blendshapes"], st.prev_activity)
                st.prev_activity = act
                faces.append({
                    "id": tid, "bbox": [x1,y1,x2,y2],
                    "attn": attn_s, "avg30": st.avg_attention_30,
                    "emo": emo, "act": act,
                    "pose": mp_res["pose"], "src": f"MP:{used_crop[0]}",
                    "conf": det["confidence"],
                })
            else:
                last_a = st.attn_ema if st.attn_ema else 50.0
                faces.append({
                    "id": tid, "bbox": [x1,y1,x2,y2],
                    "attn": round(last_a,1), "avg30": st.avg_attention_30,
                    "emo": "neutral", "act": st.prev_activity,
                    "pose": {"yaw":0,"pitch":0,"roll":0}, "src": "KF",
                    "conf": det["confidence"],
                })
        last_faces = faces

    # ── Draw per-student overlays ─────────────────────────────────────────
    for f in last_faces:
        x1, y1, x2, y2 = f["bbox"]
        attn  = f["attn"]
        avg30 = f["avg30"]
        emo   = f["emo"]
        act   = f["act"]
        pose  = f["pose"]
        tid   = f["id"]
        src   = f["src"]

        col = attn_color(attn)
        emo_col = EMO_COLORS.get(emo, GRAY)

        # ── Bounding box ──
        cv2.rectangle(display, (x1, y1), (x2, y2), col, 2)

        # ── Corner accents ──
        corner_len = min(20, (x2-x1)//4, (y2-y1)//4)
        for (cx, cy, dx, dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(display, (cx, cy), (cx + corner_len*dx, cy), col, 3)
            cv2.line(display, (cx, cy), (cx, cy + corner_len*dy), col, 3)

        # ── ID badge (top-left) ──
        badge_w = 55
        draw_rounded_rect(display, (x1, y1-28), (x1+badge_w, y1), col, radius=4, alpha=0.85)
        cv2.putText(display, f"ID:{tid}", (x1+4, y1-9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 2, cv2.LINE_AA)

        # ── Source tag (MP / KF) ──
        src_col = GREEN if src == "MP" else ORANGE
        cv2.putText(display, f"[{src}]", (x1+badge_w+4, y1-9), cv2.FONT_HERSHEY_SIMPLEX, 0.35, src_col, 1, cv2.LINE_AA)

        # ── Data panel (right side of box) ──
        px = x2 + 6
        py = y1 + 2
        panel_h = 100
        panel_w = 140
        draw_rounded_rect(display, (px-2, py-2), (px+panel_w, py+panel_h), DARK_BG, radius=6, alpha=0.8)

        # Attention
        cv2.putText(display, f"Attn: {attn:.0f}%", (px+4, py+16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        # Attention bar
        bar_x = px + 4
        bar_y = py + 22
        bar_w = panel_w - 12
        bar_h = 6
        cv2.rectangle(display, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (50,50,50), -1)
        fill = int(bar_w * min(attn, 100) / 100)
        cv2.rectangle(display, (bar_x, bar_y), (bar_x+fill, bar_y+bar_h), col, -1)

        # Avg 30
        cv2.putText(display, f"Avg30: {avg30:.0f}%", (px+4, py+42), cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1, cv2.LINE_AA)

        # Emotion
        cv2.putText(display, f"Emo: {emo}", (px+4, py+58), cv2.FONT_HERSHEY_SIMPLEX, 0.38, emo_col, 1, cv2.LINE_AA)

        # Activity
        act_col = ACT_COLORS.get(act, GRAY)
        cv2.putText(display, f"Act: {act}", (px+4, py+74), cv2.FONT_HERSHEY_SIMPLEX, 0.38, act_col, 1, cv2.LINE_AA)

        # Pose
        yaw_s = pose.get("yaw", 0)
        pit_s = pose.get("pitch", 0)
        cv2.putText(display, f"Y:{yaw_s:>4.0f} P:{pit_s:>4.0f}", (px+4, py+92), cv2.FONT_HERSHEY_SIMPLEX, 0.32, CYAN, 1, cv2.LINE_AA)

        # ── Bottom attention micro-bar ──
        bw = x2 - x1
        bh = 4
        cv2.rectangle(display, (x1, y2+2), (x2, y2+2+bh), (40,40,40), -1)
        fill_b = int(bw * min(attn, 100) / 100)
        cv2.rectangle(display, (x1, y2+2), (x1+fill_b, y2+2+bh), col, -1)

    # ── Global HUD ────────────────────────────────────────────────────────
    dt = time.perf_counter() - t0
    fps_buf.append(dt)
    if len(fps_buf) > 30: fps_buf.pop(0)
    live_fps = 1.0 / max(0.001, sum(fps_buf) / len(fps_buf))

    n_faces = len(last_faces)
    avg_attn = 0.0
    if last_faces:
        avg_attn = sum(f["attn"] for f in last_faces) / n_faces

    # Top bar
    hud_h = 86
    draw_rounded_rect(display, (8, 8), (420, 8+hud_h), DARK_BG, radius=10, alpha=0.82)

    cv2.putText(display, "SmartClass AI", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, CYAN, 2, cv2.LINE_AA)
    cv2.putText(display, "Precision v3", (190, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREEN, 1, cv2.LINE_AA)

    cv2.putText(display, f"Frame {frame_idx}/{total_fr}", (16, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.40, WHITE, 1, cv2.LINE_AA)
    cv2.putText(display, f"FPS: {live_fps:.0f}", (160, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.40, GREEN, 1, cv2.LINE_AA)
    cv2.putText(display, f"Speed: {speed_mult:.1f}x", (250, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.40, YELLOW, 1, cv2.LINE_AA)

    cv2.putText(display, f"Students: {n_faces}", (16, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.40, WHITE, 1, cv2.LINE_AA)
    avg_col = attn_color(avg_attn)
    cv2.putText(display, f"Class Attention: {avg_attn:.0f}%", (160, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.40, avg_col, 1, cv2.LINE_AA)

    # Global attention bar
    gb_x, gb_y, gb_w, gb_h = 16, 78, 395, 8
    cv2.rectangle(display, (gb_x, gb_y), (gb_x+gb_w, gb_y+gb_h), (50,50,50), -1)
    gfill = int(gb_w * min(avg_attn, 100) / 100)
    cv2.rectangle(display, (gb_x, gb_y), (gb_x+gfill, gb_y+gb_h), avg_col, -1)

    # Pause indicator
    if paused:
        draw_rounded_rect(display, (src_w//2-100, src_h//2-30), (src_w//2+100, src_h//2+30), DARK_BG, radius=12, alpha=0.85)
        cv2.putText(display, "PAUSED", (src_w//2-65, src_h//2+10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, CYAN, 3, cv2.LINE_AA)

    # ── Display ───────────────────────────────────────────────────────────
    cv2.imshow(WIN, display)

    # ── Key handling ──────────────────────────────────────────────────────
    target_delay = (1.0 / fps) / speed_mult
    wait_ms = max(1, int((target_delay - dt) * 1000)) if not paused else 30
    key = cv2.waitKey(wait_ms) & 0xFF

    if key == ord('q') or key == 27:
        break
    elif key == ord(' '):
        paused = not paused
    elif key == ord('+') or key == ord('='):
        speed_mult = min(4.0, speed_mult + 0.25)
    elif key == ord('-') or key == ord('_'):
        speed_mult = max(0.25, speed_mult - 0.25)

cap.release()
cv2.destroyAllWindows()

# ── Session summary ───────────────────────────────────────────────────────────
print()
print("=" * 60)
print("SESSION COMPLETE")
print("=" * 60)
for tid, st in sorted(proc.student_states.items()):
    print(f"  Student ID:{tid:<3} avg attention: {st.avg_attention_30:.1f}%")
print("=" * 60)
