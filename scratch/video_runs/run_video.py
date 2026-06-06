# -*- coding: utf-8 -*-
"""
SmartClass AI — Video File Runner  (Single-Pass Live Preview)
==============================================================
- SINGLE PASS: processes, annotates, previews and writes simultaneously
- Preview window opens immediately — no waiting for full processing
- Rich per-student overlay panels with attention bars
- Global HUD with class stats and progress bar
- Annotated output MP4

Run: python scratch/video_runs/run_video.py
"""
import sys
import os
import io
import time
import cv2
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend"))
sys.path.insert(0, backend_path)

from core.config import YOLO_CONF, PERSIST, SMALL_BBOX_FRAC
from core.detector import detect_faces_and_phones, analyze_face_crop, _is_within_padded_bbox
from core.stabilizer import SeatStabilizer
from core.student_state import StudentState
from core.config import SMALL_BBOX_FRAC
from core.visualizer import draw_student_overlay, draw_global_hud, reset_smooth

# ── Config ────────────────────────────────────────────────────────────────────
FRAME_SKIP   = 3
MAX_SEATS    = 6
VIDEO_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "test video"))
OUTPUT_PATH  = os.path.join(os.path.dirname(__file__), "annotated_output.mp4")
VIDEO_EXTS   = (".mp4", ".avi", ".mov", ".mkv", ".webm")
DISPLAY_W    = 1280

# ── Colour palette (BGR) ──────────────────────────────────────────────────────
C = {
    "green":   (50,  205,  50),
    "yellow":  (0,   215, 255),
    "orange":  (0,   140, 255),
    "red":     (50,   50, 220),
    "cyan":    (255, 220,   0),
    "white":   (255, 255, 255),
    "black":   (0,     0,   0),
    "dark":    (20,   20,  20),
    "panel":   (30,   30,  30),
    "purple":  (180,  50, 180),
}

ACT_COLOR = {
    "studying":   C["green"],
    "attentive":  C["green"],
    "neutral":    C["yellow"],
    "talking":    C["yellow"],
    "fidgeting":  C["orange"],
    "distracted": C["orange"],
    "drowsy":     C["red"],
    "phone":      C["red"],
    "laptop":     C["cyan"],
    "music":      C["purple"],
}

ACT_ICON = {
    "studying":   "STUDY",
    "attentive":  "FOCUS",
    "neutral":    "IDLE",
    "talking":    "TALK",
    "fidgeting":  "FIDGET",
    "distracted": "DISTRACT",
    "drowsy":     "DROWSY",
    "phone":      "PHONE!",
    "laptop":     "LAPTOP",
    "music":      "MUSIC",
}


# ── Drawing helpers ───────────────────────────────────────────────────────────
def alpha_rect(img, x1, y1, x2, y2, color, alpha=0.6):
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    sub  = img[y1:y2, x1:x2]
    rect = np.full_like(sub, color)
    cv2.addWeighted(rect, alpha, sub, 1 - alpha, 0, sub)
    img[y1:y2, x1:x2] = sub


def attn_bar(img, x, y, w, h, value):
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), -1)
    fill = int(w * min(max(value, 0), 100) / 100)
    col  = C["green"] if value >= 75 else C["yellow"] if value >= 55 else C["orange"] if value >= 35 else C["red"]
    if fill > 0:
        cv2.rectangle(img, (x, y), (x + fill, y + h), col, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (100, 100, 100), 1)
    return col


def corner_brackets(img, x1, y1, x2, y2, color, length=18, thickness=2):
    for (cx, cy, dx, dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(img, (cx, cy), (cx + length*dx, cy), color, thickness, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + length*dy), color, thickness, cv2.LINE_AA)


# ── Smoothed display state ────────────────────────────────────────────────────
_smooth = {}

def smooth(sid, key, val, alpha=0.10):
    """EMA smooth — very low alpha = very stable, no flickering."""
    k = (sid, key)
    _smooth[k] = alpha * val + (1 - alpha) * _smooth[k] if k in _smooth else val
    return _smooth[k]


# ── Per-student overlay ───────────────────────────────────────────────────────
def draw_student(frame, face):
    H, W   = frame.shape[:2]
    x1, y1, x2, y2 = face["bbox"]
    sid    = face["id"]
    act    = face["activity"]
    # Stabilise activity label — only switch display label after 4 consistent frames
    prev_act = _smooth.get((sid, "act_label"), act)
    _smooth[(sid, "act_count")] = _smooth.get((sid, "act_count"), 0) + 1 if act == prev_act else 1
    if _smooth.get((sid, "act_count"), 0) >= 4:
        _smooth[(sid, "act_label")] = act
    act = _smooth.get((sid, "act_label"), act)
    attn   = smooth(sid, "attn",  face["attentionScore"],    0.10)
    avg_a  = smooth(sid, "avg",   face["avgAttention30"],    0.06)
    eye    = smooth(sid, "eye",   face["eyeOpenness"],       0.08)
    yaw    = smooth(sid, "yaw",   face["headPose"]["yaw"],   0.10)
    pitch  = smooth(sid, "pitch", face["headPose"]["pitch"], 0.10)
    emo    = face["emotion"]
    phone  = face["phoneDetected"]
    laptop = face["laptopDetected"]
    gc     = face.get("gridCell", "")

    col  = ACT_COLOR.get(act, C["white"])
    icon = ACT_ICON.get(act, act.upper())

    # Bounding box + corner brackets
    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 1)
    corner_brackets(frame, x1, y1, x2, y2, col,
                    length=min(18, (x2-x1)//5, (y2-y1)//5))

    # Top badge
    badge_txt = f"S{sid}" + (f"  {gc}" if gc else "")
    bw = max(60, len(badge_txt) * 9 + 12)
    alpha_rect(frame, x1, y1 - 24, x1 + bw, y1 - 2, col, alpha=0.85)
    cv2.putText(frame, badge_txt, (x1 + 5, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, C["black"], 1, cv2.LINE_AA)

    # Right-side info panel
    pw, ph = 162, 112
    px = min(x2 + 6, W - pw - 4)
    py = min(max(0, y1), H - ph - 4)

    alpha_rect(frame, px, py, px + pw, py + ph, C["panel"], alpha=0.82)
    cv2.rectangle(frame, (px, py), (px + pw, py + ph), (70, 70, 70), 1)

    # Row 1: activity + score
    cv2.putText(frame, icon, (px + 6, py + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{attn:.0f}%", (px + pw - 44, py + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1, cv2.LINE_AA)

    # Row 2: attention bar
    attn_bar(frame, px + 6, py + 22, pw - 12, 7, attn)

    # Row 3: avg attention
    cv2.putText(frame, f"Avg {avg_a:.0f}%", (px + 6, py + 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, C["white"], 1, cv2.LINE_AA)
    attn_bar(frame, px + 6, py + 48, pw - 12, 4, avg_a)

    # Row 4: emotion
    emo_col = C["green"] if emo == "happy" else C["yellow"] if emo == "neutral" else C["orange"]
    cv2.putText(frame, f"Emo: {emo}", (px + 6, py + 66),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, emo_col, 1, cv2.LINE_AA)

    # Row 5: head pose
    cv2.putText(frame, f"Y:{yaw:+.0f}  P:{pitch:+.0f}", (px + 6, py + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (170, 170, 170), 1, cv2.LINE_AA)

    # Row 6: eye openness
    cv2.putText(frame, f"Eye:{eye:.2f}", (px + 6, py + 94),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (170, 170, 170), 1, cv2.LINE_AA)
    attn_bar(frame, px + 56, py + 87, pw - 64, 4, eye * 100)

    # Phone alert
    if phone:
        alpha_rect(frame, x1, y1 - 52, x2, y1 - 26, C["red"], alpha=0.90)
        cv2.putText(frame, "!! PHONE DETECTED !!", (x1 + 6, y1 - 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, C["white"], 2, cv2.LINE_AA)

    # Laptop badge
    if laptop:
        alpha_rect(frame, x1, y2 + 4, x1 + 72, y2 + 20, (30, 60, 60), alpha=0.85)
        cv2.putText(frame, "LAPTOP", (x1 + 4, y2 + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C["cyan"], 1, cv2.LINE_AA)

    # Bottom micro-bar
    bw2  = x2 - x1
    fill = int(bw2 * min(attn, 100) / 100)
    cv2.rectangle(frame, (x1, y2 + 2), (x2, y2 + 6), (50, 50, 50), -1)
    if fill > 0:
        cv2.rectangle(frame, (x1, y2 + 2), (x1 + fill, y2 + 6), col, -1)


# ── Global HUD ────────────────────────────────────────────────────────────────
def draw_hud(frame, faces, frame_idx, total_fr, fps, elapsed):
    H, W = frame.shape[:2]
    n       = len(faces)
    avg_a   = sum(f["attentionScore"] for f in faces) / max(1, n)
    study_n = sum(1 for f in faces if f["activity"] in ("studying", "attentive"))
    dist_n  = sum(1 for f in faces if f["activity"] in ("distracted", "phone", "drowsy"))

    hw, hh = 315, 102
    alpha_rect(frame, 8, 8, 8 + hw, 8 + hh, C["panel"], alpha=0.82)
    cv2.rectangle(frame, (8, 8), (8 + hw, 8 + hh), (80, 80, 80), 1)

    cv2.putText(frame, "SmartClass AI", (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, C["cyan"], 2, cv2.LINE_AA)
    cv2.putText(frame, "LIVE", (168, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, C["green"], 1, cv2.LINE_AA)

    pct = frame_idx / max(1, total_fr) * 100
    cv2.putText(frame, f"Frame {frame_idx}/{total_fr}  {pct:.0f}%", (16, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.37, C["white"], 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS:{fps:.1f}  t={elapsed:.1f}s", (16, 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.37, C["white"], 1, cv2.LINE_AA)
    cv2.putText(frame, f"Students:{n}  Study:{study_n}  Dist:{dist_n}", (16, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.37, C["white"], 1, cv2.LINE_AA)
    cv2.putText(frame, f"Class Attn: {avg_a:.0f}%", (16, 96),
                cv2.FONT_HERSHEY_SIMPLEX, 0.37, C["white"], 1, cv2.LINE_AA)
    attn_bar(frame, 132, 88, hw - 140, 9, avg_a)

    # Bottom progress bar
    fill = int(W * frame_idx / max(1, total_fr))
    cv2.rectangle(frame, (0, H - 6), (W, H), (40, 40, 40), -1)
    if fill > 0:
        cv2.rectangle(frame, (0, H - 6), (fill, H), C["cyan"], -1)


# ── Video discovery ───────────────────────────────────────────────────────────
def find_video(directory):
    if not os.path.isdir(directory):
        return None
    for f in sorted(os.listdir(directory)):
        if f.lower().endswith(VIDEO_EXTS):
            return os.path.join(directory, f)
    return None


# ── Main — single-pass ────────────────────────────────────────────────────────
def run():
    print("=" * 60)
    print("SmartClass AI — Video Runner  (Single-Pass Live Preview)")
    print("=" * 60)

    video_path = find_video(VIDEO_DIR)
    if not video_path:
        print(f"ERROR: No video found in '{VIDEO_DIR}'")
        sys.exit(1)

    print(f"Video  : {os.path.basename(video_path)}")
    print(f"Output : {OUTPUT_PATH}")
    print(f"Skip   : every {FRAME_SKIP} frames")
    print("=" * 60)
    print("Opening preview window — processing starts immediately...")

    cap     = cv2.VideoCapture(video_path)
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_src, (W, H))

    # Open preview window immediately
    WIN = "SmartClass AI — Live Processing (Q to quit)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, DISPLAY_W, int(DISPLAY_W * H / W))

    # State
    stabilizer = SeatStabilizer(max_seats=MAX_SEATS)
    states: dict[int, StudentState] = {}

    def get_state(sid):
        if sid not in states:
            states[sid] = StudentState(sid)
        return states[sid]

    frame_idx  = 0
    processed  = 0
    detections = 0
    last_faces = []   # carry forward for non-processed frames
    t0         = time.time()
    fps_buf    = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        t_frame = time.time()

        if frame_idx % FRAME_SKIP == 0:
            processed += 1

            # ── Detection ─────────────────────────────────────────────────────
            raw_dets, phones, laptops, earphones = detect_faces_and_phones(
                frame, min_conf=YOLO_CONF, persist=PERSIST
            )
            mapped = stabilizer.update_and_map(raw_dets, W, H)
            frame_area = W * H
            faces = []

            for det in mapped:
                sid = det.get("seat_id")
                if sid is None:
                    continue

                phone_det   = any(_is_within_padded_bbox(p["bbox"], det["person_bbox"]) for p in phones)
                laptop_det  = any(_is_within_padded_bbox(l["bbox"], det["person_bbox"]) for l in laptops)

                # MediaPipe — small bbox uses only upper_body crop
                bbox_area = (det["bbox"][2]-det["bbox"][0]) * (det["bbox"][3]-det["bbox"][1])
                small = frame_area > 0 and (bbox_area / frame_area) < SMALL_BBOX_FRAC
                crops = det.get("crops", [{"name":"tight","bbox":det["bbox"]}])
                if small:
                    crops = [c for c in crops if c["name"] == "upper_body"] or []

                mp_res = None
                for crop_info in crops:
                    cx1,cy1,cx2,cy2 = crop_info["bbox"]
                    crop = frame[max(0,cy1):max(0,cy2), max(0,cx1):max(0,cx2)]
                    if crop.size == 0:
                        continue
                    mp_res = analyze_face_crop(crop)
                    if mp_res:
                        break

                if mp_res:
                    hp   = mp_res["head_pose"]
                    eye  = mp_res["eye_openness"]
                    mout = mp_res["mouth_open"]
                    emo  = mp_res["emotion"]
                    bsh  = mp_res["blendshapes"]
                else:
                    hp   = {"yaw":0.0,"pitch":0.0,"roll":0.0}
                    eye  = 1.0; mout = 0.0; emo = "neutral"; bsh = {}

                st  = get_state(sid)
                sig = st.extract_signals(det["bbox"], det["person_bbox"],
                                         phone_detected=phone_det,
                                         eye_openness=eye, mouth_open=mout,
                                         head_pose=hp)
                raw   = st.compute_attn(sig)
                attn  = st.smooth(raw)
                act   = st.classify_act(sig, attn,
                                        laptop_detected=laptop_det,
                                        earphone_detected=False)

                x1,y1,x2,y2 = det["bbox"]
                faces.append({
                    "id":              sid,
                    "bbox":            [x1,y1,x2,y2],
                    "attentionScore":  attn,
                    "avgAttention30":  st.avg,
                    "activity":        act,
                    "emotion":         emo,
                    "headPose":        hp,
                    "eyeOpenness":     eye,
                    "phoneDetected":   phone_det,
                    "laptopDetected":  laptop_det,
                    "gridCell":        "",
                })

            last_faces  = faces
            detections += len(faces)

        # ── Draw overlays on every frame ──────────────────────────────────────
        for face in last_faces:
            draw_student_overlay(frame, face)

        elapsed = time.time() - t0
        fps_buf.append(time.time() - t_frame)
        if len(fps_buf) > 20:
            fps_buf.pop(0)
        live_fps = 1.0 / max(0.001, sum(fps_buf) / len(fps_buf))

        draw_global_hud(frame, last_faces, {
            "mode":         "VIDEO",
            "fps":          live_fps,
            "elapsed":      elapsed,
            "frame_idx":    frame_idx,
            "total_frames": total_fr,
            "hint":         "Q=quit preview",
        })

        out.write(frame)
        cv2.imshow(WIN, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quit by user.")
            break

    cv2.destroyAllWindows()
    cap.release()
    out.release()

    elapsed_total = time.time() - t0
    unique_ids = sorted(states.keys())

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Frames processed   : {processed}")
    print(f"  Total detections   : {detections}")
    print(f"  Unique seat IDs    : {unique_ids}")
    print(f"  Total time         : {elapsed_total:.2f}s")
    print(f"  Annotated video    : {OUTPUT_PATH}")
    if states:
        print()
        print(f"  {'Seat':<6} {'Avg Attn':>9} {'Peak':>7} {'Activity':<28}")
        print("  " + "-" * 50)
        for sid, st in sorted(states.items()):
            rec_act, note = st.reconciled_activity()
            display = f"{rec_act}" + (f" ({note})" if note else "")
            print(f"  {sid:<6} {st.avg:>8.1f}%  {st.peak:>6.1f}%  {display:<28}")
    print("=" * 60)


if __name__ == "__main__":
    run()
