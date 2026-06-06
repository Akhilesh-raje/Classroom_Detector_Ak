"""
SmartClass AI — Shared Visualizer
Shared rendering engine used by both run_webcam.py and run_video.py.
Draws per-student overlays, global HUD, grid, and attention bars.
"""
from __future__ import annotations
import cv2
import numpy as np

# ── Colour palette (BGR) ──────────────────────────────────────────────────────
C = {
    "green":   (50,  205,  50),
    "yellow":  (0,   215, 255),
    "orange":  (0,   140, 255),
    "red":     (50,   50, 220),
    "cyan":    (255, 220,   0),
    "white":   (255, 255, 255),
    "black":   (0,     0,   0),
    "panel":   (30,   30,  30),
    "purple":  (180,  50, 180),
    "grid":    (160, 160, 160),
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


# ── Drawing primitives ────────────────────────────────────────────────────────

def alpha_rect(img, x1, y1, x2, y2, color, alpha=0.6):
    """Semi-transparent filled rectangle."""
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    sub  = img[y1:y2, x1:x2]
    rect = np.full_like(sub, color)
    cv2.addWeighted(rect, alpha, sub, 1 - alpha, 0, sub)
    img[y1:y2, x1:x2] = sub


def attn_bar(img, x, y, w, h, value):
    """Horizontal attention progress bar with color coding."""
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), -1)
    fill = int(w * min(max(value, 0), 100) / 100)
    col  = (C["green"] if value >= 75 else C["yellow"] if value >= 55
            else C["orange"] if value >= 35 else C["red"])
    if fill > 0:
        cv2.rectangle(img, (x, y), (x + fill, y + h), col, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (100, 100, 100), 1)
    return col


def corner_brackets(img, x1, y1, x2, y2, color, length=18, thickness=2):
    """Corner bracket accents on a bounding box."""
    for (cx, cy, dx, dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(img, (cx, cy), (cx + length*dx, cy), color, thickness, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + length*dy), color, thickness, cv2.LINE_AA)


# ── Smoothed display state ────────────────────────────────────────────────────
_smooth_state: dict = {}

def smooth(sid, key, val, alpha=0.10):
    """EMA smooth a display value — low alpha = very stable, no flickering."""
    k = (sid, key)
    _smooth_state[k] = alpha * val + (1 - alpha) * _smooth_state[k] if k in _smooth_state else val
    return _smooth_state[k]

def stable_label(sid, key, new_val, hold=5):
    """
    Only update a displayed label after hold consecutive identical values.
    Prevents activity label flickering in the overlay.
    """
    cnt_k  = (sid, key + "_cnt")
    lbl_k  = (sid, key + "_lbl")
    cand_k = (sid, key + "_cand")

    current_lbl  = _smooth_state.get(lbl_k)
    current_cand = _smooth_state.get(cand_k)

    if current_lbl is None:
        _smooth_state[lbl_k]  = new_val
        _smooth_state[cand_k] = new_val
        _smooth_state[cnt_k]  = hold
        return new_val

    if new_val == current_cand:
        _smooth_state[cnt_k] = _smooth_state.get(cnt_k, 0) + 1
    else:
        _smooth_state[cand_k] = new_val
        _smooth_state[cnt_k]  = 1

    if _smooth_state.get(cnt_k, 0) >= hold:
        _smooth_state[lbl_k] = new_val

    return _smooth_state[lbl_k]

def reset_smooth():
    """Clear all smoothing state (call on new session)."""
    _smooth_state.clear()


# ── Grid overlay ──────────────────────────────────────────────────────────────

def draw_grid_overlay(frame, spatial_grid, student_name_map: dict | None = None):
    """Draw semi-transparent grid cells with labels on frame."""
    if spatial_grid is None:
        return
    overlay = frame.copy()
    for cell in spatial_grid.cells:
        x1, y1, x2, y2 = cell.bbox
        H, W = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (180, 180, 180), -1)
    cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)

    for cell in spatial_grid.cells:
        x1, y1, x2, y2 = cell.bbox
        H, W = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), C["grid"], 1)
        label = cell.label
        if student_name_map and cell.label in student_name_map:
            label = student_name_map[cell.label]
        cv2.putText(frame, label, (x1 + 4, min(H - 4, y1 + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, C["white"], 1, cv2.LINE_AA)


def draw_grid_lines(frame, spatial_grid, student_name_map: dict | None = None):
    """Alias for draw_grid_overlay — used by run_webcam.py."""
    draw_grid_overlay(frame, spatial_grid, student_name_map)


# ── Per-student overlay ───────────────────────────────────────────────────────

def draw_student_overlay(frame, face):
    """
    Draw full per-student info panel on frame.
    face dict must have: id, bbox, attentionScore, avgAttention30, activity,
    emotion, headPose, eyeOpenness, phoneDetected, laptopDetected, gridCell.
    """
    H, W   = frame.shape[:2]
    x1, y1, x2, y2 = face["bbox"]
    sid    = face.get("id", "?")

    act    = stable_label(sid, "act",
                          face.get("reconciledActivity", face.get("activity", "neutral")),
                          hold=5)
    emo    = stable_label(sid, "emo",   face.get("emotion", "neutral"),  hold=6)
    attn   = smooth(sid, "attn",  face.get("attentionScore", 0),         0.10)
    avg_a  = smooth(sid, "avg",   face.get("avgAttention30", attn),      0.06)
    eye    = smooth(sid, "eye",   face.get("eyeOpenness", 1.0),          0.08)
    yaw    = smooth(sid, "yaw",   face.get("headPose", {}).get("yaw", 0),   0.10)
    pitch  = smooth(sid, "pitch", face.get("headPose", {}).get("pitch", 0), 0.10)
    phone  = face.get("phoneDetected", False)
    laptop = face.get("laptopDetected", False)
    gc     = face.get("gridCell", "")
    note   = face.get("activityNote", "")
    factors = face.get("attentionFactors", {})

    col  = ACT_COLOR.get(act, C["white"])
    icon = ACT_ICON.get(act, act.upper())

    # Bounding box + corner brackets
    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 1)
    corner_brackets(frame, x1, y1, x2, y2, col,
                    length=min(18, max(4, (x2-x1)//5), max(4, (y2-y1)//5)))

    # Top badge
    badge = f"S{sid}" + (f"  {gc}" if gc else "")
    bw = max(60, len(badge) * 9 + 12)
    alpha_rect(frame, x1, max(0, y1 - 24), x1 + bw, max(0, y1 - 2), col, alpha=0.85)
    cv2.putText(frame, badge, (x1 + 5, max(12, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, C["black"], 1, cv2.LINE_AA)

    # Right-side info panel
    pw, ph = 168, 130
    px = min(x2 + 6, W - pw - 4)
    py = min(max(0, y1), H - ph - 4)

    alpha_rect(frame, px, py, px + pw, py + ph, C["panel"], alpha=0.82)
    cv2.rectangle(frame, (px, py), (px + pw, py + ph), (70, 70, 70), 1)

    cv2.putText(frame, icon, (px + 6, py + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{attn:.0f}%", (px + pw - 44, py + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1, cv2.LINE_AA)
    attn_bar(frame, px + 6, py + 22, pw - 12, 7, attn)

    cv2.putText(frame, f"Avg {avg_a:.0f}%", (px + 6, py + 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, C["white"], 1, cv2.LINE_AA)
    attn_bar(frame, px + 6, py + 48, pw - 12, 4, avg_a)

    emo_col = (C["green"] if emo == "happy" else
               C["yellow"] if emo == "neutral" else C["orange"])
    cv2.putText(frame, f"Emo: {emo}", (px + 6, py + 66),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, emo_col, 1, cv2.LINE_AA)

    cv2.putText(frame, f"Y:{yaw:+.0f}  P:{pitch:+.0f}", (px + 6, py + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (170, 170, 170), 1, cv2.LINE_AA)

    cv2.putText(frame, f"Eye:{eye:.2f}", (px + 6, py + 94),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (170, 170, 170), 1, cv2.LINE_AA)
    attn_bar(frame, px + 56, py + 87, pw - 64, 4, eye * 100)

    if factors:
        hp_f  = factors.get("head_pose", 1.0)
        mot_f = factors.get("motion_stability", 1.0)
        cv2.putText(frame, f"HP:{hp_f:.2f} Mot:{mot_f:.2f}", (px + 6, py + 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (120, 120, 120), 1, cv2.LINE_AA)

    if note:
        cv2.putText(frame, note, (px + 6, py + 122),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.26, (100, 180, 100), 1, cv2.LINE_AA)

    if phone:
        alpha_rect(frame, x1, max(0, y1 - 52), x2, max(0, y1 - 26), C["red"], alpha=0.90)
        cv2.putText(frame, "!! PHONE DETECTED !!", (x1 + 6, max(12, y1 - 32)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, C["white"], 2, cv2.LINE_AA)

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

def draw_global_hud(frame, faces, info: dict):
    """
    Draw global HUD panel top-left.
    info dict keys: mode, fps, elapsed, frame_idx, total_frames, hint, grid_label
    """
    H, W = frame.shape[:2]
    n       = len(faces)
    avg_a   = sum(f.get("attentionScore", 0) for f in faces) / max(1, n)
    study_n = sum(1 for f in faces if f.get("reconciledActivity", f.get("activity","")) in ("studying","attentive"))
    dist_n  = sum(1 for f in faces if f.get("reconciledActivity", f.get("activity","")) in ("distracted","phone","drowsy"))

    hw, hh = 330, 110
    alpha_rect(frame, 8, 8, 8 + hw, 8 + hh, C["panel"], alpha=0.82)
    cv2.rectangle(frame, (8, 8), (8 + hw, 8 + hh), (80, 80, 80), 1)

    mode_col = C["green"] if info.get("mode","").startswith("LIVE") else C["cyan"]
    cv2.putText(frame, "SmartClass AI", (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, C["cyan"], 2, cv2.LINE_AA)
    cv2.putText(frame, info.get("mode", ""), (172, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, mode_col, 1, cv2.LINE_AA)

    fps_val = info.get("fps", 0)
    elapsed = info.get("elapsed", 0)
    cv2.putText(frame, f"FPS:{fps_val:.1f}  t={elapsed:.0f}s", (16, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.37, C["white"], 1, cv2.LINE_AA)
    cv2.putText(frame, f"Students:{n}  Study:{study_n}  Dist:{dist_n}", (16, 66),
                cv2.FONT_HERSHEY_SIMPLEX, 0.37, C["white"], 1, cv2.LINE_AA)
    cv2.putText(frame, f"Class Attn: {avg_a:.0f}%", (16, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.37, C["white"], 1, cv2.LINE_AA)
    attn_bar(frame, 138, 74, hw - 146, 9, avg_a)

    grid_lbl = info.get("grid_label", "")
    if grid_lbl:
        cv2.putText(frame, f"Grid: {grid_lbl}", (16, 98),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 100, 100), 1, cv2.LINE_AA)

    hint = info.get("hint", "Q=quit")
    cv2.putText(frame, hint, (20, max(20, H - 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)

    # Bottom progress bar (video mode)
    if "frame_idx" in info and "total_frames" in info:
        fill = int(W * info["frame_idx"] / max(1, info["total_frames"]))
        cv2.rectangle(frame, (0, H - 6), (W, H), (40, 40, 40), -1)
        if fill > 0:
            cv2.rectangle(frame, (0, H - 6), (fill, H), C["cyan"], -1)
