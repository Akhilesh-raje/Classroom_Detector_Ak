# -*- coding: utf-8 -*-
"""
SmartClass AI — Classroom Scanner & Seat Mapper
=================================================
Connects to your 4K external camera, scans the EMPTY classroom,
uses ALL available AI models to detect and name every seat:

  1. YOLO11s   — detects chairs, desks, benches (COCO classes)
  2. Perspective correction — straightens the camera angle
  3. Row/column clustering — groups seats into rows automatically
  4. Ollama/Gemma — generates smart seat names (R1S1, R1S2 or custom)
  5. Saves layout to JSON + annotated image

Output:
  classroom_layout.json   — full seat map with pixel coords + names
  classroom_layout.png    — annotated image showing all detected seats
  classroom_layout_grid.png — clean grid overlay on the classroom photo

Run: python scan_classroom.py
Controls: SPACE=capture | R=retry | S=save | Q=quit
"""
import sys
import os
import io
import cv2
import json
import time
import math
import numpy as np
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "backend"))
sys.path.insert(0, backend_path)

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR   = os.path.dirname(__file__)
LAYOUT_JSON  = os.path.join(OUTPUT_DIR, "classroom_layout.json")
LAYOUT_IMG   = os.path.join(OUTPUT_DIR, "classroom_layout.png")
LAYOUT_GRID  = os.path.join(OUTPUT_DIR, "classroom_layout_grid.png")
DISPLAY_W    = 1280

# YOLO classes to detect as "seats"
# 56=chair, 60=dining table, 59=bed(bench), 57=couch
SEAT_CLASSES  = [56, 60, 59, 57]
DETECT_CONF   = 0.12   # low threshold to catch all seats

# Row clustering — use a LARGER gap so rows are separated correctly
# 0.06 = 6% of frame height between rows (good for tiered classrooms)
ROW_GAP_FRAC  = 0.06

# Resolution fallback chain
RESOLUTIONS = [(3840, 2160), (1920, 1080), (1280, 720)]

# Zoom config
ZOOM_MIN  = 1.0
ZOOM_MAX  = 8.0
ZOOM_STEP = 0.25

# ── Colours ───────────────────────────────────────────────────────────────────
COLORS = {
    "chair":  (50,  205,  50),   # green
    "desk":   (255, 220,   0),   # cyan
    "row1":   (0,   215, 255),   # yellow
    "row2":   (50,  205,  50),   # green
    "row3":   (0,   140, 255),   # orange
    "row4":   (50,   50, 220),   # red
    "row5":   (180,  50, 180),   # purple
    "label":  (255, 255, 255),
    "panel":  (20,   20,  20),
    "grid":   (100, 200, 255),
}

ROW_COLORS = [COLORS[f"row{i}"] for i in range(1, 6)] + [(200, 200, 200)] * 10


# ── Camera ────────────────────────────────────────────────────────────────────

def scan_cameras():
    available = []
    for i in range(6):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
        cap.release()
    return available


def open_best_camera(available):
    """Open highest-index camera (external USB) at best resolution."""
    for idx in reversed(available):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            continue
        for rw, rh in RESOLUTIONS:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  rw)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, rh)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w >= 1280:
                print(f"[CAM] Opened camera {idx} at {w}x{h}")
                return cap, idx, w, h
        # Try any resolution
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w > 0 and h > 0:
            print(f"[CAM] Opened camera {idx} at {w}x{h} (fallback)")
            return cap, idx, w, h
        cap.release()
    return None, -1, 0, 0


# ── Digital Zoom ──────────────────────────────────────────────────────────────

def apply_zoom(frame, zoom: float, pan_x: float = 0.5, pan_y: float = 0.5):
    """
    Apply digital zoom by cropping the center of the frame.
    zoom=1.0 = no zoom, zoom=2.0 = 2x zoom (50% crop), etc.
    pan_x/pan_y: 0.0-1.0, center of zoom region (default 0.5 = center).
    Returns the zoomed frame at original resolution.
    """
    if zoom <= 1.0:
        return frame

    h, w = frame.shape[:2]
    crop_w = int(w / zoom)
    crop_h = int(h / zoom)

    # Pan offset — clamp so crop stays within frame
    cx = int(pan_x * w)
    cy = int(pan_y * h)

    x1 = max(0, min(cx - crop_w // 2, w - crop_w))
    y1 = max(0, min(cy - crop_h // 2, h - crop_h))
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    cropped = frame[y1:y2, x1:x2]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def draw_zoom_indicator(frame, zoom: float, pan_x: float, pan_y: float):
    """Draw a small zoom level indicator and pan position in the corner."""
    H, W = frame.shape[:2]
    # Zoom badge top-right
    badge = f"ZOOM  {zoom:.1f}x"
    col   = (0, 215, 255) if zoom > 1.0 else (100, 100, 100)
    bw    = 110
    bh    = 28
    bx    = W - bw - 10
    by    = 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (bx, by), (bx+bw, by+bh), (20,20,20), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), col, 1)
    cv2.putText(frame, badge, (bx+8, by+19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1, cv2.LINE_AA)

    # Mini pan indicator (small rectangle showing crop position)
    if zoom > 1.0:
        ind_size = 60
        ind_x    = W - ind_size - 10
        ind_y    = by + bh + 6
        # Background
        cv2.rectangle(frame, (ind_x, ind_y), (ind_x+ind_size, ind_y+ind_size), (40,40,40), -1)
        cv2.rectangle(frame, (ind_x, ind_y), (ind_x+ind_size, ind_y+ind_size), (80,80,80), 1)
        # Crop region indicator
        crop_frac = 1.0 / zoom
        cw = int(ind_size * crop_frac)
        ch = int(ind_size * crop_frac)
        cx = ind_x + int(pan_x * (ind_size - cw))
        cy = ind_y + int(pan_y * (ind_size - ch))
        cv2.rectangle(frame, (cx, cy), (cx+cw, cy+ch), col, 1)
        cv2.putText(frame, "PAN", (ind_x+2, ind_y+ind_size+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (100,100,100), 1, cv2.LINE_AA)


# ── YOLO Detection ────────────────────────────────────────────────────────────

def detect_all_seats(frame):
    """
    Hybrid detection for Indian classroom bench-desk units.
    Combines:
      1. YOLO (all furniture classes, very low conf)
      2. Color-based bench detection (dark brown/maroon rectangular surfaces)
      3. Edge/contour-based horizontal surface detection
    Merges all results and deduplicates.
    """
    H, W = frame.shape[:2]
    all_dets = []

    # ── Method 1: YOLO (all classes, very low conf) ───────────────────────────
    try:
        from core.detector import _get_yolo
        model = _get_yolo()
        # Try ALL classes at very low conf to catch anything furniture-like
        results = model.predict(frame, conf=0.08, verbose=False, iou=0.35)
        cls_names = {
            56:"chair", 57:"couch", 58:"potted plant", 59:"bed",
            60:"dining table", 61:"toilet", 62:"tv", 63:"laptop",
        }
        furniture_cls = {56, 57, 59, 60}
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                if cls not in furniture_cls:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                bw, bh = x2-x1, y2-y1
                # Filter: must be wider than tall (bench/desk shape) and reasonable size
                if bw < 30 or bh < 20:
                    continue
                cx, cy = (x1+x2)/2, (y1+y2)/2
                all_dets.append({
                    "bbox": [x1,y1,x2,y2], "cls": cls,
                    "cls_name": cls_names.get(cls, f"cls{cls}"),
                    "confidence": round(conf, 3),
                    "center": [round(cx,1), round(cy,1)],
                    "area": bw*bh, "method": "yolo",
                })
    except Exception as e:
        print(f"[YOLO] Detection error: {e}")

    # ── Method 2: Color-based bench detection ─────────────────────────────────
    # Indian classroom benches are typically dark brown/maroon/dark wood
    color_dets = detect_benches_by_color(frame)
    all_dets.extend(color_dets)

    # ── Method 3: Horizontal edge detection ──────────────────────────────────
    edge_dets = detect_benches_by_edges(frame)
    all_dets.extend(edge_dets)

    # ── Merge and deduplicate ─────────────────────────────────────────────────
    merged = deduplicate_detections(all_dets, iou_thresh=0.35)
    merged.sort(key=lambda d: (d["center"][1], d["center"][0]))
    print(f"[DETECT] YOLO:{sum(1 for d in all_dets if d.get('method')=='yolo')}  "
          f"Color:{sum(1 for d in all_dets if d.get('method')=='color')}  "
          f"Edge:{sum(1 for d in all_dets if d.get('method')=='edge')}  "
          f"→ Merged:{len(merged)}")
    return merged


def detect_benches_by_color(frame):
    """
    Detect dark brown/maroon bench surfaces using HSV color segmentation.
    Tuned for Indian classroom bench-desk units (dark wood color).
    """
    H, W = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Dark brown / maroon / dark wood ranges in HSV
    color_ranges = [
        # Dark brown/maroon (Indian classroom benches)
        (np.array([0,  40,  20]), np.array([20, 255, 120])),
        # Reddish brown
        (np.array([0,  50,  30]), np.array([15, 200, 150])),
        # Dark wood / walnut
        (np.array([10, 30,  15]), np.array([25, 180, 100])),
    ]

    mask = np.zeros((H, W), dtype=np.uint8)
    for lo, hi in color_ranges:
        m = cv2.inRange(hsv, lo, hi)
        mask = cv2.bitwise_or(mask, m)

    # Morphological cleanup — connect nearby regions
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 3))  # wide horizontal
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 10))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_h)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_v)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_RECT, (20,5)))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dets = []
    min_area = (W * H) * 0.005   # at least 0.5% of frame
    max_area = (W * H) * 0.35    # at most 35% of frame

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        # Must be wider than tall (bench shape) — aspect ratio > 1.5
        if bw < bh * 1.5:
            continue
        # Must not be too thin
        if bh < 15:
            continue
        cx, cy = x + bw/2, y + bh/2
        dets.append({
            "bbox": [x, y, x+bw, y+bh], "cls": 59,
            "cls_name": "bench",
            "confidence": round(min(0.75, area / (W*H) * 20), 3),
            "center": [round(cx,1), round(cy,1)],
            "area": bw*bh, "method": "color",
        })
    return dets


def detect_benches_by_edges(frame):
    """
    Detect horizontal bench surfaces using Canny edge detection.
    Finds long horizontal lines that indicate bench/desk tops.
    """
    H, W = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray  = clahe.apply(gray)

    # Canny edges
    edges = cv2.Canny(gray, 30, 100)

    # Dilate horizontally to connect bench top edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 2))
    edges  = cv2.dilate(edges, kernel)

    # Find contours of edge regions
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dets = []
    min_w = W * 0.08   # bench must be at least 8% of frame width
    max_w = W * 0.95

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < min_w or bw > max_w:
            continue
        # Must be a horizontal strip (wide and short)
        if bh > bw * 0.4:
            continue
        if bh < 8:
            continue
        # Expand height slightly to capture the full bench surface
        pad = max(10, bh * 2)
        y1  = max(0, y - pad)
        y2  = min(H, y + bh + pad)
        cx  = x + bw/2
        cy  = (y1 + y2) / 2
        area = bw * (y2 - y1)
        if area < (W * H) * 0.003:
            continue
        dets.append({
            "bbox": [x, y1, x+bw, y2], "cls": 60,
            "cls_name": "desk-edge",
            "confidence": round(min(0.65, bw / W), 3),
            "center": [round(cx,1), round(cy,1)],
            "area": area, "method": "edge",
        })
    return dets


# ── Perspective & Geometry ────────────────────────────────────────────────────

def estimate_vanishing_point(detections, frame_w, frame_h):
    """
    Estimate the vanishing point from detected seat positions.
    Used to understand camera angle and correct perspective labeling.
    """
    if len(detections) < 3:
        return frame_w / 2, 0  # default: center top

    # Use bottom-center points of detections to find convergence
    points = [(d["center"][0], d["bbox"][3]) for d in detections]

    # Simple: find the x-coordinate where rows converge
    # Group by approximate row, find row centers
    rows = cluster_into_rows(detections, frame_h)
    if len(rows) < 2:
        return frame_w / 2, 0

    row_centers = []
    for row in rows:
        xs = [d["center"][0] for d in row]
        ys = [d["center"][1] for d in row]
        row_centers.append((sum(xs)/len(xs), sum(ys)/len(ys)))

    # Vanishing point x = weighted average of row center xs
    vp_x = sum(rc[0] for rc in row_centers) / len(row_centers)
    return vp_x, 0


def cluster_into_rows(detections, frame_h, gap_frac=None):
    """
    Cluster detections into rows by Y-center proximity.
    Uses adaptive gap detection — finds natural breaks in Y distribution.
    gap_frac: fraction of frame height as minimum row gap (default ROW_GAP_FRAC).
    """
    if not detections:
        return []

    gap = (gap_frac or ROW_GAP_FRAC) * frame_h
    sorted_dets = sorted(detections, key=lambda d: d["center"][1])

    # Find natural Y gaps between consecutive detections
    # Use the median bbox height as a reference for "same row" tolerance
    heights = [(d["bbox"][3] - d["bbox"][1]) for d in sorted_dets]
    med_h   = sorted(heights)[len(heights)//2] if heights else 50
    # Two desks are in the same row if their Y centers are within 40% of median height
    row_tol = max(gap, med_h * 0.40)

    rows = [[sorted_dets[0]]]
    for det in sorted_dets[1:]:
        cy      = det["center"][1]
        last_cy = rows[-1][-1]["center"][1]
        if abs(cy - last_cy) <= row_tol:
            rows[-1].append(det)
        else:
            rows.append([det])

    # Sort each row left-to-right
    for row in rows:
        row.sort(key=lambda d: d["center"][0])

    return rows


# ── Seat Naming ───────────────────────────────────────────────────────────────

def generate_seat_names(rows, ollama_available=False):
    """
    Generate seat names for all detected seats.
    Format: R{row}S{seat} e.g. R1S1, R1S2, R2S1
    If Ollama available, use LLM for smarter naming.
    Returns dict: seat_key → display_name
    """
    names = {}
    grid_labels = []

    for r_idx, row in enumerate(rows):
        for s_idx, det in enumerate(row):
            key   = f"det_{r_idx}_{s_idx}"
            label = f"R{r_idx + 1}S{s_idx + 1}"
            names[key]  = label
            det["seat_label"] = label
            det["row"]        = r_idx + 1
            det["col"]        = s_idx + 1
            det["seat_key"]   = key
            grid_labels.append(label)

    if ollama_available:
        try:
            from core.ollama_labeler import ollama_labeler
            mapping = ollama_labeler.label_seats(grid_labels)
            # Update names with Ollama suggestions
            for key, label in zip(names.keys(), grid_labels):
                if label in mapping:
                    names[key] = mapping[label]
            print(f"[OLLAMA] Seat names: {list(names.values())[:6]}...")
        except Exception as e:
            print(f"[OLLAMA] Labeling failed: {e} — using default names")

    return names


# ── Annotation ────────────────────────────────────────────────────────────────

def alpha_rect(img, x1, y1, x2, y2, color, alpha=0.4):
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    sub  = img[y1:y2, x1:x2]
    rect = np.full_like(sub, color)
    cv2.addWeighted(rect, alpha, sub, 1 - alpha, 0, sub)
    img[y1:y2, x1:x2] = sub


def draw_seat(frame, det, row_idx, show_details=True):
    """Draw a single detected seat with label and info."""
    x1, y1, x2, y2 = det["bbox"]
    label  = det.get("seat_label", "?")
    cls_nm = det.get("cls_name", "")
    conf   = det.get("confidence", 0)
    col    = ROW_COLORS[row_idx % len(ROW_COLORS)]

    # Filled semi-transparent box
    alpha_rect(frame, x1, y1, x2, y2, col, alpha=0.18)

    # Border
    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)

    # Corner accents
    L = min(20, (x2-x1)//4, (y2-y1)//4)
    for (cx, cy, dx, dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (cx,cy), (cx+L*dx,cy), col, 3, cv2.LINE_AA)
        cv2.line(frame, (cx,cy), (cx,cy+L*dy), col, 3, cv2.LINE_AA)

    # Seat label badge
    bh = 26
    bw = max(70, len(label) * 11 + 14)
    alpha_rect(frame, x1, y1 - bh - 2, x1 + bw, y1 - 2, col, alpha=0.90)
    cv2.putText(frame, label, (x1 + 6, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

    if show_details:
        # Class + confidence below box
        info = f"{cls_nm} {conf:.0%}"
        cv2.putText(frame, info, (x1 + 4, y2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

    # Center dot
    cx, cy = int(det["center"][0]), int(det["center"][1])
    cv2.circle(frame, (cx, cy), 5, col, -1)
    cv2.circle(frame, (cx, cy), 7, (255,255,255), 1)


def draw_row_lines(frame, rows):
    """Draw lines connecting seats in the same row."""
    for r_idx, row in enumerate(rows):
        if len(row) < 2:
            continue
        col = ROW_COLORS[r_idx % len(ROW_COLORS)]
        centers = [(int(d["center"][0]), int(d["center"][1])) for d in row]
        for i in range(len(centers) - 1):
            cv2.line(frame, centers[i], centers[i+1], col, 1, cv2.LINE_AA)


def draw_hud(frame, detections, rows, status_msg, frame_count):
    """Draw info panel."""
    H, W = frame.shape[:2]
    hw, hh = 340, 130
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8+hw, 8+hh), (20,20,20), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(frame, (8, 8), (8+hw, 8+hh), (80,80,80), 1)

    cv2.putText(frame, "SmartClass AI", (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255,220,0), 2, cv2.LINE_AA)
    cv2.putText(frame, "CLASSROOM SCANNER", (180, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100,200,255), 1, cv2.LINE_AA)

    cv2.putText(frame, f"Detected: {len(detections)} seats  |  {len(rows)} rows",
                (16, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1, cv2.LINE_AA)

    cls_counts = {}
    for d in detections:
        cls_counts[d["cls_name"]] = cls_counts.get(d["cls_name"], 0) + 1
    cls_str = "  ".join(f"{k}:{v}" for k,v in cls_counts.items())
    cv2.putText(frame, cls_str, (16, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180,180,180), 1, cv2.LINE_AA)

    cv2.putText(frame, status_msg, (16, 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100,255,100), 1, cv2.LINE_AA)

    cv2.putText(frame, f"Frame: {frame_count}", (16, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100,100,100), 1, cv2.LINE_AA)

    hint = "SPACE=scan  S=save  R=retry  N=next cam  +=zoom in  -=zoom out  WASD=pan  0=reset zoom  Q=quit"
    cv2.putText(frame, hint, (20, max(20, H-12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160,160,160), 1, cv2.LINE_AA)

def build_grid_image(frame, rows, frame_w, frame_h):
    """Build a clean grid overlay image showing the seating arrangement."""
    grid_img = frame.copy()

    # Draw grid lines between rows
    draw_row_lines(grid_img, rows)

    # Draw all seats
    for r_idx, row in enumerate(rows):
        for det in row:
            draw_seat(grid_img, det, r_idx, show_details=False)

    # Draw row labels on left side
    for r_idx, row in enumerate(rows):
        if not row:
            continue
        col = ROW_COLORS[r_idx % len(ROW_COLORS)]
        y   = int(sum(d["center"][1] for d in row) / len(row))
        cv2.putText(grid_img, f"Row {r_idx+1}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)

    return grid_img


# ── Layout JSON ───────────────────────────────────────────────────────────────

def build_layout_json(rows, frame_w, frame_h, cam_idx, timestamp):
    """Build the full layout JSON structure."""
    seats = []
    for r_idx, row in enumerate(rows):
        for s_idx, det in enumerate(row):
            seats.append({
                "seat_label":  det.get("seat_label", f"R{r_idx+1}S{s_idx+1}"),
                "row":         r_idx + 1,
                "col":         s_idx + 1,
                "bbox":        det["bbox"],
                "center":      det["center"],
                "cls_name":    det["cls_name"],
                "confidence":  det["confidence"],
                "area_px":     det["area"],
                "student_name": None,   # to be filled when students are registered
                "student_id":   None,
            })

    return {
        "version":    "2.0",
        "timestamp":  timestamp,
        "camera_idx": cam_idx,
        "frame_size": [frame_w, frame_h],
        "total_seats": len(seats),
        "total_rows":  len(rows),
        "max_cols":    max((len(r) for r in rows), default=0),
        "seats":       seats,
        "grid_cells":  [s["seat_label"] for s in seats],
        "note": (
            "Seat positions detected from empty classroom scan. "
            "Assign students via /api/students/{id}/assign-seat endpoint "
            "or the Students page in the frontend."
        ),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print("\n" + "=" * 65)
    print("SmartClass AI — Classroom Scanner")
    print("=" * 65)
    print("Point your 4K camera at the EMPTY classroom.")
    print("The AI will detect all seats, group them into rows,")
    print("and generate a named seating arrangement.")
    print("=" * 65)
    print()
    print("Controls:")
    print("  SPACE       = Scan current frame")
    print("  S           = Save layout")
    print("  R           = Retry / clear")
    print("  N           = Next camera")
    print("  0-9         = Select camera by index")
    print("  + / =       = Zoom in")
    print("  - / _       = Zoom out")
    print("  0           = Reset zoom to 1x")
    print("  W/A/S/D     = Pan up/left/down/right when zoomed")
    print("  Q / ESC     = Quit")
    print()

    # ── Check Ollama ──────────────────────────────────────────────────────────
    ollama_ok = False
    try:
        from core.ollama_labeler import ollama_labeler
        ollama_ok = ollama_labeler.is_available()
        print(f"[OLLAMA] {'Available' if ollama_ok else 'Offline — using default R1S1 naming'}")
    except Exception:
        print("[OLLAMA] Not available — using default naming")

    # ── Open camera ───────────────────────────────────────────────────────────
    available = scan_cameras()
    if not available:
        print("ERROR: No camera found.")
        sys.exit(1)

    print(f"[CAM] Available cameras: {available}")
    cap, cam_idx, W, H = open_best_camera(available)

    if cap is None:
        print("ERROR: Could not open any camera.")
        sys.exit(1)

    # Warm up
    print("[CAM] Warming up camera (2 seconds)...")
    for _ in range(60):
        cap.read()

    WIN = "SmartClass AI — Classroom Scanner"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    # Fit window to screen — never larger than 1280 wide
    disp_h = int(DISPLAY_W * H / W) if W > 0 else 720
    disp_h = min(disp_h, 720)   # cap height at 720 so it fits on screen
    disp_w = int(disp_h * W / H) if H > 0 else DISPLAY_W
    cv2.resizeWindow(WIN, disp_w, disp_h)

    # ── State ─────────────────────────────────────────────────────────────────
    detections  = []
    rows        = []
    frame_count = 0
    status_msg  = "Press SPACE to scan the classroom"
    last_frame  = None
    saved       = False

    # Zoom & pan state
    zoom  = 1.0
    pan_x = 0.5   # 0.0=left  1.0=right
    pan_y = 0.5   # 0.0=top   1.0=bottom
    PAN_STEP = 0.05

    print("\n[READY] Press SPACE to scan. Make sure classroom is EMPTY.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame_count += 1
        last_frame = frame.copy()

        # Apply digital zoom
        display = apply_zoom(frame, zoom, pan_x, pan_y)

        # Draw existing detections
        if detections:
            draw_row_lines(display, rows)
            for r_idx, row in enumerate(rows):
                for det in row:
                    draw_seat(display, det, r_idx)

        draw_hud(display, detections, rows, status_msg, frame_count)
        draw_zoom_indicator(display, zoom, pan_x, pan_y)

        cv2.imshow(WIN, display)
        key = cv2.waitKey(1) & 0xFF

        # ── SPACE: scan ───────────────────────────────────────────────────────
        if key == ord(" "):
            status_msg = "Scanning... please wait"
            cv2.imshow(WIN, display)
            cv2.waitKey(1)

            print("\n[SCAN] Running AI detection...")
            t0 = time.time()

            # Collect 5 frames, merge detections
            all_dets = []
            for _ in range(5):
                ret2, f2 = cap.read()
                if ret2:
                    # Apply zoom to the raw frame before detection
                    # so YOLO sees the zoomed region
                    f2_zoomed = apply_zoom(f2, zoom, pan_x, pan_y)
                    dets = detect_all_seats(f2_zoomed)
                    # Scale bbox coords back to original frame if zoomed
                    if zoom > 1.0:
                        dets = scale_bboxes_from_zoom(dets, f2.shape, zoom, pan_x, pan_y)
                    all_dets.extend(dets)
                    last_frame = f2.copy()

            detections = deduplicate_detections(all_dets, iou_thresh=0.4)
            rows       = cluster_into_rows(detections, H)
            generate_seat_names(rows, ollama_available=ollama_ok)

            elapsed    = time.time() - t0
            status_msg = f"Found {len(detections)} seats in {len(rows)} rows ({elapsed:.1f}s) — Press S to save"
            saved      = False

            print(f"[SCAN] {len(detections)} seats, {len(rows)} rows in {elapsed:.1f}s")
            for r_idx, row in enumerate(rows):
                print(f"       Row {r_idx+1}: {[d.get('seat_label','?') for d in row]}")

        # ── S: save ───────────────────────────────────────────────────────────
        elif key == ord("s"):
            if not detections:
                status_msg = "Nothing to save — scan first (SPACE)"
            else:
                ts     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                layout = build_layout_json(rows, W, H, cam_idx, ts)
                with open(LAYOUT_JSON, "w", encoding="utf-8") as f:
                    json.dump(layout, f, indent=2)

                ann = last_frame.copy()
                draw_row_lines(ann, rows)
                for r_idx, row in enumerate(rows):
                    for det in row:
                        draw_seat(ann, det, r_idx, show_details=True)
                draw_hud(ann, detections, rows, "SAVED", frame_count)
                cv2.imwrite(LAYOUT_IMG, ann)

                grid_img = build_grid_image(last_frame.copy(), rows, W, H)
                cv2.imwrite(LAYOUT_GRID, grid_img)

                saved      = True
                status_msg = f"Saved! {len(detections)} seats → classroom_layout.json"
                print(f"\n[SAVE] {LAYOUT_JSON}")
                print_layout_summary(layout)

        # ── R: retry ──────────────────────────────────────────────────────────
        elif key == ord("r"):
            detections = []
            rows       = []
            status_msg = "Cleared. Press SPACE to scan again."
            saved      = False

        # ── N: next camera ────────────────────────────────────────────────────
        elif key == ord("n"):
            cap.release()
            cur = available.index(cam_idx) if cam_idx in available else 0
            nxt = (cur + 1) % len(available)
            cam_idx = available[nxt]
            print(f"[CAM] Switching to camera {cam_idx}...")
            cap, cam_idx, W, H = open_best_camera([cam_idx])
            if cap is None:
                cap, cam_idx, W, H = open_best_camera(available)
            disp_h = min(int(DISPLAY_W * H / W) if W > 0 else 720, 720)
            disp_w = int(disp_h * W / H) if H > 0 else DISPLAY_W
            cv2.resizeWindow(WIN, disp_w, disp_h)
            for _ in range(20):
                cap.read()
            detections = []
            rows       = []
            zoom       = 1.0
            pan_x      = 0.5
            pan_y      = 0.5
            status_msg = f"Camera {cam_idx} — Press SPACE to scan"

        # ── 0-9: select camera ────────────────────────────────────────────────
        elif ord("1") <= key <= ord("9"):
            desired = key - ord("0")
            if desired in available:
                cap.release()
                print(f"[CAM] Switching to camera {desired}...")
                cap, cam_idx, W, H = open_best_camera([desired])
                if cap is None:
                    cap, cam_idx, W, H = open_best_camera(available)
                disp_h = min(int(DISPLAY_W * H / W) if W > 0 else 720, 720)
                disp_w = int(disp_h * W / H) if H > 0 else DISPLAY_W
                cv2.resizeWindow(WIN, disp_w, disp_h)
                for _ in range(20):
                    cap.read()
                detections = []
                rows       = []
                zoom       = 1.0
                pan_x      = 0.5
                pan_y      = 0.5
                status_msg = f"Camera {cam_idx} — Press SPACE to scan"
            else:
                status_msg = f"Camera {desired} not available. Available: {available}"

        # ── + / =: zoom in ────────────────────────────────────────────────────
        elif key in (ord("+"), ord("=")):
            zoom = min(ZOOM_MAX, round(zoom + ZOOM_STEP, 2))
            status_msg = f"Zoom: {zoom:.1f}x"

        # ── - / _: zoom out ───────────────────────────────────────────────────
        elif key in (ord("-"), ord("_")):
            zoom = max(ZOOM_MIN, round(zoom - ZOOM_STEP, 2))
            if zoom == 1.0:
                pan_x, pan_y = 0.5, 0.5
            status_msg = f"Zoom: {zoom:.1f}x"

        # ── 0: reset zoom ─────────────────────────────────────────────────────
        elif key == ord("0"):
            zoom  = 1.0
            pan_x = 0.5
            pan_y = 0.5
            status_msg = "Zoom reset to 1x"

        # ── WASD: pan ─────────────────────────────────────────────────────────
        elif key == ord("w"):
            pan_y = max(0.0, pan_y - PAN_STEP)
        elif key == ord("s") and zoom > 1.0:
            pan_y = min(1.0, pan_y + PAN_STEP)
        elif key == ord("a"):
            pan_x = max(0.0, pan_x - PAN_STEP)
        elif key == ord("d"):
            pan_x = min(1.0, pan_x + PAN_STEP)

        # ── Q / ESC: quit ─────────────────────────────────────────────────────
        elif key in (ord("q"), 27):
            if detections and not saved:
                print("[INFO] Auto-saving before quit...")
                ts     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                layout = build_layout_json(rows, W, H, cam_idx, ts)
                with open(LAYOUT_JSON, "w", encoding="utf-8") as f:
                    json.dump(layout, f, indent=2)
                if last_frame is not None:
                    ann = last_frame.copy()
                    draw_row_lines(ann, rows)
                    for r_idx, row in enumerate(rows):
                        for det in row:
                            draw_seat(ann, det, r_idx)
                    cv2.imwrite(LAYOUT_IMG, ann)
                print(f"[SAVE] Auto-saved to {LAYOUT_JSON}")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[DONE] Classroom scanner closed.")


def scale_bboxes_from_zoom(dets, orig_shape, zoom, pan_x, pan_y):
    """
    When YOLO runs on a zoomed crop, scale bbox coords back to original frame.
    """
    if zoom <= 1.0 or not dets:
        return dets

    oh, ow = orig_shape[:2]
    crop_w = int(ow / zoom)
    crop_h = int(oh / zoom)
    cx     = int(pan_x * ow)
    cy     = int(pan_y * oh)
    x1_off = max(0, min(cx - crop_w // 2, ow - crop_w))
    y1_off = max(0, min(cy - crop_h // 2, oh - crop_h))

    scaled = []
    for d in dets:
        bx1, by1, bx2, by2 = d["bbox"]
        # Scale from zoomed coords back to original
        sx1 = int(x1_off + bx1 / zoom)
        sy1 = int(y1_off + by1 / zoom)
        sx2 = int(x1_off + bx2 / zoom)
        sy2 = int(y1_off + by2 / zoom)
        scx = (sx1 + sx2) / 2
        scy = (sy1 + sy2) / 2
        nd  = dict(d)
        nd["bbox"]   = [sx1, sy1, sx2, sy2]
        nd["center"] = [round(scx, 1), round(scy, 1)]
        nd["area"]   = (sx2 - sx1) * (sy2 - sy1)
        scaled.append(nd)
    return scaled


# ── Deduplication ─────────────────────────────────────────────────────────────

def deduplicate_detections(dets, iou_thresh=0.4):
    """Remove duplicate detections using IoU-based NMS."""
    if not dets:
        return []

    # Sort by confidence descending
    dets = sorted(dets, key=lambda d: d["confidence"], reverse=True)
    kept = []

    for det in dets:
        overlap = False
        for k in kept:
            if iou(det["bbox"], k["bbox"]) > iou_thresh:
                overlap = True
                break
        if not overlap:
            kept.append(det)

    return kept


def iou(b1, b2):
    """Intersection over Union of two bboxes."""
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    if inter == 0:
        return 0.0
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1 + a2 - inter)


# ── Summary ───────────────────────────────────────────────────────────────────

def print_layout_summary(layout):
    print()
    print("=" * 65)
    print("CLASSROOM LAYOUT SUMMARY")
    print("=" * 65)
    print(f"  Total seats : {layout['total_seats']}")
    print(f"  Total rows  : {layout['total_rows']}")
    print(f"  Max cols    : {layout['max_cols']}")
    print(f"  Frame size  : {layout['frame_size'][0]}x{layout['frame_size'][1]}")
    print()
    print(f"  {'Seat':<10} {'Row':<6} {'Col':<6} {'Type':<10} {'Conf':<8} {'Center'}")
    print(f"  {'-'*60}")
    for s in layout["seats"]:
        cx, cy = s["center"]
        print(f"  {s['seat_label']:<10} {s['row']:<6} {s['col']:<6} "
              f"{s['cls_name']:<10} {s['confidence']:<8.0%} ({cx:.0f}, {cy:.0f})")
    print("=" * 65)
    print()
    print("Next steps:")
    print("  1. Start the backend:  cd backend && uvicorn main:app --reload")
    print("  2. Open the frontend:  npm run dev")
    print("  3. Go to Students page → Register students → Assign seats")
    print("  4. Run webcam:         python scratch/webcam/run_webcam.py")
    print("=" * 65)


if __name__ == "__main__":
    run()
