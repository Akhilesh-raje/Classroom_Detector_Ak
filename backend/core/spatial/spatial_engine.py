# -*- coding: utf-8 -*-
"""
SmartClass Spatial Perception V2 — Structural Analysis Engine
=============================================================
Infers desk geometry and seat positions from person detections used as
geometric fiducials.  No seat topology is assumed beforehand.

Why persons-as-fiducials works:
  • The desk front panel lies just below each seated person's torso.
  • Grouping persons by Y into rows locates bench rows without knowing
    how many rows exist.
  • The X-span of persons in a row gives the desk width.
  • Evenly-spacing seat slots along that width gives stable seat anchors.

This decouples *seat scaffolding* (done once from desk geometry) from
*per-frame person tracking* (done every frame).  The scaffold is locked
after calibration; seats persist even when a student is absent.

Pipeline:
  1. Detect persons with YOLO (reuse existing model instance)
  2. Aggregate person bbox data across N calibration frames
  3. Cluster person centroids into bench rows using ratio-gap on Y
  4. For each bench row:
       – desk_y    = median of person bbox bottoms in that row
       – x_left    = min person bbox left  (with margin)
       – x_right   = max person bbox right (with margin)
       – n_seats   = round(desk_width / SEAT_WIDTH_PX)
  5. Evenly-space SeatSlot anchors along the desk span
  6. Return ClassroomLayout (immutable scaffold)

ponytail: O(n·frames) calibration — fine for n≤20, frames≤30.
          Upgrade path: fuse with Hough desk-edge detection to refine
          desk_y once persons are temporarily absent.
"""
from __future__ import annotations
import math, statistics
import numpy as np
from dataclasses import dataclass, field


# ── Tuning ────────────────────────────────────────────────────────────────────
_SEAT_WIDTH_PX    = 140    # approx px per seated student at 1920-wide frame
_X_MARGIN_FRAC    = 0.04   # extend desk span by this fraction of frame width
_MIN_ROW_PERSONS  = 2      # a row needs at least this many persons
_GAP_RATIO        = 2.0    # inter-row gap must be >= GAP_RATIO × next-largest gap


# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class DeskGeometry:
    id:         str
    row:        int
    y_band:     float        # Y of the desk-front edge (below seated persons)
    x_left:     float        # leftmost seat anchor X
    x_right:    float        # rightmost seat anchor X
    length_px:  float        # x_right - x_left
    n_seats:    int
    centerline: list         # [[x1,y1],[x2,y2]]
    center:     list         # [cx, cy]
    bbox:       list         # [x1,y1,x2,y2]

@dataclass
class SeatSlot:
    id:       str
    desk_id:  str
    row:      int
    col:      int
    position: list           # [cx, cy] pixel — the anchor the stabiliser seeds
    occupied: bool            = False
    track_id: int | None      = None

@dataclass
class ClassroomLayout:
    desks:       list  = field(default_factory=list)
    seat_slots:  list  = field(default_factory=list)
    n_rows:      int   = 0
    frame_size:  list  = field(default_factory=lambda: [0, 0])
    confidence:  float = 0.0
    source:      str   = "person_fiducial"
    n_persons_used: int = 0


# ── Step 1 helpers ────────────────────────────────────────────────────────────
def _row_cluster(person_bboxes: list, frame_h: int) -> list:
    """
    Cluster persons into bench rows using ratio-gap partitioning on Y-center.
    No row count assumed.  Returns list of row-lists, sorted top→bottom.
    """
    if not person_bboxes:
        return []

    cy_list = sorted(person_bboxes, key=lambda b: (b[1] + b[3]) / 2)
    ys      = [(b[1] + b[3]) / 2 for b in cy_list]
    gaps    = [ys[i+1] - ys[i] for i in range(len(ys)-1)]

    if not gaps:
        return [cy_list]

    sorted_gaps = sorted(gaps, reverse=True)
    largest     = sorted_gaps[0]
    second      = sorted_gaps[1] if len(sorted_gaps) > 1 else 0
    # threshold: gap must be clearly larger than the next-biggest
    threshold   = max(largest / _GAP_RATIO, second * _GAP_RATIO)

    rows = [[cy_list[0]]]
    for i, gap in enumerate(gaps):
        if gap >= threshold and gap == largest:
            rows.append([])
        rows[-1].append(cy_list[i+1])

    return rows


# ── Step 2: build ClassroomLayout from aggregated person data ─────────────────
def build_layout_from_persons(
    person_bboxes: list,
    frame_w: int,
    frame_h: int,
) -> ClassroomLayout:
    """
    Core geometry engine.  person_bboxes is a flat list of [x1,y1,x2,y2]
    representing ALL person detections across N calibration frames,
    deduplicated by spatial proximity.

    Returns a ClassroomLayout with stable desk geometry and seat slots.
    """
    if not person_bboxes:
        return ClassroomLayout(frame_size=[frame_w, frame_h])

    # Deduplicate: keep unique spatial positions (≥8% frame-width apart)
    dedup: list = []
    for bb in person_bboxes:
        cx = (bb[0] + bb[2]) / 2
        cy = (bb[1] + bb[3]) / 2
        if not any(
            abs(cx - (e[0]+e[2])/2) < frame_w * 0.07 and
            abs(cy - (e[1]+e[3])/2) < frame_h * 0.07
            for e in dedup
        ):
            dedup.append(bb)

    rows = _row_cluster(dedup, frame_h)

    desks: list    = []
    seats: list    = []
    seat_n: int    = 1
    margin         = frame_w * _X_MARGIN_FRAC

    for r_idx, row_persons in enumerate(rows):
        if len(row_persons) < _MIN_ROW_PERSONS:
            continue

        # Sort left-to-right
        row_sorted = sorted(row_persons, key=lambda b: (b[0]+b[2])/2)

        # Desk geometry: desk_y = bottom of sitting persons (80% down bbox)
        desk_y = statistics.median(
            b[1] + (b[3] - b[1]) * 0.80 for b in row_sorted
        )
        # Head anchor Y = top quarter of person bbox — this is what the
        # stabiliser's YOLO head-crop detections will match against
        head_y = statistics.median(
            b[1] + (b[3] - b[1]) * 0.25 for b in row_sorted
        )
        # Span between outermost detected persons + small margin
        # NOT wall-to-wall — avoids ghost seats in empty desk area
        x_left  = max(0, min(b[0] for b in row_sorted) - margin)
        x_right = min(frame_w - 1, max(b[2] for b in row_sorted) + margin)
        length  = x_right - x_left
        cx      = (x_left + x_right) / 2
        # Seat count: detected persons + 1 spare for camera-occluded students
        # capped so we never exceed the physically plausible span
        n = min(len(row_sorted) + 1, round(length / _SEAT_WIDTH_PX))

        desk = DeskGeometry(
            id         = f"bench_{r_idx+1:02d}",
            row        = r_idx,
            y_band     = round(desk_y, 1),
            x_left     = round(x_left, 1),
            x_right    = round(x_right, 1),
            length_px  = round(length, 1),
            n_seats    = n,
            centerline = [[int(x_left), int(desk_y)],
                          [int(x_right), int(desk_y)]],
            center     = [round(cx, 1), round(head_y, 1)],
            bbox       = [int(x_left), int(desk_y)-10,
                          int(x_right), int(desk_y)+10],
        )
        desks.append(desk)

        # Seat slots positioned at head_y — matches where YOLO head-crop
        # centers will appear, so stabiliser proximity matching is accurate
        seat_y = head_y
        p1 = np.array([x_left,  seat_y])
        p2 = np.array([x_right, seat_y])
        for k in range(n):
            t   = (k + 0.5) / n
            pos = p1 + t * (p2 - p1)
            seats.append(SeatSlot(
                id       = f"seat_{seat_n:02d}",
                desk_id  = desk.id,
                row      = r_idx,
                col      = k + 1,
                position = [round(float(pos[0]), 1), round(float(pos[1]), 1)],
            ))
            seat_n += 1

    confidence = min(1.0, len(desks) / max(1, len(rows)))

    return ClassroomLayout(
        desks          = desks,
        seat_slots     = seats,
        n_rows         = len(desks),
        frame_size     = [frame_w, frame_h],
        confidence     = round(confidence, 2),
        source         = "person_fiducial",
        n_persons_used = len(dedup),
    )


# ── Calibration helper: aggregate persons across frames ───────────────────────
def calibrate_from_video_frames(
    frames: list,
    detect_fn,           # callable: frame → list[{bbox, confidence, id}]
    frame_w: int,
    frame_h: int,
) -> ClassroomLayout:
    """
    Run detect_fn on each frame, pool all person bboxes, then build layout.
    detect_fn should return the same format as detector.track_persons().

    Usage:
        layout = calibrate_from_video_frames(
            first_30_frames, detector.track_persons, W, H
        )
    """
    all_bboxes: list = []
    for frame in frames:
        dets = detect_fn(frame)
        for d in dets:
            all_bboxes.append(d["bbox"] if isinstance(d, dict) else d)

    return build_layout_from_persons(all_bboxes, frame_w, frame_h)


# ── Seat scaffold for the stabiliser ─────────────────────────────────────────
def layout_to_seat_scaffold(layout: ClassroomLayout) -> list:
    """
    Convert ClassroomLayout seat slots to the dict format expected by
    SeatStabilizer.seats so it can be pre-seeded and locked immediately.

    Returns list of seat dicts:
        {"id": int, "center": (cx, cy), "last_seen": 0, "last_bbox": [...], "missed": 0}
    """
    import time as _time
    scaffold = []
    now = _time.time()
    for i, slot in enumerate(layout.seat_slots):
        cx, cy = slot.position
        # Approximate a head-crop-sized bbox around the seat anchor
        # (matches what YOLO head-crop detections produce)
        half_w, half_h = 70, 80
        scaffold.append({
            "id":        i + 1,
            "center":    (cx, cy),
            "last_seen": now,
            "last_bbox": [int(cx - half_w), int(cy - half_h),
                          int(cx + half_w), int(cy + half_h)],
            "missed":    0,
            # Extra metadata not used by stabiliser but useful for debugging
            "_desk_id":  slot.desk_id,
            "_row":      slot.row,
            "_col":      slot.col,
        })
    return scaffold


# ── Serialisation ─────────────────────────────────────────────────────────────
def layout_to_dict(layout: ClassroomLayout) -> dict:
    return {
        "source":       layout.source,
        "confidence":   layout.confidence,
        "frame_size":   layout.frame_size,
        "n_rows":       layout.n_rows,
        "n_persons_used": layout.n_persons_used,
        "desks": [
            {"id": d.id, "row": d.row, "y_band": d.y_band,
             "x_left": d.x_left, "x_right": d.x_right,
             "length_px": d.length_px, "n_seats": d.n_seats,
             "centerline": d.centerline, "center": d.center}
            for d in layout.desks
        ],
        "seats": [
            {"id": s.id, "desk_id": s.desk_id, "row": s.row,
             "col": s.col, "position": s.position}
            for s in layout.seat_slots
        ],
        "total_seats": len(layout.seat_slots),
        "note": (
            "Seat geometry derived from person-fiducial desk inference — "
            "persons locate the desk, desk defines the seats. "
            "No seat count or row count was assumed."
        ),
    }


# ── Visualisation ─────────────────────────────────────────────────────────────
def render_layout(frame, layout: ClassroomLayout) -> "np.ndarray":
    import cv2
    vis  = frame.copy()
    H, W = vis.shape[:2]
    ROW_COLORS = [(60,200,60),(60,100,255),(255,140,40),(200,60,220),(0,200,200)]
    SEAT_CLR   = (255, 255, 50)

    for desk in layout.desks:
        col = ROW_COLORS[desk.row % len(ROW_COLORS)]
        p1  = tuple(desk.centerline[0])
        p2  = tuple(desk.centerline[1])
        cv2.line(vis, p1, p2, col, 4)
        ov = vis.copy()
        y  = int(desk.y_band)
        cv2.rectangle(ov, (int(desk.x_left), y-8),
                      (int(desk.x_right), y+8), col, -1)
        vis = cv2.addWeighted(vis, 0.78, ov, 0.22, 0)
        lbl = f"{desk.id}  n_seats={desk.n_seats}  len={desk.length_px:.0f}px"
        cv2.putText(vis, lbl, (int(desk.x_left)+4, y-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 2, cv2.LINE_AA)

    # Row separator between consecutive bench rows
    if layout.desks:
        rows_idx: dict = {}
        for d in layout.desks:
            rows_idx.setdefault(d.row, []).append(d)
        for r_idx in range(layout.n_rows - 1):
            ra = rows_idx.get(r_idx, [])
            rb = rows_idx.get(r_idx + 1, [])
            if ra and rb:
                bot = max(d.bbox[3] for d in ra)
                top = min(d.bbox[1] for d in rb)
                sep = (bot + top) // 2
                cv2.line(vis, (0, sep), (W, sep), (0, 220, 220), 2)

    for seat in layout.seat_slots:
        sx, sy = int(seat.position[0]), int(seat.position[1])
        col    = ROW_COLORS[seat.row % len(ROW_COLORS)]
        cv2.circle(vis, (sx, sy), 13, SEAT_CLR, 2)
        cv2.circle(vis, (sx, sy),  4, SEAT_CLR, -1)
        cv2.putText(vis, f"R{seat.row+1}C{seat.col}", (sx-22, sy-17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, SEAT_CLR, 1, cv2.LINE_AA)

    hud = (f"Benches:{len(layout.desks)}  Seats:{len(layout.seat_slots)}  "
           f"conf:{layout.confidence:.2f}  [{layout.source}]  "
           f"persons_used:{layout.n_persons_used}  NO STUDENT Y FOR TOPOLOGY")
    cv2.rectangle(vis, (0,0), (min(W, len(hud)*8+14), 22), (0,0,0), -1)
    cv2.putText(vis, hud, (4,15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1, cv2.LINE_AA)
    return vis
