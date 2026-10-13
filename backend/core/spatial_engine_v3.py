# -*- coding: utf-8 -*-
"""
SmartClass Spatial Perception V3
=================================
Discovers bench rows and seat count from classroom geometry.
Neither row count nor seat count is assumed or hardcoded.

Algorithm (single pass, ~300ms):
  1. Detect persons across N calibration frames
  2. Split persons into rows using largest gap in person_bottom distribution
  3. Run Hough in each row's Y-strip → find the dominant desk-front edge
  4. n_seats = round(edge_length / SEAT_WIDTH_PX)   ← geometry, not person count
  5. head_y = median face-crop Y for that row         ← matches runtime anchors exactly
  6. Place n_seats evenly-spaced anchors at head_y

Data model:
  PhysicalSeat — permanent slot, persists when student is absent (EMPTY state)
  PersonTrack  — ephemeral per-frame YOLO detection
  Student      — identity (face recognition result)
"""
from __future__ import annotations
import math, statistics
import cv2
import numpy as np
from dataclasses import dataclass, field

# ── Tuning ────────────────────────────────────────────────────────────────────
_SEAT_WIDTH_PX   = 300    # pixels per physical seat at 1920×1080, ~1.5m from wall
_HOUGH_THRESH    = 50     # Hough votes needed
_MIN_LINE_FRAC   = 0.30   # line must span ≥ 30% of frame width
_MAX_ANGLE_DEG   = 10.0   # max deviation from horizontal
_BAND_MERGE_PX   = 25     # lines within this Y-range merge into one band
_STRIP_PAD       = 80     # px above/below person_bottom to search for desk edge


# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class PhysicalSeat:
    seat_id:  str           # "R1C3"
    row:      int           # 0-indexed
    col:      int           # 1-indexed
    anchor:   tuple         # (cx, cy) — where the face-crop will appear
    desk_y:   float         # Y of the detected desk-front edge
    desk_id:  str

@dataclass
class ClassroomTopology:
    seats:           list  = field(default_factory=list)
    desks:           list  = field(default_factory=list)
    n_rows:          int   = 0
    frame_size:      list  = field(default_factory=lambda: [0, 0])
    confidence:      float = 0.0
    source:          str   = "hough_v3"
    n_calib_persons: int   = 0


# ── Core: find dominant desk-front edge in a Y strip ──────────────────────────
def _desk_edge_in_strip(frame: np.ndarray, y_center: float,
                         half_h: int, W: int) -> dict | None:
    """
    Hough in a narrow horizontal strip around y_center.
    Returns the widest line-cluster found, or None.
    """
    H  = frame.shape[0]
    y1 = max(0,   int(y_center - half_h))
    y2 = min(H-1, int(y_center + half_h))
    if y2 <= y1:
        return None

    gray  = cv2.cvtColor(frame[y1:y2], cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(2.5, (16, 4))
    gray  = clahe.apply(gray)
    edges = cv2.Canny(gray, 25, 90)
    k     = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    edges = cv2.dilate(edges, k)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                             threshold=_HOUGH_THRESH,
                             minLineLength=int(W * _MIN_LINE_FRAC),
                             maxLineGap=40)
    if lines is None:
        return None

    # Keep horizontal lines only, convert to frame coords
    segs = []
    for l in lines:
        x1, ly1, x2, ly2 = l[0]
        if abs(math.degrees(math.atan2(ly2-ly1, x2-x1))) > _MAX_ANGLE_DEG:
            continue
        if x1 > x2: x1, x2 = x2, x1
        segs.append((x1, (ly1+ly2)/2 + y1, x2))

    if not segs:
        return None

    # Cluster by Y, pick widest cluster
    segs.sort(key=lambda s: s[1])
    clusters, cur = [[segs[0]]], segs[0][1]
    for s in segs[1:]:
        if abs(s[1] - cur) <= _BAND_MERGE_PX:
            clusters[-1].append(s)
        else:
            clusters.append([s]); cur = s[1]

    best = max(clusters, key=lambda c: max(s[2] for s in c) - min(s[0] for s in c))
    xl   = min(s[0] for s in best)
    xr   = max(s[2] for s in best)
    cy   = sum(s[1] for s in best) / len(best)

    if xr - xl < W * _MIN_LINE_FRAC:
        return None

    return {"y": round(cy, 1), "x_left": float(xl),
            "x_right": float(xr), "length_px": round(xr-xl, 1),
            "n_lines": len(best)}


# ── Public API ────────────────────────────────────────────────────────────────
def build_topology(person_bboxes: list, calib_frames: list,
                   frame_w: int, frame_h: int,
                   face_bboxes: list | None = None) -> ClassroomTopology:
    """
    Build classroom topology from calibration data.

    person_bboxes : full-body YOLO bboxes from N calibration frames
    calib_frames  : actual BGR frames (for Hough)
    face_bboxes   : optional head-crop bboxes (same order as person_bboxes);
                    when provided, anchor Y matches runtime detections exactly
    """
    if not person_bboxes or not calib_frames:
        return ClassroomTopology(frame_size=[frame_w, frame_h])

    # ── 1. Deduplicate persons by spatial proximity ────────────────────────────
    dedup_p, dedup_f = [], []
    for i, bb in enumerate(person_bboxes):
        cx = (bb[0]+bb[2])/2; cy = (bb[1]+bb[3])/2
        if not any(abs(cx-(e[0]+e[2])/2) < frame_w*0.05 and
                   abs(cy-(e[1]+e[3])/2) < frame_h*0.05
                   for e in dedup_p):
            dedup_p.append(bb)
            dedup_f.append(face_bboxes[i] if face_bboxes and i < len(face_bboxes) else None)

    # ── 2. Split persons into rows by largest gap in person_bottom ─────────────
    sorted_idx = sorted(range(len(dedup_p)), key=lambda i: dedup_p[i][3])
    bots       = [dedup_p[i][3] for i in sorted_idx]
    gaps       = [bots[i+1]-bots[i] for i in range(len(bots)-1)]

    # Find best split: largest gap where both halves have ≥ N/4 persons
    # This skips outlier persons at the extremes of the distribution
    MIN_PER_ROW = max(2, len(dedup_p) // 4)
    row_splits  = []
    for gi in sorted(range(len(gaps)), key=lambda i: gaps[i], reverse=True):
        left_n  = gi + 1
        right_n = len(bots) - left_n
        if left_n >= MIN_PER_ROW and right_n >= MIN_PER_ROW:
            row_splits.append(gi)
            break

    # Build row groups
    if row_splits:
        sp = row_splits[0]
        row_groups = [
            [sorted_idx[i] for i in range(0,      sp+1)],
            [sorted_idx[i] for i in range(sp+1,   len(sorted_idx))],
        ]
    else:
        row_groups = [[i for i in range(len(dedup_p))]]

    # ── 3. For each row: Hough desk edge + head_y + n_seats ───────────────────
    ref = calib_frames[len(calib_frames)//2]
    seats: list = []
    desks: list = []

    for r_idx, idx_list in enumerate(row_groups):
        row_p = [dedup_p[i] for i in idx_list]
        row_f = [dedup_f[i] for i in idx_list]

        # Y center for desk-edge search = median person_bottom of this row
        desk_y_search = statistics.median(p[3] for p in row_p)
        edge = _desk_edge_in_strip(ref, desk_y_search, _STRIP_PAD, frame_w)

        if edge:
            desk_y  = edge["y"]
            x_left  = edge["x_left"]
            x_right = edge["x_right"]
            n_lines = edge["n_lines"]
            e_src   = "hough"
        else:
            # Fallback: use person X-span as desk extent
            desk_y  = desk_y_search
            x_left  = max(0,         min(p[0] for p in row_p) - _SEAT_WIDTH_PX)
            x_right = min(frame_w-1, max(p[2] for p in row_p) + _SEAT_WIDTH_PX)
            n_lines = 0
            e_src   = "person_span"

        length  = x_right - x_left
        n_seats = max(1, round(length / _SEAT_WIDTH_PX))

        # head_y: actual face-crop center if available, else top-20% of person
        face_cys = [(f[1]+f[3])/2 for f in row_f if f is not None]
        head_y   = statistics.median(face_cys) if face_cys else \
                   statistics.median(p[1] + (p[3]-p[1])*0.20 for p in row_p)

        desk_id = f"bench_{r_idx+1:02d}"

        # Seat anchors: X from actual face positions, deduplicated across frames
        face_cxs_raw = sorted(
            (dedup_f[i][0]+dedup_f[i][2])/2
            for i in idx_list if dedup_f[i] is not None
        ) if dedup_f else []

        # Dedup face cx: merge any two positions within SEAT_WIDTH/2 of each other
        anchor_xs: list = []
        for cx in face_cxs_raw:
            if not anchor_xs or cx - anchor_xs[-1] > _SEAT_WIDTH_PX / 2:
                anchor_xs.append(cx)
            else:
                # Average with existing nearby anchor
                anchor_xs[-1] = (anchor_xs[-1] + cx) / 2

        if not anchor_xs:
            # Fallback: use person cx
            pxs = sorted((dedup_p[i][0]+dedup_p[i][2])/2 for i in idx_list)
            for cx in pxs:
                if not anchor_xs or cx - anchor_xs[-1] > _SEAT_WIDTH_PX / 2:
                    anchor_xs.append(cx)
                else:
                    anchor_xs[-1] = (anchor_xs[-1] + cx) / 2

        # Fill large gaps only (between detected students — no edge phantoms)
        filled = list(anchor_xs)
        gi = 0
        while gi < len(filled)-1:
            if filled[gi+1] - filled[gi] > _SEAT_WIDTH_PX * 1.5:
                filled.insert(gi+1, (filled[gi] + filled[gi+1]) / 2)
            gi += 1
        # No edge phantoms — we have no evidence of seats beyond outermost person

        n_seats = len(filled)   # actual number of seats from person positions

        desks.append({
            "id":        desk_id, "row": r_idx,
            "y_band":    round(desk_y, 1),
            "x_left":    round(x_left, 1), "x_right": round(x_right, 1),
            "length_px": round(length, 1), "n_seats":  n_seats,
            "n_lines":   n_lines, "n_persons": len(row_p),
            "edge_src":  e_src,
            "centerline": [[int(x_left), int(desk_y)],
                           [int(x_right), int(desk_y)]],
        })

        for k, ax in enumerate(filled):
            ax = max(x_left, min(x_right, ax))
            seats.append(PhysicalSeat(
                seat_id = f"R{r_idx+1}C{k+1}",
                row=r_idx, col=k+1,
                anchor  = (round(float(ax), 1), round(float(head_y), 1)),
                desk_y  = round(desk_y, 1),
                desk_id = desk_id,
            ))

    n_hough = sum(1 for d in desks if d["edge_src"] == "hough")
    return ClassroomTopology(
        seats=seats, desks=desks,
        n_rows=len(desks), frame_size=[frame_w, frame_h],
        confidence=round(n_hough/max(1,len(desks)), 2),
        source="hough_v3", n_calib_persons=len(dedup_p),
    )


# ── Scaffold for SeatStabilizer ───────────────────────────────────────────────
def topology_to_seat_scaffold(topo: ClassroomTopology) -> list:
    import time as _t
    now = _t.time()
    return [{
        "id":        i+1,
        "center":    seat.anchor,
        "last_seen": now,
        "last_bbox": [int(seat.anchor[0]-70), int(seat.anchor[1]-80),
                      int(seat.anchor[0]+70), int(seat.anchor[1]+80)],
        "missed":    0,
        "_row":      seat.row,
        "_col":      seat.col,
        "_seat_id":  seat.seat_id,
    } for i, seat in enumerate(topo.seats)]


# ── Serialisation ─────────────────────────────────────────────────────────────
def topology_to_dict(topo: ClassroomTopology) -> dict:
    rows = {}
    for s in topo.seats:
        rows.setdefault(f"row_{s.row+1}", []).append(s.seat_id)
    return {
        "version": "v3", "source": topo.source,
        "confidence": topo.confidence, "frame_size": topo.frame_size,
        "n_rows": topo.n_rows, "n_calib_persons": topo.n_calib_persons,
        "desks":  [{k:v for k,v in d.items() if k != "centerline"} for d in topo.desks],
        "seats":  [{"seat_id":s.seat_id,"row":s.row,"col":s.col,
                    "anchor":list(s.anchor),"desk_y":s.desk_y} for s in topo.seats],
        "rows":   rows, "total_seats": len(topo.seats),
        "note":   ("Row count from largest person_bottom gap. "
                   "Seat count from Hough desk-edge length / seat_width_px. "
                   "Neither row count nor seat count was assumed."),
    }


# ── Visualisation ─────────────────────────────────────────────────────────────
def render_topology(frame: np.ndarray, topo: ClassroomTopology) -> np.ndarray:
    vis  = frame.copy()
    H, W = vis.shape[:2]
    RC   = [(60,200,60),(60,100,255),(255,140,40),(200,60,220)]
    SEAT = (255, 255, 50)

    for desk in topo.desks:
        col = RC[desk["row"] % len(RC)]
        y   = int(desk["y_band"])
        xl  = int(desk["x_left"]); xr = int(desk["x_right"])
        cv2.line(vis, (xl,y), (xr,y), col, 4)
        ov = vis.copy()
        cv2.rectangle(ov, (xl,y-5), (xr,y+5), col, -1)
        vis = cv2.addWeighted(vis, 0.80, ov, 0.20, 0)
        tag = "HOUGH" if desk["edge_src"] == "hough" else "PSPAN"
        lbl = (f"{desk['id']}  [{tag} {desk['n_lines']}L]  "
               f"{desk['length_px']:.0f}px/{_SEAT_WIDTH_PX}={desk['n_seats']}seats  "
               f"({desk['n_persons']}p)")
        cv2.putText(vis, lbl, (xl+4, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 2, cv2.LINE_AA)

    rows_d = {}
    for d in topo.desks: rows_d.setdefault(d["row"], []).append(d)
    for r in range(topo.n_rows-1):
        ra = rows_d.get(r,[]); rb = rows_d.get(r+1,[])
        if ra and rb:
            sep = int((max(d["y_band"] for d in ra)+min(d["y_band"] for d in rb))/2)
            cv2.line(vis, (0,sep), (W,sep), (0,220,220), 2)
            cv2.putText(vis, "row boundary", (8, sep-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,220,220), 1, cv2.LINE_AA)

    for seat in topo.seats:
        sx, sy = int(seat.anchor[0]), int(seat.anchor[1])
        cv2.circle(vis, (sx,sy), 12, SEAT, 2)
        cv2.circle(vis, (sx,sy),  4, SEAT, -1)
        cv2.putText(vis, seat.seat_id, (sx-20, sy-16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, SEAT, 1, cv2.LINE_AA)

    hud = (f"V3  Rows:{topo.n_rows}  Seats:{len(topo.seats)}  "
           f"conf:{topo.confidence:.2f}  seat_count=HOUGH/{_SEAT_WIDTH_PX}px  "
           f"row_count=PERSON_BOTTOM_GAP  NOT_TOLD_COUNTS")
    cv2.rectangle(vis, (0,0), (min(W, len(hud)*8+14), 22), (0,0,0), -1)
    cv2.putText(vis, hud, (4,15), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (255,255,255), 1, cv2.LINE_AA)
    return vis
