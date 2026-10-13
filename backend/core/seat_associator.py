# -*- coding: utf-8 -*-
"""
SmartClass AI — Seat Associator (Truth 3)
=========================================
Implements the decoupled association pipeline:
  Track Manager → Row Classifier (with hysteresis) → Seat Associator

Enforces:
  1. Separation of physical seats, person tracks, and seat occupancy.
  2. Row stability: A person cannot jump rows unless consistently in a new band for 5+ frames.
  3. No double-booking: Strictly one track per seat.
  4. Seat hysteresis: Preference for maintaining previous seat assignment.
  5. UNASSIGNED is a valid state (no force-assignment).
  6. Ghost-hold: Missing tracks held for up to GHOST_HOLD_FRAMES before seat is empty.
"""
from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
import numpy as np

from core.topology import PhysicalSeat, ClassroomTopology, PersonTrack

GHOST_HOLD_FRAMES = 8
ROW_HYSTERESIS_FRAMES = 5
SEAT_MAX_DISTANCE_PX = 380.0
HYSTERESIS_BONUS_PX = 45.0


@dataclass
class SeatAssignment:
    """
    Truth 3: Seat Occupancy.
    The binding between a track and a physical seat.
    """
    track_id: Optional[int]
    seat_label: str                    # "R1C3" or "UNASSIGNED"
    cost: float                        # matching distance in pixels
    row_match: bool                    # True if track's Y is in the seat's row band
    status: str                        # "OK" | "UNASSIGNED" | "GHOST" | "CONFLICT"
    prev_seat: Optional[str] = None
    row: Optional[int] = None
    col: Optional[int] = None
    bbox: List[int] = field(default_factory=list)        # Face or seat bbox
    person_bbox: List[int] = field(default_factory=list) # Full body bbox
    confidence: float = 1.0                              # Detection confidence
    seat_confidence: float = 1.0                         # V2.2 Seat spatial assignment confidence [0, 1]
    match_type: str = "EXACT_REGION"                     # "EXACT_REGION" | "OVERLAP_REGION" | "NEAREST_FALLBACK" | "RESCUE" | "GHOST" | "UNASSIGNED"
    ghost: bool = False

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "seat_label": self.seat_label,
            "cost": round(self.cost, 2),
            "row_match": self.row_match,
            "status": self.status,
            "prev_seat": self.prev_seat,
            "row": self.row,
            "col": self.col,
            "bbox": self.bbox,
            "person_bbox": self.person_bbox,
            "confidence": self.confidence,
            "seat_confidence": round(self.seat_confidence, 2),
            "match_type": self.match_type,
            "ghost": self.ghost,
        }


class RowClassifier:
    """
    Assigns each person track to a row index using topology row Y-bands.
    Applies temporal hysteresis: a track cannot switch rows unless it stays
    in the new band for ROW_HYSTERESIS_FRAMES consecutive frames.
    """

    def __init__(self, hysteresis_frames: int = ROW_HYSTERESIS_FRAMES):
        self.hysteresis_frames = hysteresis_frames
        # track_id -> current confirmed row
        self._confirmed_row: Dict[int, int] = {}
        # track_id -> list of recent row candidates
        self._row_history: Dict[int, List[int]] = {}
        self._last_head_pos: Dict[int, Tuple[float, float]] = {}

    def classify(self, track: PersonTrack, topology: ClassroomTopology) -> int:
        tid = track.track_id
        hx, hy = track.head_point
        if tid in self._last_head_pos:
            prev_hx, prev_hy = self._last_head_pos[tid]
            if math.hypot(hx - prev_hx, hy - prev_hy) > 130.0:
                # Track ID jumped across room: YOLO track swap detected!
                self._row_history[tid] = []
                self._confirmed_row.pop(tid, None)
        self._last_head_pos[tid] = (hx, hy)

        bands = topology.row_y_bands
        n_rows = len(bands)
        if n_rows <= 1:
            return 0

        # Feature for row assignment: blend head_y and bot_y with perspective slant compensation
        hy = track.head_point[1]
        by = track.bottom_point[1]
        hx = track.head_point[0]
        y_feat = 0.20 * hy + 0.80 * by
        theta = getattr(topology, "row_slant_theta", 0.0)
        if abs(theta) > 1e-4:
            y_query = y_feat * math.cos(theta) - hx * math.sin(theta)
        else:
            y_query = y_feat

        # In topology, bands are ordered:
        # Row 0 (Front): larger Y
        # Row N-1 (Back): smaller Y
        # Find which band matches
        candidate_row = 0
        min_dist_to_band = float("inf")

        # First check inside band
        found = False
        for r_idx, (y_min, y_max) in enumerate(bands):
            if y_min <= y_query <= y_max:
                candidate_row = r_idx
                found = True
                break

        if not found:
            # Fallback to nearest band midpoint
            for r_idx, (y_min, y_max) in enumerate(bands):
                mid = (y_min + y_max) / 2.0
                dist = abs(y_query - mid)
                if dist < min_dist_to_band:
                    min_dist_to_band = dist
                    candidate_row = r_idx

        # Apply temporal hysteresis
        if tid not in self._row_history:
            self._row_history[tid] = []

        hist = self._row_history[tid]
        hist.append(candidate_row)
        if len(hist) > self.hysteresis_frames * 2:
            hist.pop(0)

        if tid not in self._confirmed_row:
            # First time seeing track: confirm immediately
            self._confirmed_row[tid] = candidate_row
            return candidate_row

        curr_confirmed = self._confirmed_row[tid]
        if candidate_row != curr_confirmed:
            # Only switch if last hysteresis_frames are ALL the new candidate
            recent = hist[-self.hysteresis_frames:]
            if len(recent) >= self.hysteresis_frames and all(r == candidate_row for r in recent):
                self._confirmed_row[tid] = candidate_row
                return candidate_row
            else:
                return curr_confirmed

    def get_confirmed_row(self, tid: int) -> Optional[int]:
        return self._confirmed_row.get(tid)

    def clear(self):
        self._confirmed_row.clear()
        self._row_history.clear()
        self._last_head_pos.clear()


class SeatAssociator:
    """
    Decoupled Seat Association Engine.
    Matches YOLO person tracks to PhysicalSeats within their classified row.
    """

    def __init__(
        self,
        max_dist_px: float = SEAT_MAX_DISTANCE_PX,
        ghost_hold_frames: int = GHOST_HOLD_FRAMES,
    ):
        self.max_dist_px = max_dist_px
        self.ghost_hold_frames = ghost_hold_frames
        self.row_classifier = RowClassifier()

        # Track history & state
        self._prev_track_to_seat: Dict[int, str] = {}    # track_id -> seat_label
        self._prev_seat_to_track: Dict[str, int] = {}    # seat_label -> track_id
        self._last_head_pos: Dict[int, Tuple[float, float]] = {}
        
        # Ghost tracking: seat_label -> dict of last known state
        self._ghost_seats: Dict[str, dict] = {}
        self.frame_idx = 0

    def associate(
        self,
        tracks: List[PersonTrack],
        topology: ClassroomTopology,
        frame_idx: int = 0,
    ) -> List[SeatAssignment]:
        """
        Main association step:
          1. Classify each track into a row band (with temporal hysteresis).
          2. Within each row, match tracks to seats using greedy min-cost with hysteresis.
          3. Unmatched tracks are UNASSIGNED.
          4. Missing seats are held as GHOST for up to ghost_hold_frames.
        """
        self.frame_idx = frame_idx

        # Invalidate stale seat bindings for tracks that teleported across room (YOLO ID swap)
        for t in tracks:
            hx, hy = t.head_point
            if t.track_id in self._last_head_pos:
                prev_hx, prev_hy = self._last_head_pos[t.track_id]
                if math.hypot(hx - prev_hx, hy - prev_hy) > 130.0:
                    self._prev_track_to_seat.pop(t.track_id, None)
            self._last_head_pos[t.track_id] = (hx, hy)
        if not topology.seats:
            # Empty topology: all tracks are UNASSIGNED
            return [
                SeatAssignment(
                    track_id=t.track_id,
                    seat_label="UNASSIGNED",
                    cost=999.0,
                    row_match=False,
                    status="UNASSIGNED",
                    prev_seat=self._prev_track_to_seat.get(t.track_id),
                    bbox=t.face_bbox,
                    person_bbox=t.bbox,
                    confidence=t.confidence,
                )
                for t in tracks
            ]

        # 1. Classify tracks by row
        tracks_by_row: Dict[int, List[PersonTrack]] = {r: [] for r in range(topology.rows)}
        for t in tracks:
            row_idx = self.row_classifier.classify(t, topology)
            if row_idx not in tracks_by_row:
                tracks_by_row[row_idx] = []
            tracks_by_row[row_idx].append(t)

        assignments: List[SeatAssignment] = []
        assigned_seat_labels: Set[str] = set()
        current_seat_to_track: Dict[str, int] = {}
        current_track_to_seat: Dict[int, str] = {}

        # 2. Process each row independently
        for r_idx in range(max(topology.rows, 1)):
            row_seats = topology.get_seats_for_row(r_idx)
            row_tracks = tracks_by_row.get(r_idx, [])

            if not row_seats:
                # Detections in row with no seats are unassigned
                for t in row_tracks:
                    assignments.append(
                        SeatAssignment(
                            track_id=t.track_id,
                            seat_label="UNASSIGNED",
                            cost=999.0,
                            row_match=False,
                            status="UNASSIGNED",
                            prev_seat=self._prev_track_to_seat.get(t.track_id),
                            row=r_idx,
                            bbox=t.face_bbox,
                            person_bbox=t.bbox,
                            confidence=t.confidence,
                        )
                    )
                continue

            # Build candidate pairs
            is_struct_ai = (topology.source == "structure_ai")
            pairs = []
            for t_idx, t in enumerate(row_tracks):
                tx, ty = t.head_point
                prev_seat = self._prev_track_to_seat.get(t.track_id)
                for s_idx, s in enumerate(row_seats):
                    sx, sy = s.anchor
                    dist = math.hypot(tx - sx, ty - sy)

                    # Check spatial regions
                    head_reg = s.expected_head_region
                    in_head = False
                    if head_reg and len(head_reg) == 4:
                        if head_reg[0] <= tx <= head_reg[2] and head_reg[1] <= ty <= head_reg[3]:
                            in_head = True

                    in_col = False
                    if s.column_boundary_left is not None and s.column_boundary_right is not None:
                        if s.column_boundary_left <= tx <= s.column_boundary_right:
                            in_col = True

                    prev_matched = (prev_seat == s.label and dist <= 120.0)

                    if is_struct_ai:
                        # Multi-Signal Spatial Scoring (Physical Furniture Relationship)
                        score, match_type = _compute_multi_signal_score(
                            t=t,
                            s=s,
                            prev_matched=prev_matched,
                            max_dist=self.max_dist_px,
                        )
                        cost = (1.0 - score) * 300.0
                        if dist <= self.max_dist_px:
                            pairs.append((cost, dist, in_head, in_col, score, match_type, t_idx, s_idx))
                    else:
                        # Baseline V2.1/V2.2 Distance-based Cost
                        cost = dist
                        if in_head:
                            cost -= 80.0
                        elif in_col:
                            cost -= 40.0
                        if prev_matched:
                            cost -= HYSTERESIS_BONUS_PX

                        if dist <= self.max_dist_px:
                            pairs.append((cost, dist, in_head, in_col, None, None, t_idx, s_idx))

            # Sort by cost ascending
            pairs.sort(key=lambda x: x[0])

            assigned_t = set()
            assigned_s = set()

            for item in pairs:
                if is_struct_ai:
                    cost, raw_dist, in_head, in_col, comp_score, m_type, t_idx, s_idx = item
                else:
                    cost, raw_dist, in_head, in_col, _, _, t_idx, s_idx = item
                    comp_score, m_type = None, None

                if t_idx in assigned_t or s_idx in assigned_s:
                    continue

                t = row_tracks[t_idx]
                s = row_seats[s_idx]
                prev_seat = self._prev_track_to_seat.get(t.track_id)

                if is_struct_ai and comp_score is not None:
                    seat_conf = comp_score
                    match_type = m_type
                else:
                    seat_conf, match_type = _compute_seat_confidence(
                        dist=raw_dist,
                        in_head_region=in_head,
                        in_column=in_col,
                        prev_matched=(prev_seat == s.label),
                        max_dist=self.max_dist_px,
                    )

                assignment = SeatAssignment(
                    track_id=t.track_id,
                    seat_label=s.label,
                    cost=raw_dist,
                    row_match=True,
                    status="OK",
                    prev_seat=prev_seat,
                    row=s.row,
                    col=s.col,
                    bbox=t.face_bbox,
                    person_bbox=t.bbox,
                    confidence=t.confidence,
                    seat_confidence=seat_conf,
                    match_type=match_type,
                    ghost=False,
                )
                assignments.append(assignment)
                assigned_t.add(t_idx)
                assigned_s.add(s_idx)
                assigned_seat_labels.add(s.label)
                current_seat_to_track[s.label] = t.track_id
                current_track_to_seat[t.track_id] = s.label

                # Update ghost store: this seat is actively seen
                self._ghost_seats[s.label] = {
                    "last_seen_frame": frame_idx,
                    "last_bbox": list(t.face_bbox),
                    "last_person_bbox": list(t.bbox),
                    "last_track_id": t.track_id,
                    "row": s.row,
                    "col": s.col,
                    "anchor": s.anchor,
                }

        # Step 2b: Cross-row rescue matching for any unassigned tracks
        unassigned_tracks = [t for t in tracks if t.track_id not in current_track_to_seat]
        free_seats = [s for s in topology.seats if s.label not in assigned_seat_labels]

        if unassigned_tracks and free_seats:
            rescue_pairs = []
            for t in unassigned_tracks:
                tx, ty = t.head_point
                bx, by = t.bottom_point
                prev_seat = self._prev_track_to_seat.get(t.track_id)
                t_row = self.row_classifier.get_confirmed_row(t.track_id)
                for s in free_seats:
                    sx, sy = s.anchor
                    dist = math.hypot(tx - sx, ty - sy)
                    cost = dist
                    if t_row is not None and s.row != t_row:
                        cost += 80.0  # Soft penalty for cross-row rescue
                    if prev_seat == s.label and dist <= 120.0:
                        cost -= (HYSTERESIS_BONUS_PX * 2.0)
                    if dist <= self.max_dist_px:
                        rescue_pairs.append((cost, dist, t, s))

            rescue_pairs.sort(key=lambda x: x[0])
            rescued_tids = set()
            for cost, raw_dist, t, s in rescue_pairs:
                if t.track_id in rescued_tids or s.label in assigned_seat_labels:
                    continue
                rescue_conf = round(max(0.20, 0.50 - (raw_dist / self.max_dist_px) * 0.3), 2)
                assignment = SeatAssignment(
                    track_id=t.track_id,
                    seat_label=s.label,
                    cost=raw_dist,
                    row_match=False,
                    status="OK",
                    prev_seat=self._prev_track_to_seat.get(t.track_id),
                    row=s.row,
                    col=s.col,
                    bbox=t.face_bbox,
                    person_bbox=t.bbox,
                    confidence=t.confidence,
                    seat_confidence=rescue_conf,
                    match_type="RESCUE",
                    ghost=False,
                )
                assignments.append(assignment)
                rescued_tids.add(t.track_id)
                assigned_seat_labels.add(s.label)
                current_seat_to_track[s.label] = t.track_id
                current_track_to_seat[t.track_id] = s.label
                self._ghost_seats[s.label] = {
                    "last_seen_frame": frame_idx,
                    "last_bbox": list(t.face_bbox),
                    "last_person_bbox": list(t.bbox),
                    "last_track_id": t.track_id,
                    "row": s.row,
                    "col": s.col,
                    "anchor": s.anchor,
                }

        # Step 2c: Any remaining unassigned tracks
        for t in tracks:
            if t.track_id not in current_track_to_seat:
                assignments.append(
                    SeatAssignment(
                        track_id=t.track_id,
                        seat_label="UNASSIGNED",
                        cost=999.0,
                        row_match=False,
                        status="UNASSIGNED",
                        prev_seat=self._prev_track_to_seat.get(t.track_id),
                        row=None,
                        bbox=t.face_bbox,
                        person_bbox=t.bbox,
                        confidence=t.confidence,
                        seat_confidence=0.0,
                        match_type="UNASSIGNED",
                    )
                )

        # 3. Handle ghost seats (seats in topology not assigned this frame)
        for s in topology.seats:
            if s.label not in assigned_seat_labels:
                if s.label in self._ghost_seats:
                    ghost_info = self._ghost_seats[s.label]
                    missed = frame_idx - ghost_info["last_seen_frame"]
                    if missed <= self.ghost_hold_frames:
                        ghost_conf = round(max(0.10, 0.80 - (missed / float(self.ghost_hold_frames)) * 0.7), 2)
                        # Re-inject as ghost assignment
                        assignments.append(
                            SeatAssignment(
                                track_id=ghost_info["last_track_id"],
                                seat_label=s.label,
                                cost=0.0,
                                row_match=True,
                                status="GHOST",
                                prev_seat=s.label,
                                row=s.row,
                                col=s.col,
                                bbox=ghost_info["last_bbox"],
                                person_bbox=ghost_info["last_person_bbox"],
                                confidence=0.0,
                                seat_confidence=ghost_conf,
                                match_type="GHOST",
                                ghost=True,
                            )
                        )
                        current_seat_to_track[s.label] = ghost_info["last_track_id"]

        # Update previous maps
        self._prev_track_to_seat = current_track_to_seat
        self._prev_seat_to_track = current_seat_to_track

        # Sort assignments: by row, then col, unassigned at the end
        def sort_key(a: SeatAssignment):
            r = a.row if a.row is not None else 99
            c = a.col if a.col is not None else 99
            return (0 if a.status != "UNASSIGNED" else 1, r, c)

        assignments.sort(key=sort_key)
        return assignments

    def reset(self):
        self._prev_track_to_seat.clear()
        self._prev_seat_to_track.clear()
        self._ghost_seats.clear()
        self._last_head_pos.clear()
        self.row_classifier.clear()
        self.frame_idx = 0


def _compute_seat_confidence(
    dist: float,
    in_head_region: bool,
    in_column: bool,
    prev_matched: bool,
    max_dist: float = SEAT_MAX_DISTANCE_PX
) -> Tuple[float, str]:
    """
    Computes V2.2 seat spatial assignment confidence in [0.0, 1.0] and match_type label.
    """
    base_score = max(0.0, 1.0 - (dist / max_dist))
    
    if in_head_region:
        score = min(1.0, base_score + 0.25)
        match_type = "EXACT_REGION"
    elif in_column:
        score = min(0.90, base_score + 0.15)
        match_type = "OVERLAP_REGION"
    else:
        score = min(0.65, base_score)
        match_type = "NEAREST_FALLBACK"

    if prev_matched:
        score = min(1.0, score + 0.10)

    return round(float(score), 2), match_type


def _compute_multi_signal_score(
    t: PersonTrack,
    s: PhysicalSeat,
    prev_matched: bool,
    max_dist: float = SEAT_MAX_DISTANCE_PX
) -> Tuple[float, str]:
    """
    Computes spatial score based on furniture-person physical geometry:
      score = (
          0.25 * head_alignment +
          0.20 * torso_alignment +
          0.15 * bottom_alignment +
          0.15 * desk_relationship +
          0.10 * depth_consistency +
          0.10 * temporal_consistency +
          0.05 * occupancy_geometry
      )
    """
    hx, hy = t.head_point
    bx, by = t.bottom_point

    # 1. Head alignment (0.50) — primary spatial anchor for seated students
    head_reg = s.expected_head_region
    in_head = False
    dist_anchor = math.hypot(hx - s.anchor[0], hy - s.anchor[1])
    if head_reg and len(head_reg) == 4:
        if head_reg[0] <= hx <= head_reg[2] and head_reg[1] <= hy <= head_reg[3]:
            head_alignment = 1.0
            in_head = True
        else:
            head_alignment = max(0.0, 1.0 - (dist_anchor / 160.0))
    else:
        head_alignment = max(0.0, 1.0 - (dist_anchor / max_dist))

    # 2. Torso alignment (0.20)
    torso_x = t.torso_point[0] if t.torso_point else hx
    torso_y = t.torso_point[1] if t.torso_point else (hy + (by - hy) * 0.45)
    torso_reg = s.torso_region
    if torso_reg and len(torso_reg) == 4:
        if torso_reg[0] <= torso_x <= torso_reg[2] and torso_reg[1] <= torso_y <= torso_reg[3]:
            torso_alignment = 1.0
        else:
            tcx = (torso_reg[0] + torso_reg[2]) / 2.0
            tcy = (torso_reg[1] + torso_reg[3]) / 2.0
            dist_t = math.hypot(torso_x - tcx, torso_y - tcy)
            torso_alignment = max(0.0, 1.0 - (dist_t / 200.0))
    else:
        torso_alignment = head_alignment * 0.85

    # 3. Bottom alignment (0.10)
    if s.desk_y is not None:
        dist_desk = abs(by - s.desk_y)
        bottom_alignment = max(0.0, 1.0 - (dist_desk / 200.0))
    else:
        bottom_alignment = 0.70

    # 4. Depth consistency (0.10)
    if t.depth is not None and s.expected_depth is not None:
        diff_depth = abs(t.depth - s.expected_depth)
        depth_consistency = max(0.0, 1.0 - (diff_depth / 0.30))
    else:
        depth_consistency = 0.75

    # 5. Temporal consistency (seat lock)
    temporal_consistency = 1.0 if prev_matched else 0.50

    # Weighted composite score
    composite_score = (
        0.45 * head_alignment +
        0.20 * torso_alignment +
        0.15 * bottom_alignment +
        0.10 * depth_consistency +
        0.10 * temporal_consistency
    )
    if prev_matched:
        composite_score = min(1.0, composite_score + 0.15)

    if in_head or head_alignment >= 0.70 or (prev_matched and head_alignment >= 0.50):
        match_type = "EXACT_REGION"
    elif head_alignment >= 0.40 or composite_score >= 0.55:
        match_type = "OVERLAP_REGION"
    elif composite_score >= 0.30:
        match_type = "NEAREST_FALLBACK"
    else:
        match_type = "RESCUE"

    return round(float(composite_score), 2), match_type

import importlib
_mod = importlib.import_module("core.spatial.seat_associator")
for _k, _v in _mod.__dict__.items():
    if not _k.startswith("__"):
        globals()[_k] = _v
