# -*- coding: utf-8 -*-
"""
SmartClass AI — Physical Classroom Topology (Truth 1)
=====================================================
Defines physical seats and classroom topology completely decoupled
from person tracking and student identity.

Core concepts:
  - PhysicalSeat: A seat exists in the physical world whether occupied or empty.
  - ClassroomTopology: The physical layout (seats, rows, row Y-bands).
  - TopologyConfig: Configuration for AUTO discovery or CONFIGURED mode.
  - TopologyDiscovery: Discovers topology from temporal clusters of person tracks.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np


@dataclass
class RowGeometry:
    """
    V2.2 Physical Row Geometry.
    Defines explicit physical meaning for row clusters.
    """
    row_idx: int                      # 0-indexed internal index
    label: str                        # "R1" or "R2"
    semantic: str                     # "FRONT" or "BACK"
    y_median: float                   # median y_feat of row occupants
    y_min: float                      # top boundary in image space (smaller Y)
    y_max: float                      # bottom boundary in image space (larger Y)

    def to_dict(self) -> dict:
        return {
            "row_idx": self.row_idx,
            "label": self.label,
            "semantic": self.semantic,
            "y_median": round(float(self.y_median), 2),
            "y_min": round(float(self.y_min), 2),
            "y_max": round(float(self.y_max), 2),
        }


@dataclass
class PhysicalSeat:
    """
    Truth 1: Physical Seat.
    A seat exists independently of detection, occupancy, or identity.
    """
    label: str                     # e.g. "R1C1", "R1C2", "R2C1"
    row: int                       # 0-indexed: 0 = Front row (R1), 1 = Back row (R2)
    col: int                       # 0-indexed within row: 0 = leftmost, 1 = next...
    anchor: Tuple[float, float]    # (cx, cy) anchor point in image coordinates
    bbox: List[int]                # [x1, y1, x2, y2] physical seat region
    desk_y: Optional[float] = None # estimated desk edge Y
    is_ghost: bool = False
    
    # V2.2 Spatial Region Extensions
    expected_head_region: Optional[List[int]] = None   # [x1, y1, x2, y2] expected head location
    expected_body_region: Optional[List[int]] = None   # [x1, y1, x2, y2] expected body bounding box
    column_boundary_left: Optional[float] = None       # x-min boundary for column
    column_boundary_right: Optional[float] = None      # x-max boundary for column

    # Structure AI Extensions (Geometry-first 3-coordinate model)
    seat_id: Optional[str] = None                      # Permanent geometric ID (e.g. "seat_01")
    bench_id: Optional[str] = None                     # Parent bench ID (e.g. "BENCH_01")
    desk_anchor: Optional[Tuple[float, float]] = None  # Contact point on furniture surface
    occupant_anchor: Optional[Tuple[float, float]] = None  # Seating position behind desk
    expected_depth: Optional[float] = None             # Monocular relative depth
    torso_region: Optional[List[int]] = None           # [x1, y1, x2, y2] expected torso location
    polygon: Optional[List[Tuple[int, int]]] = None    # Precise seat spatial polygon

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "seat_id": self.seat_id,
            "bench_id": self.bench_id,
            "row": self.row,
            "col": self.col,
            "anchor": list(self.anchor),
            "bbox": list(self.bbox),
            "desk_y": self.desk_y,
            "expected_depth": round(self.expected_depth, 3) if self.expected_depth is not None else None,
            "expected_head_region": self.expected_head_region,
            "expected_body_region": self.expected_body_region,
            "torso_region": self.torso_region,
            "column_boundary_left": round(self.column_boundary_left, 1) if self.column_boundary_left is not None else None,
            "column_boundary_right": round(self.column_boundary_right, 1) if self.column_boundary_right is not None else None,
        }


@dataclass
class ClassroomTopology:
    """
    Truth 1: Physical Classroom Topology.
    Contains the full physical arrangement of seats and row boundaries.
    """
    seats: List[PhysicalSeat] = field(default_factory=list)
    rows: int = 0
    cols_per_row: List[int] = field(default_factory=list)
    row_y_bands: List[Tuple[float, float]] = field(default_factory=list)  # (y_min, y_max) per row
    row_geometries: List[RowGeometry] = field(default_factory=list)      # Explicit semantic row metadata
    row_slant_theta: float = 0.0   # Camera perspective slant angle in radians
    source: str = "auto"           # "auto" | "configured" | "furniture"
    confidence: float = 1.0
    frame_size: Tuple[int, int] = (1920, 1080)
    structure_map: Optional[Any] = None

    def get_seat(self, label: str) -> Optional[PhysicalSeat]:
        for s in self.seats:
            if s.label == label or (s.seat_id and s.seat_id == label):
                return s
        return None

    def get_seats_for_row(self, row_idx: int) -> List[PhysicalSeat]:
        return [s for s in self.seats if s.row == row_idx]

    @classmethod
    def from_structure_map(cls, struct_map: Any) -> ClassroomTopology:
        """Converts a geometry-first ClassroomTopologyMap into a ClassroomTopology instance."""
        seats: List[PhysicalSeat] = []
        rows_set = sorted(list(set(s.row_id for s in struct_map.seats)))
        row_id_to_idx = {r_id: idx for idx, r_id in enumerate(rows_set)}
        cols_per_row = [0] * len(rows_set)

        # Compute row Y-bands dynamically from seat occupant anchors
        row_seat_ys: Dict[str, list] = {r_id: [] for r_id in rows_set}
        for s in struct_map.seats:
            row_seat_ys[s.row_id].append(float(s.center[1]))

        row_medians = {r_id: float(np.median(ys)) if ys else 0.0 for r_id, ys in row_seat_ys.items()}
        # Sort rows so row 0 is Front (larger Y)
        rows_sorted_by_y = sorted(rows_set, key=lambda r_id: row_medians[r_id], reverse=True)
        row_id_to_idx = {r_id: idx for idx, r_id in enumerate(rows_sorted_by_y)}
        cols_per_row = [0] * len(rows_sorted_by_y)

        n_r = len(rows_sorted_by_y)
        frame_h = float(struct_map.frame_size[1])
        row_y_bands = []
        for i in range(n_r):
            curr_y = row_medians[rows_sorted_by_y[i]]
            if n_r == 1:
                row_y_bands.append((0.0, frame_h))
            elif i == 0:
                next_y = row_medians[rows_sorted_by_y[1]]
                row_y_bands.append(((curr_y + next_y) / 2.0, frame_h))
            elif i == n_r - 1:
                prev_y = row_medians[rows_sorted_by_y[i - 1]]
                row_y_bands.append((0.0, (prev_y + curr_y) / 2.0))
            else:
                prev_y = row_medians[rows_sorted_by_y[i - 1]]
                next_y = row_medians[rows_sorted_by_y[i + 1]]
                y_hi = (prev_y + curr_y) / 2.0
                y_lo = (curr_y + next_y) / 2.0
                row_y_bands.append((min(y_lo, y_hi), max(y_lo, y_hi)))

        row_geometries = []

        for b in struct_map.benches:
            r_idx = row_id_to_idx.get(b.row_id, 0)
            y_band = row_y_bands[r_idx] if r_idx < len(row_y_bands) else (0.0, frame_h)
            row_geometries.append(RowGeometry(
                row_idx=r_idx,
                label=b.row_id,
                semantic=b.semantic,
                y_median=float(b.desk_edge_y),
                y_min=y_band[0],
                y_max=y_band[1]
            ))

        for s_geom in struct_map.seats:
            r_idx = row_id_to_idx.get(s_geom.row_id, 0)
            # Parse col index from display_label e.g. R1C3 -> col 2 (0-indexed)
            try:
                c_idx = int(s_geom.display_label.split("C")[-1]) - 1
            except Exception:
                c_idx = cols_per_row[r_idx]

            cols_per_row[r_idx] = max(cols_per_row[r_idx], c_idx + 1)

            seats.append(PhysicalSeat(
                label=s_geom.display_label,
                row=r_idx,
                col=c_idx,
                anchor=(float(s_geom.center[0]), float(s_geom.center[1])),
                bbox=list(s_geom.bbox),
                desk_y=float(s_geom.desk_contact_line[0][1]),
                expected_head_region=list(s_geom.head_region),
                expected_body_region=list(s_geom.bbox),
                column_boundary_left=float(s_geom.bbox[0]),
                column_boundary_right=float(s_geom.bbox[2]),
                seat_id=s_geom.seat_id,
                bench_id=s_geom.bench_id,
                desk_anchor=getattr(s_geom, "desk_anchor", (float(s_geom.center[0]), float(s_geom.center[1]))),
                occupant_anchor=getattr(s_geom, "occupant_anchor", (float(s_geom.center[0]), float(s_geom.center[1]))),
                expected_depth=float(s_geom.expected_depth),
                torso_region=list(s_geom.torso_region),
                polygon=list(s_geom.polygon)
            ))

        return cls(
            seats=seats,
            rows=len(rows_set),
            cols_per_row=cols_per_row,
            row_y_bands=row_y_bands,
            row_geometries=row_geometries,
            source="structure_ai",
            confidence=struct_map.confidence,
            frame_size=struct_map.frame_size,
            structure_map=struct_map
        )

    def to_dict(self) -> dict:
        return {
            "rows": self.rows,
            "cols_per_row": self.cols_per_row,
            "source": self.source,
            "confidence": self.confidence,
            "frame_size": list(self.frame_size),
            "row_y_bands": [list(b) for b in self.row_y_bands],
            "row_geometries": [rg.to_dict() for rg in self.row_geometries],
            "seats": [s.to_dict() for s in self.seats],
        }


@dataclass
class TopologyConfig:
    """
    Optional user-provided classroom layout constraints.
    Default is 'auto', where everything is discovered from data.
    """
    mode: str = "auto"                         # "auto" | "configured" | "structure_ai"
    rows: Optional[int] = None                 # e.g. 2
    cols_per_row: Optional[List[int]] = None   # e.g. [6, 6]


@dataclass
class PersonTrack:
    """
    Truth 2: Person Track.
    A tracked person detection from YOLO.
    """
    track_id: int
    bbox: List[int]                            # [x1, y1, x2, y2] full person bbox
    face_bbox: List[int]                       # [x1, y1, x2, y2] face crop bbox
    head_point: Tuple[float, float]            # (cx, cy) of face crop
    bottom_point: Tuple[float, float]          # (cx, y_bottom) of person bbox
    confidence: float
    frame_idx: int
    torso_point: Optional[Tuple[float, float]] = None
    depth: Optional[float] = None              # Monocular depth at person torso


@dataclass
class PersonObservation:
    """
    Decoupled Person Perception Observation.
    Separates human perception (track, landmarks, attention, activity)
    from physical seating structures.
    """
    track_id: int
    bbox: List[int]
    head_point: Tuple[float, float]
    torso_point: Tuple[float, float]
    bottom_point: Tuple[float, float]
    depth: float = 0.5
    face_landmarks: Optional[Any] = None
    attention: float = 0.0
    activity: str = "neutral"
    confidence: float = 1.0




class TopologyDiscovery:
    """
    Discovers physical classroom topology from temporal accumulation of person tracks.
    
    Principles:
      1. Aggregates person tracks over multiple calibration frames (e.g. 30-100+ frames).
      2. Uses median positions of persistent tracks to eliminate momentary movement.
      3. Discovers rows by vertical clustering (perspective causes front row to have larger Y).
      4. Discovers seats within rows by X-clustering of actual occupants (NO even-spacing assumption).
      5. Generates stable row Y-bands with clear midpoints for row-constrained association.
    """

    def __init__(self, config: Optional[TopologyConfig] = None, frame_size: Tuple[int, int] = (1920, 1080)):
        self.config = config or TopologyConfig(mode="auto")
        self.frame_size = frame_size
        self.frame_count = 0
        # track_id -> dict of accumulated observations
        self._tracks: Dict[int, dict] = {}
        self._frame_tids: List[set] = []

    def accumulate(self, tracks: List[PersonTrack]):
        """Accumulate person tracks from a single frame."""
        self.frame_count += 1
        tids_in_frame = set()
        for t in tracks:
            tid = t.track_id
            if tid is None:
                continue
            tids_in_frame.add(tid)
            if tid not in self._tracks:
                self._tracks[tid] = {
                    "count": 0,
                    "head_x": [],
                    "head_y": [],
                    "bot_y": [],
                    "bbox_w": [],
                    "bbox_h": [],
                    "conf": [],
                }
            entry = self._tracks[tid]
            entry["count"] += 1
            entry["head_x"].append(t.head_point[0])
            entry["head_y"].append(t.head_point[1])
            entry["bot_y"].append(t.bottom_point[1])
            entry["bbox_w"].append(t.bbox[2] - t.bbox[0])
            entry["bbox_h"].append(t.bbox[3] - t.bbox[1])
            entry["conf"].append(t.confidence)
        self._frame_tids.append(tids_in_frame)

    def accumulate_raw(self, detections: list, frame_idx: int = 0):
        """Helper to accumulate raw engine detections list directly."""
        tracks = []
        for d in detections:
            tid = d.get("id")
            if tid is None:
                continue
            pb = d.get("person_bbox", d.get("bbox"))
            fb = d.get("bbox", pb)
            pw = pb[2] - pb[0]
            hx = (fb[0] + fb[2]) / 2.0
            hy = (fb[1] + fb[3]) / 2.0
            bx = (pb[0] + pb[2]) / 2.0
            by = float(pb[3])
            tracks.append(PersonTrack(
                track_id=int(tid),
                bbox=list(pb),
                face_bbox=list(fb),
                head_point=(hx, hy),
                bottom_point=(bx, by),
                confidence=float(d.get("confidence", 0.5)),
                frame_idx=frame_idx,
            ))
        self.accumulate(tracks)

    def estimate(self, min_observations: int = 5) -> ClassroomTopology:
        """
        Produce ClassroomTopology from accumulated observations.
        Requires tracks to have at least min_observations to be considered stable.
        """
        W, H = self.frame_size

        # 1. Filter tracks that have sufficient observations
        stable_tracks = {}
        for tid, data in self._tracks.items():
            if data["count"] >= min_observations:
                stable_tracks[tid] = {
                    "tid": tid,
                    "count": data["count"],
                    "head_x": float(np.median(data["head_x"])),
                    "head_y": float(np.median(data["head_y"])),
                    "bot_y": float(np.median(data["bot_y"])),
                    "bbox_w": float(np.median(data["bbox_w"])),
                    "bbox_h": float(np.median(data["bbox_h"])),
                    "conf": float(np.mean(data["conf"])),
                }

        if not stable_tracks:
            # Fallback if no persistent tracks
            return self._build_empty_topology()

        # 2. Compute combined Y-depth feature for vertical clustering:
        # In classroom scenes, desk contact line (bot_y) is the dominant depth feature.
        # We blend 0.20 * head_y + 0.80 * bot_y for clean desk-plane separation.
        track_list = list(stable_tracks.values())
        for t in track_list:
            t["y_feat"] = 0.20 * t["head_y"] + 0.80 * t["bot_y"]

        # De-duplicate tracks that represent the same physical seat (tracker ID switches).
        # Crucial: If two tracks co-occur in ANY frame, they are distinct physical students.
        track_list.sort(key=lambda x: x["count"], reverse=True)
        merged_tracks = []
        for t in track_list:
            is_dup = False
            for m in merged_tracks:
                co_occur = sum(1 for f_set in self._frame_tids if t["tid"] in f_set and m["tid"] in f_set)
                d_head = math.hypot(t["head_x"] - m["head_x"], t["head_y"] - m["head_y"])
                d_yfeat = abs(t["y_feat"] - m["y_feat"])
                if co_occur == 0 and d_head < 60.0 and d_yfeat < 50.0:
                    is_dup = True
                    break
            if not is_dup:
                merged_tracks.append(t)
        track_list = merged_tracks

        # 3. Slant & Row Clustering (invariant to camera perspective angle)
        optimal_theta, chosen_k, rows_grouped = self._cluster_rows_slant(track_list)

        # Sort rows such that:
        # Row 0 = Front Row (closest to camera, larger projected Y) -> labelled "R1"
        # Row 1 = Middle / Second Row -> labelled "R2"
        # Row 2 = Back Row -> labelled "R3"
        def row_proj_y(r):
            return np.median([t["y_feat"] * math.cos(optimal_theta) - t["head_x"] * math.sin(optimal_theta) for t in r])
        rows_grouped.sort(key=row_proj_y, reverse=True)

        # Within each row, merge any tracker ID duplicates sharing the same physical seat (dx < 90px)
        deduped_rows = []
        for g in rows_grouped:
            g.sort(key=lambda t: t["head_x"])
            merged_g = []
            for t in g:
                if not merged_g or abs(t["head_x"] - merged_g[-1]["head_x"]) >= 90.0:
                    merged_g.append(t)
                else:
                    if t["count"] > merged_g[-1]["count"]:
                        merged_g[-1] = t
            deduped_rows.append(merged_g)
        rows_grouped = deduped_rows

        # 4. Compute row Y-bands in projected coordinate space
        row_y_bands = self._compute_row_bands_projected(rows_grouped, optimal_theta, H)

        # 5. Build explicit RowGeometries with physical semantics
        row_geometries: List[RowGeometry] = []
        for r_idx, r_tracks in enumerate(rows_grouped):
            y_med = float(np.median([t["y_feat"] for t in r_tracks])) if r_tracks else 0.0
            r_label = f"R{r_idx + 1}"
            y_band = row_y_bands[r_idx] if r_idx < len(row_y_bands) else (0.0, float(H))
            if len(rows_grouped) == 3:
                semantic = "FRONT" if r_idx == 0 else ("MIDDLE" if r_idx == 1 else "BACK")
            elif len(rows_grouped) == 2:
                semantic = "FRONT" if r_idx == 0 else "BACK"
            elif len(rows_grouped) == 1:
                semantic = "FRONT"
            else:
                semantic = "FRONT" if r_idx == 0 else ("BACK" if r_idx == len(rows_grouped) - 1 else f"ROW_{r_idx + 1}")

            row_geometries.append(RowGeometry(
                row_idx=r_idx,
                label=r_label,
                semantic=semantic,
                y_median=y_med,
                y_min=y_band[0],
                y_max=y_band[1],
            ))

        # 6. Build physical seats per row (Bench Grid Completion)
        # Determine maximum bench seat capacity across the room
        max_bench_cols = max(len(g) for g in rows_grouped)
        all_xs = [t["head_x"] for t in track_list]
        room_x_min = min(all_xs) if all_xs else 200.0
        room_x_max = max(all_xs) if all_xs else (W - 200.0)

        all_seats: List[PhysicalSeat] = []
        cols_per_row: List[int] = []

        for row_idx, row_tracks in enumerate(rows_grouped):
            cfg_cols = None
            if (self.config.mode == "configured" and 
                self.config.cols_per_row and 
                row_idx < len(self.config.cols_per_row)):
                cfg_cols = self.config.cols_per_row[row_idx]

            # In auto mode, if a bench has fewer occupants than the room's bench capacity,
            # synthesize the full physical bench grid across the room width.
            target_cols = cfg_cols or max_bench_cols
            seats_in_row = self._create_row_seats_regularized(row_tracks, row_idx, target_cols, room_x_min, room_x_max, W, H)
            cols_per_row.append(len(seats_in_row))
            all_seats.extend(seats_in_row)

        topology = ClassroomTopology(
            seats=all_seats,
            rows=len(rows_grouped),
            cols_per_row=cols_per_row,
            row_y_bands=row_y_bands,
            row_geometries=row_geometries,
            row_slant_theta=optimal_theta,
            source=self.config.mode,
            confidence=min(1.0, len(stable_tracks) / max(1, sum(cols_per_row))),
            frame_size=(W, H),
        )
        return topology

    def _cluster_rows_slant(self, track_list: list) -> Tuple[float, int, List[list]]:
        """
        Estimates the camera perspective slant angle theta and clusters tracks into rows.
        Uses BIC and physical row separation (min_row_gap >= 65px) to prevent oversplitting
        within-bench posture variation into false extra rows.
        """
        n = len(track_list)
        if n <= 1:
            return 0.0, 1, [track_list]
        if n <= 3:
            return 0.0, 1, [track_list]

        # Explicit user-configured row count override
        target_k = None
        if (self.config.mode in ("configured", "auto")) and self.config.rows:
            target_k = self.config.rows

        thetas = np.linspace(-np.radians(15), np.radians(15), 31)
        candidates = {}

        for k in [1, 2, 3]:
            best_bic = 1e9
            best_theta = 0.0
            best_groups = [track_list]
            best_min_gap = 0.0

            for theta in thetas:
                proj_vals = [t["y_feat"] * np.cos(theta) - t["head_x"] * np.sin(theta) for t in track_list]
                s_idx = np.argsort(proj_vals)
                s_proj = [proj_vals[i] for i in s_idx]
                sorted_tracks = [track_list[i] for i in s_idx]
                tot_var = float(np.var(s_proj))
                tot_mean = float(np.mean(s_proj))

                if k == 1:
                    wss = tot_var * n
                    groups = [sorted_tracks]
                elif k == 2:
                    best_v = -1.0
                    best_split = 1
                    for i in range(1, n):
                        g1, g2 = s_proj[:i], s_proj[i:]
                        v = (len(g1)*(float(np.mean(g1))-tot_mean)**2 + len(g2)*(float(np.mean(g2))-tot_mean)**2)/n
                        if v > best_v:
                            best_v = v
                            best_split = i
                    wss = (tot_var - best_v) * n
                    groups = [sorted_tracks[:best_split], sorted_tracks[best_split:]]
                elif k == 3:
                    if n < 6:
                        continue
                    best_v = -1.0
                    best_splits = (1, 2)
                    for i in range(1, n - 1):
                        for j in range(i + 1, n):
                            g1, g2, g3 = s_proj[:i], s_proj[i:j], s_proj[j:]
                            v = (len(g1)*(float(np.mean(g1))-tot_mean)**2 + len(g2)*(float(np.mean(g2))-tot_mean)**2 + len(g3)*(float(np.mean(g3))-tot_mean)**2)/n
                            if v > best_v:
                                best_v = v
                                best_splits = (i, j)
                    wss = (tot_var - best_v) * n
                    i, j = best_splits
                    groups = [sorted_tracks[:i], sorted_tracks[i:j], sorted_tracks[j:]]

                variance = max(1e-4, wss / n)
                bic = n * np.log(variance) + (k * 2) * np.log(n)
                row_meds = sorted([np.median([t["y_feat"] for t in g]) for g in groups])
                min_gap = min([row_meds[idx+1] - row_meds[idx] for idx in range(len(row_meds)-1)]) if len(row_meds) > 1 else 999.0

                if bic < best_bic:
                    best_bic = bic
                    best_theta = theta
                    best_groups = groups
                    best_min_gap = min_gap

            candidates[k] = (best_bic, best_theta, best_groups, best_min_gap)

        if target_k and target_k in candidates:
            chosen_k = target_k
        else:
            # Physical row validation: physical rows in classroom have >= 65px vertical separation
            valid_ks = [k for k in candidates if candidates[k][3] >= 65.0]
            chosen_k = min(valid_ks, key=lambda k: candidates[k][0]) if valid_ks else 1

        _, opt_theta, opt_groups, _ = candidates[chosen_k]
        return float(opt_theta), chosen_k, opt_groups

    def _compute_row_bands_projected(self, rows_grouped: List[list], theta: float, frame_h: int) -> List[Tuple[float, float]]:
        """
        Compute decision boundaries in projected coordinate space:
        y_proj = y_feat * cos(theta) - x * sin(theta).
        Row 0 = Front row (largest projected Y).
        """
        if not rows_grouped:
            return [(0.0, float(frame_h))]

        row_medians = [
            float(np.median([t["y_feat"] * math.cos(theta) - t["head_x"] * math.sin(theta) for t in r]))
            for r in rows_grouped
        ]

        bands = []
        n = len(rows_grouped)
        for i in range(n):
            curr_y = row_medians[i]
            if n == 1:
                bands.append((-9999.0, 9999.0))
            elif i == 0:
                # Front row: midpoint with next row (smaller projected Y)
                next_y = row_medians[1]
                mid_y = (curr_y + next_y) / 2.0
                bands.append((mid_y, 9999.0))
            elif i == n - 1:
                # Backmost row: midpoint with previous row
                prev_y = row_medians[i - 1]
                mid_y = (prev_y + curr_y) / 2.0
                bands.append((-9999.0, mid_y))
            else:
                prev_y = row_medians[i - 1]
                next_y = row_medians[i + 1]
                y_hi = (prev_y + curr_y) / 2.0
                y_lo = (curr_y + next_y) / 2.0
                bands.append((min(y_lo, y_hi), max(y_lo, y_hi)))

        return bands

    def _create_row_seats_regularized(
        self,
        row_tracks: list,
        row_idx: int,
        target_cols: int,
        room_x_min: float,
        room_x_max: float,
        frame_w: int,
        frame_h: int
    ) -> List[PhysicalSeat]:
        """
        Creates physical seats for this row / bench with regular grid completion.
        If the bench has fewer active occupants than target_cols, physical empty seats
        are synthesized across the bench width so the classroom structure is complete.
        """
        row_label_num = row_idx + 1
        seats = []

        if not row_tracks and target_cols <= 0:
            return seats

        row_tracks.sort(key=lambda t: t["head_x"])
        med_hy = float(np.median([t["head_y"] for t in row_tracks])) if row_tracks else float(frame_h * 0.5)
        med_by = float(np.median([t["bot_y"] for t in row_tracks])) if row_tracks else (med_hy + 200.0)
        med_bw = float(np.median([t["bbox_w"] for t in row_tracks])) if row_tracks else 200.0
        med_bh = float(np.median([t["bbox_h"] for t in row_tracks])) if row_tracks else 250.0

        if len(row_tracks) == target_cols:
            grid_xs = [t["head_x"] for t in row_tracks]
        else:
            grid_xs = np.linspace(room_x_min, room_x_max, target_cols).tolist()

        n_seats = len(grid_xs)
        for col_idx, gx in enumerate(grid_xs):
            col_label_num = col_idx + 1
            label = f"R{row_label_num}C{col_label_num}"

            # Check if an observed occupant matches this grid seat
            best_track = None
            min_dx = float("inf")
            for t in row_tracks:
                dx = abs(t["head_x"] - gx)
                if dx < min_dx and dx < (med_bw * 0.85):
                    min_dx = dx
                    best_track = t

            if best_track is not None:
                cx = best_track["head_x"]
                cy = best_track["head_y"]
                bot_y = best_track["bot_y"]
                bw = max(60.0, best_track["bbox_w"] * 1.1)
                bh = max(80.0, best_track["bbox_h"] * 1.05)
            else:
                # Synthesized empty physical seat on this bench
                cx = gx
                cy = med_hy
                bot_y = med_by
                bw = max(60.0, med_bw * 1.1)
                bh = max(80.0, med_bh * 1.05)

            x1 = max(0, int(cx - bw / 2.0))
            y1 = max(0, int(cy - bh * 0.3))
            x2 = min(frame_w, int(cx + bw / 2.0))
            y2 = min(frame_h, int(bot_y + 10))

            # Column boundaries
            if col_idx == 0:
                if n_seats > 1:
                    spacing = grid_xs[1] - grid_xs[0]
                    left_col = max(0.0, cx - spacing / 2.0)
                else:
                    left_col = max(0.0, cx - bw)
            else:
                left_col = (grid_xs[col_idx - 1] + cx) / 2.0

            if col_idx == n_seats - 1:
                if n_seats > 1:
                    spacing = grid_xs[-1] - grid_xs[-2]
                    right_col = min(float(frame_w), cx + spacing / 2.0)
                else:
                    right_col = min(float(frame_w), cx + bw)
            else:
                right_col = (cx + grid_xs[col_idx + 1]) / 2.0

            head_h = max(40.0, bh * 0.4)
            head_y1 = max(0, int(cy - head_h * 0.6))
            head_y2 = min(frame_h, int(cy + head_h * 0.6))
            exp_head_region = [int(left_col), head_y1, int(right_col), head_y2]

            body_y1 = max(0, int(cy - bh * 0.3))
            body_y2 = min(frame_h, int(bot_y + 15))
            exp_body_region = [int(left_col), body_y1, int(right_col), body_y2]

            seats.append(PhysicalSeat(
                label=label,
                row=row_idx,
                col=col_idx,
                anchor=(cx, cy),
                bbox=[x1, y1, x2, y2],
                desk_y=bot_y,
                expected_head_region=exp_head_region,
                expected_body_region=exp_body_region,
                column_boundary_left=left_col,
                column_boundary_right=right_col,
            ))

        return seats

    def _create_row_seats(self, row_tracks: list, row_idx: int, cfg_cols: Optional[int], frame_w: int, frame_h: int) -> List[PhysicalSeat]:
        """
        Create physical seats from discovered cluster centroids in this row.
        Row labels: R1 = Front row (row_idx=0), R2 = Back row (row_idx=1).
        Seats are numbered C1..CN left-to-right.
        Computes explicit column boundaries and head/body expected regions for V2.2.
        """
        row_label_num = row_idx + 1  # 1-indexed for display: R1 = Front, R2 = Back
        seats = []

        if not row_tracks:
            return seats

        n_seats = len(row_tracks)
        anchor_xs = [t["head_x"] for t in row_tracks]

        for col_idx, t in enumerate(row_tracks):
            col_label_num = col_idx + 1
            label = f"R{row_label_num}C{col_label_num}"
            cx = t["head_x"]
            cy = t["head_y"]
            bw = max(60.0, t["bbox_w"] * 1.1)
            bh = max(80.0, t["bbox_h"] * 1.05)
            
            x1 = max(0, int(cx - bw / 2.0))
            y1 = max(0, int(cy - bh * 0.3))
            x2 = min(frame_w, int(cx + bw / 2.0))
            y2 = min(frame_h, int(t["bot_y"] + 10))

            # Column boundaries (midpoints with left and right neighbors)
            if col_idx == 0:
                if n_seats > 1:
                    spacing = anchor_xs[1] - anchor_xs[0]
                    left_col = max(0.0, cx - spacing / 2.0)
                else:
                    left_col = max(0.0, cx - bw)
            else:
                left_col = (anchor_xs[col_idx - 1] + cx) / 2.0

            if col_idx == n_seats - 1:
                if n_seats > 1:
                    spacing = anchor_xs[-1] - anchor_xs[-2]
                    right_col = min(float(frame_w), cx + spacing / 2.0)
                else:
                    right_col = min(float(frame_w), cx + bw)
            else:
                right_col = (cx + anchor_xs[col_idx + 1]) / 2.0

            # Expected head region: column boundaries x head height span
            head_h = max(40.0, bh * 0.4)
            head_y1 = max(0, int(cy - head_h * 0.6))
            head_y2 = min(frame_h, int(cy + head_h * 0.6))
            exp_head_region = [int(left_col), head_y1, int(right_col), head_y2]

            # Expected body region: column boundaries x full person height span
            body_y1 = max(0, int(cy - bh * 0.3))
            body_y2 = min(frame_h, int(t["bot_y"] + 15))
            exp_body_region = [int(left_col), body_y1, int(right_col), body_y2]

            seats.append(PhysicalSeat(
                label=label,
                row=row_idx,
                col=col_idx,
                anchor=(cx, cy),
                bbox=[x1, y1, x2, y2],
                desk_y=t["bot_y"],
                expected_head_region=exp_head_region,
                expected_body_region=exp_body_region,
                column_boundary_left=left_col,
                column_boundary_right=right_col,
            ))

        return seats

    def _build_empty_topology(self) -> ClassroomTopology:
        return ClassroomTopology(
            seats=[],
            rows=0,
            cols_per_row=[],
            row_y_bands=[],
            source="empty",
            confidence=0.0,
            frame_size=self.frame_size,
        )


def create_classroom_grid_topology(
    rows: int = 3,
    cols: int = 2,
    frame_size: Tuple[int, int] = (1920, 1080),
) -> ClassroomTopology:
    """
    Parametric grid topology generator.
    Creates an evenly-spaced, perspective-scaled physical ClassroomTopology for any
    row x column count and resolution without hardcoded pixel coordinates.
    """
    W, H = frame_size
    seats: List[PhysicalSeat] = []

    # In camera perspective:
    # Row 0 = Front Row (larger Y, closer to camera)
    # Row rows-1 = Back Row (smaller Y, further from camera)
    row_anchors_y = []
    for r in range(rows):
        if rows == 1:
            y_norm = 0.55
        else:
            # Linear spread between back (0.35) and front (0.65)
            y_norm = 0.65 - (r / (rows - 1)) * 0.30
        row_anchors_y.append(float(y_norm * H))

    # Row bands midpoints
    row_y_bands = []
    for r in range(rows):
        curr_y = row_anchors_y[r]
        if rows == 1:
            row_y_bands.append((0.0, float(H)))
        elif r == 0:
            next_y = row_anchors_y[1]
            row_y_bands.append(((curr_y + next_y) / 2.0, float(H)))
        elif r == rows - 1:
            prev_y = row_anchors_y[r - 1]
            row_y_bands.append((0.0, (prev_y + curr_y) / 2.0))
        else:
            prev_y = row_anchors_y[r - 1]
            next_y = row_anchors_y[r + 1]
            y_hi = (prev_y + curr_y) / 2.0
            y_lo = (curr_y + next_y) / 2.0
            row_y_bands.append((min(y_lo, y_hi), max(y_lo, y_hi)))

    row_geometries = []
    for r in range(rows):
        r_label = f"R{r + 1}"
        y_band = row_y_bands[r]
        if rows == 3:
            semantic = "FRONT" if r == 0 else ("MIDDLE" if r == 1 else "BACK")
        elif rows == 2:
            semantic = "FRONT" if r == 0 else "BACK"
        elif rows == 1:
            semantic = "FRONT"
        else:
            semantic = "FRONT" if r == 0 else ("BACK" if r == rows - 1 else f"ROW_{r + 1}")
        row_geometries.append(RowGeometry(
            row_idx=r,
            label=r_label,
            semantic=semantic,
            y_median=row_anchors_y[r],
            y_min=y_band[0],
            y_max=y_band[1],
        ))

    # Generate seats
    col_step = float(W) / float(cols)
    for r in range(rows):
        cy = row_anchors_y[r]
        # Perspective scaling: front row students are wider than back row
        scale = 1.0 - (r / max(1, rows - 1)) * 0.35 if rows > 1 else 1.0
        bw = max(80.0, min(col_step * 0.8, 260.0 * scale))
        bh = max(100.0, 320.0 * scale)
        desk_y = min(float(H), cy + bh * 0.5)

        for c in range(cols):
            cx = (c + 0.5) * col_step
            left_col = c * col_step
            right_col = (c + 1) * col_step
            x1 = max(0, int(cx - bw / 2.0))
            y1 = max(0, int(cy - bh * 0.3))
            x2 = min(W, int(cx + bw / 2.0))
            y2 = min(H, int(cy + bh * 0.7))

            label = f"R{r + 1}C{c + 1}"
            seats.append(PhysicalSeat(
                label=label,
                row=r,
                col=c,
                anchor=(cx, cy),
                bbox=[x1, y1, x2, y2],
                desk_y=desk_y,
                expected_head_region=[int(cx - bw * 0.4), int(cy - bh * 0.3), int(cx + bw * 0.4), int(cy + bh * 0.3)],
                expected_body_region=[x1, y1, x2, y2],
                column_boundary_left=left_col,
                column_boundary_right=right_col,
            ))

    return ClassroomTopology(
        seats=seats,
        rows=rows,
        cols_per_row=[cols] * rows,
        row_y_bands=row_y_bands,
        row_geometries=row_geometries,
        source="configured",
        confidence=1.0,
        frame_size=frame_size,
    )
import importlib
_mod = importlib.import_module("core.spatial.topology")
for _k, _v in _mod.__dict__.items():
    if not _k.startswith("__"):
        globals()[_k] = _v
