"""
SmartClass AI — Pipeline Stage: Spatial Association
Maps detected person tracks to physical seat anchors using Hungarian bipartite matching
with geometry and temporal continuity priors.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
from core.topology import ClassroomTopology, PersonTrack
from core.seat_associator import SeatAssociator, SeatAssignment
from core.stabilizer import SeatStabilizer


class AssociationStage:
    """Assigns detected persons to physical seats."""

    def execute(
        self,
        tracks: List[PersonTrack],
        raw_dets: List[Dict[str, Any]],
        topology: Optional[ClassroomTopology],
        associator: SeatAssociator,
        seat_label_to_id: Dict[str, int],
        stabilizer: SeatStabilizer,
        frame_idx: int,
        W_px: int,
        H_px: int,
        t_by_id: Dict[int, PersonTrack],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Executes seat association.

        Returns:
            mapped_dets: List of assigned detection records
            benchmark_records: Records formatted for benchmark logging
        """
        mapped_dets: List[Dict[str, Any]] = []
        benchmark_records: List[Dict[str, Any]] = []

        if topology is not None and topology.seats:
            assignments: List[SeatAssignment] = associator.associate(
                tracks, topology, frame_idx=frame_idx
            )
            raw_by_id = {d.get("id"): d for d in raw_dets if d.get("id") is not None}

            for a in assignments:
                raw_d = raw_by_id.get(a.track_id) if (a.track_id is not None and not a.ghost) else None
                sid = seat_label_to_id.get(a.seat_label, 900 + (a.track_id or 0)) if a.status != "UNASSIGNED" else 999

                det_entry = {
                    "id": a.track_id,
                    "track_id": a.track_id,
                    "seat_id": sid,
                    "seat_label": a.seat_label,
                    "status": a.status,
                    "cost": a.cost,
                    "prev_seat": a.prev_seat,
                    "row": a.row,
                    "col": a.col,
                    "bbox": list(a.bbox) if (raw_d is None) else list(raw_d.get("bbox", a.bbox)),
                    "person_bbox": list(raw_d.get("person_bbox") or raw_d.get("bbox") or a.person_bbox or a.bbox) if raw_d else (list(a.person_bbox) if a.person_bbox else list(a.bbox)),
                    "confidence": a.confidence if (raw_d is None) else float(raw_d.get("confidence", 0.0)),
                    "seat_confidence": a.seat_confidence,
                    "geometric_match": a.seat_confidence,
                    "match_type": a.match_type,
                    "crops": raw_d.get("crops", []) if raw_d else [],
                    "ghost": a.ghost,
                    # Structured hierarchy
                    "topology": {
                        "seat_label": a.seat_label,
                        "row": a.row,
                        "col": a.col,
                        "seat_confidence": a.seat_confidence,
                        "geometric_match": a.seat_confidence,
                        "match_type": a.match_type,
                    },
                    "state": {
                        "track_id": a.track_id,
                        "status": a.status,
                        "cost": a.cost,
                        "ghost": a.ghost,
                    },
                    "identity": {
                        "student_id": None,
                        "student_name": "Unrecognized",
                        "confidence": 0.0,
                    }
                }
                mapped_dets.append(det_entry)

                # Record benchmark observation
                row_label = f"R{a.row + 1}" if a.row is not None else "UNASSIGNED"
                pt = t_by_id.get(a.track_id)
                hx = f"{pt.head_point[0]:.1f}" if (pt and not a.ghost) else "GHOST"
                hy = f"{pt.head_point[1]:.1f}" if (pt and not a.ghost) else "GHOST"
                by = f"{pt.bottom_point[1]:.1f}" if (pt and not a.ghost) else "GHOST"

                benchmark_records.append({
                    "frame": frame_idx,
                    "track_id": a.track_id if a.track_id is not None else "GHOST",
                    "head_x": hx,
                    "head_y": hy,
                    "bottom_y": by,
                    "row": row_label,
                    "seat": a.seat_label,
                    "cost": round(a.cost, 1),
                    "status": a.status,
                    "prev_seat": a.prev_seat or a.seat_label,
                    "geometric_match": a.seat_confidence,
                    "match_type": a.match_type,
                })
        else:
            mapped_dets = stabilizer.update_and_map(raw_dets, W_px, H_px)

        return mapped_dets, benchmark_records
