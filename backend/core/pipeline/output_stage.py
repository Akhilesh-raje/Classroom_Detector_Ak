"""
SmartClass AI — Pipeline Stage: Output & Scene State Construction
Synthesizes entity-level perception results into both backward-compatible API payloads
and Universal Scene Representation (SceneState) with evidence tracking and relationships.
"""
from __future__ import annotations

import time
from typing import List, Dict, Any, Optional, Tuple
from core.scene.scene_state import EntityState, SceneState, SceneRelationship
from core.topology import ClassroomTopology


class OutputStage:
    """Formats engine output and constructs Universal Scene State."""

    def execute(
        self,
        processed_faces: List[Dict[str, Any]],
        topology: Optional[ClassroomTopology] = None,
        frame_idx: int = 0,
        timestamp: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], SceneState]:
        """
        Builds SceneState and validates output payload.

        Returns:
            processed_faces: The validated face dictionary list
            scene_state: The unified SceneState object
        """
        ts = timestamp if timestamp is not None else time.time()
        entities: List[EntityState] = []
        unknowns: List[str] = []
        relationships: List[SceneRelationship] = []

        total_seats = len(topology.seats) if (topology and topology.seats) else 0
        occupied_seats = 0
        total_attention = 0.0
        active_entities_count = 0

        # Build EntityState for each processed detection
        for f in processed_faces:
            sid = f.get("id")
            seat_lbl = f.get("seat_label") or f.get("gridCell")
            status = f.get("status", "OK")
            is_ghost = f.get("ghost", False)
            tid = f.get("track_id")
            entity_id = f"person_{tid}" if tid is not None else f"seat_{seat_lbl}"

            if status != "UNASSIGNED" and not is_ghost:
                occupied_seats += 1

            attn_score = float(f.get("attentionScore", 0.0))
            if not is_ghost:
                total_attention += attn_score
                active_entities_count += 1

            # Determine seat contact anchor (cx, y_bottom)
            bbox = f.get("bbox", [0, 0, 0, 0])
            cx = (bbox[0] + bbox[2]) / 2.0
            y_bottom = float(bbox[3])
            seat_anchor = (cx, y_bottom)

            # Row/col metadata
            topo_dict = f.get("topology", {})
            row = topo_dict.get("row") if topo_dict.get("row") is not None else f.get("row")
            col = topo_dict.get("col") if topo_dict.get("col") is not None else f.get("col")

            # Pose & Gaze
            head_pose = f.get("headPose", {"yaw": 0.0, "pitch": 0.0, "roll": 0.0})
            yaw = head_pose.get("yaw", 0.0)
            pitch = head_pose.get("pitch", 0.0)
            if abs(yaw) > 30.0:
                head_dir = "RIGHT" if yaw > 0 else "LEFT"
            elif pitch < -15.0:
                head_dir = "DOWN"
            else:
                head_dir = "FRONT"

            # Check unknowns
            if status == "UNASSIGNED":
                unknowns.append(f"Entity {entity_id} detected but unassigned to any seat")
            if f.get("geometric_match", 1.0) < 0.65 and status != "UNASSIGNED":
                unknowns.append(f"Low geometric confidence for seat {seat_lbl} ({f.get('geometric_match', 0.0):.2f})")

            # Identity info
            ident_dict = f.get("identity", {})

            ent = EntityState(
                entity_id=entity_id,
                entity_class="person",
                bbox=bbox,
                seat_anchor=seat_anchor,
                depth=None,
                pose_state="seated",
                head_direction=head_dir,
                head_pose=head_pose,
                gaze_confidence=1.0 if not is_ghost else 0.0,
                activity=f.get("activity"),
                activity_confidence=0.85 if not is_ghost else 0.0,
                attention_score=attn_score,
                attention_raw=float(f.get("attentionRaw", 0.0)),
                attention_breakdown=f.get("attentionBreakdown", {}),
                seat_id=sid,
                seat_label=seat_lbl,
                row=row,
                col=col,
                student_id=ident_dict.get("student_id"),
                student_name=ident_dict.get("student_name"),
                identity_confidence=float(ident_dict.get("confidence", 0.0)),
                detection_confidence=float(f.get("geometric_match", 1.0)),
                tracking_confidence=1.0 if not is_ghost else 0.5,
                association_confidence=float(f.get("geometric_match", 1.0)),
                status=status,
                is_ghost=is_ghost,
                cost=float(f.get("cost", 0.0)),
                phone_detected=bool(f.get("phoneDetected", False)),
                laptop_detected=bool(f.get("laptopDetected", False)),
                earphone_detected=bool(f.get("earphoneDetected", False)),
                metadata={
                    "cropUsed": f.get("cropUsed"),
                    "trend": f.get("trend"),
                    "emotion": f.get("emotion"),
                }
            )
            entities.append(ent)

        # ── Infer Scene Graph Relationships ───────────────────────────────────
        for i, e1 in enumerate(entities):
            for e2 in entities[i + 1:]:
                if e1.row is not None and e1.row == e2.row:
                    relationships.append(SceneRelationship(
                        source_id=e1.entity_id,
                        relation="same_row",
                        target_id=e2.entity_id,
                        confidence=1.0,
                        metadata={"row": e1.row}
                    ))
                    if e1.col is not None and e2.col is not None and abs(e1.col - e2.col) == 1:
                        relationships.append(SceneRelationship(
                            source_id=e1.entity_id,
                            relation="adjacent_to",
                            target_id=e2.entity_id,
                            confidence=0.95
                        ))

        avg_attention = (total_attention / active_entities_count) if active_entities_count > 0 else 0.0

        topo_summary = None
        if topology is not None:
            topo_summary = {
                "source": topology.source,
                "rows": topology.rows,
                "total_seats": len(topology.seats),
                "cols_per_row": topology.cols_per_row,
                "confidence": topology.confidence,
            }

        scene_state = SceneState(
            timestamp=ts,
            frame_idx=frame_idx,
            entities=entities,
            people_count=active_entities_count,
            occupied_seats=occupied_seats,
            total_seats=total_seats,
            average_attention=round(avg_attention, 1),
            topology=topo_summary,
            relationships=relationships,
            scene_confidence=1.0 if not unknowns else max(0.5, 1.0 - 0.1 * len(unknowns)),
            unknowns=unknowns,
        )

        return processed_faces, scene_state
