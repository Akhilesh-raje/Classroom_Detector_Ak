"""
SmartClass AI — Pipeline Stage: Detection
Handles YOLO-based person/face/phone/laptop/earphone detection and
prepares PersonTrack representations for spatial association.
"""
from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional
from core.config import YOLO_CONF
from core.detector import detect_faces_and_phones
from core.topology import PersonTrack, ClassroomTopology


class DetectionStage:
    """Runs detection and forms track representations."""

    def __init__(self, min_conf: float = YOLO_CONF):
        self.min_conf = min_conf

    def execute(
        self,
        frame,
        persist: bool = True,
        topology: Optional[ClassroomTopology] = None,
        frame_idx: int = 0,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[PersonTrack], Dict[int, PersonTrack]]:
        """
        Runs object detection on the frame and converts detections into PersonTrack objects.

        Returns:
            raw_dets: List of raw person detection dicts
            phones: List of detected phones
            laptops: List of detected laptops
            earphones: List of detected earphones
            tracks: List of PersonTrack objects for associator
            t_by_id: Map of track_id -> PersonTrack
        """
        raw_dets, phones, laptops, earphones = detect_faces_and_phones(
            frame, min_conf=self.min_conf, persist=persist
        )

        tracks: List[PersonTrack] = []
        t_by_id: Dict[int, PersonTrack] = {}

        if topology is not None and topology.seats:
            temp_id = 800
            filtered_dets = []
            frame_h = frame.shape[0]
            for d in raw_dets:
                pb = d.get("person_bbox", d.get("bbox"))
                # Exclude standing instructor/teacher from student seat association
                if (pb[3] - pb[1]) > (frame_h * 0.65) and pb[1] < (frame_h * 0.22):
                    continue
                filtered_dets.append(d)

            # Suppress duplicate overlapping person detections (torso vs full body)
            keep_dets = []
            for i, d1 in enumerate(filtered_dets):
                pb1 = d1.get("person_bbox", d1.get("bbox"))
                suppress = False
                for j, d2 in enumerate(filtered_dets):
                    if i == j:
                        continue
                    pb2 = d2.get("person_bbox", d2.get("bbox"))
                    ix1, iy1 = max(pb1[0], pb2[0]), max(pb1[1], pb2[1])
                    ix2, iy2 = min(pb1[2], pb2[2]), min(pb1[3], pb2[3])
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    if inter > 0:
                        a1 = (pb1[2] - pb1[0]) * (pb1[3] - pb1[1])
                        a2 = (pb2[2] - pb2[0]) * (pb2[3] - pb2[1])
                        iou = inter / max(1, a1 + a2 - inter)
                        if iou > 0.45 and (d2.get("confidence", 0) > d1.get("confidence", 0) or (d2.get("confidence", 0) == d1.get("confidence", 0) and j < i)):
                            suppress = True
                            break
                if not suppress:
                    keep_dets.append(d1)

            for d in keep_dets:
                tid = d.get("id")
                if tid is None:
                    tid = temp_id
                    temp_id += 1
                    d["id"] = tid
                pb = d.get("person_bbox", d.get("bbox"))
                fb = d.get("bbox", pb)
                hx = (fb[0] + fb[2]) / 2.0
                hy = (fb[1] + fb[3]) / 2.0
                bx = (pb[0] + pb[2]) / 2.0
                by = float(pb[3])
                tx = bx
                ty = hy + (by - hy) * 0.40

                p_depth = None
                if topology.source == "structure_ai":
                    # Monocular perspective depth proxy based on vertical position
                    p_depth = max(0.05, min(0.95, (by - 300.0) / 780.0 * 0.65))

                pt = PersonTrack(
                    track_id=int(tid),
                    bbox=list(pb),
                    face_bbox=list(fb),
                    head_point=(hx, hy),
                    bottom_point=(bx, by),
                    confidence=float(d.get("confidence", 0.5)),
                    frame_idx=frame_idx,
                    torso_point=(tx, ty),
                    depth=p_depth,
                )
                tracks.append(pt)
                t_by_id[int(tid)] = pt

        return raw_dets, phones, laptops, earphones, tracks, t_by_id
