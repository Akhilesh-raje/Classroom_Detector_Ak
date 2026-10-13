"""
SmartClass AI — Pipeline Stage: Analysis
Performs multimodal student state analysis: face pose, gaze, attention score,
activity classification, behavior event logging, and facial recognition.
"""
from __future__ import annotations

import time
import math
import cv2
import base64
from io import BytesIO
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple, Callable

from core.config import SMALL_BBOX_FRAC, FACE_MATCH_INTERVAL
from core.detector import analyze_face_crop, _is_within_padded_bbox
from core.face_recognition_engine import face_rec_engine
from core.student_state import StudentState
from core.behavior_tracker import BehaviorTracker


def crop_to_b64(frame, x1: int, y1: int, x2: int, y2: int, size: int = 128) -> Optional[str]:
    """Extract and base64-encode a face crop thumbnail."""
    c = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if c.size == 0:
        return None
    img = Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)).resize((size, size))
    buf = BytesIO()
    img.save(buf, "JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def compute_trend(hist) -> str:
    """Compute recent trend string (UP/DN/STDY)."""
    if len(hist) < 12:
        return "  ~   "
    r = sum(list(hist)[-6:]) / 6
    o = sum(list(hist)[-12:-6]) / 6
    d = r - o
    if d > 2.5:
        return "  UP  "
    if d < -2.5:
        return "  DN  "
    return " STDY "


def nearest_grid_cell(spatial_grid, cx: float, cy: float):
    """Return the GridCell nearest to (cx, cy)."""
    if spatial_grid is None or not spatial_grid.cells:
        return None
    best = None
    best_dist = float("inf")
    for cell in spatial_grid.cells:
        bx1, by1, bx2, by2 = cell.bbox
        ccx = (bx1 + bx2) / 2
        ccy = (by1 + by2) / 2
        dist = math.hypot(cx - ccx, cy - ccy)
        if dist < best_dist or (dist == best_dist and best is not None and
                                (cell.row < best.row or
                                 (cell.row == best.row and cell.col < best.col))):
            best_dist = dist
            best = cell
    return best


class AnalysisStage:
    """Extracts behavioral, attention, and identity signals per mapped detection."""

    def execute(
        self,
        frame,
        mapped_dets: List[Dict[str, Any]],
        raw_dets: List[Dict[str, Any]],
        phones: List[Dict[str, Any]],
        laptops: List[Dict[str, Any]],
        earphones: List[Dict[str, Any]],
        student_states: Dict[int, StudentState],
        behavior_trackers: Dict[int, BehaviorTracker],
        face_match_counters: Dict[int, int],
        registered_faces: List[Tuple[Any, str, Any]],
        session_start_time: float,
        W_px: int,
        H_px: int,
        spatial_grid: Any = None,
        seat_to_grid_label_fn: Optional[Callable[[int], str]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Analyzes each detection, computes attention, tracks behaviors, matches identity.

        Returns:
            processed_faces: Full list of processed face/person dicts
            new_attendance_events: Attendance events queued in this frame
        """
        frame_area = W_px * H_px
        processed_faces: List[Dict[str, Any]] = []
        new_attendance_events: List[Dict[str, Any]] = []

        def get_state(sid: int) -> StudentState:
            if sid not in student_states:
                student_states[sid] = StudentState(sid)
            return student_states[sid]

        def get_tracker(sid: int) -> BehaviorTracker:
            if sid not in behavior_trackers:
                behavior_trackers[sid] = BehaviorTracker(session_start_time)
            return behavior_trackers[sid]

        for det in mapped_dets:
            sid = det.get("seat_id")
            if sid is None:
                continue

            is_ghost = det.get("ghost", False)
            px1, py1, px2, py2 = det["person_bbox"]

            # ── Object detections in student zone ─────────────────────────────
            phone_detected = False
            for ph in phones:
                if _is_within_padded_bbox(ph["bbox"], det["person_bbox"], 0.10):
                    phone_detected = True
                    break

            laptop_detected = False
            for lp in laptops:
                if _is_within_padded_bbox(lp["bbox"], det["person_bbox"], 0.10):
                    laptop_detected = True
                    break

            earphone_detected = False
            person_idx = None
            for i, p in enumerate(raw_dets):
                if p.get("id") == det.get("id"):
                    person_idx = i
                    break
            if person_idx is not None:
                for ep in earphones:
                    if ep["person_idx"] == person_idx:
                        earphone_detected = True
                        break

            # ── Ghost Fast-Path ───────────────────────────────────────────────
            if is_ghost:
                st = get_state(sid)
                tr = compute_trend(st.attn_hist)
                x1, y1, x2, y2 = det["bbox"]
                seat_lbl = det.get("seat_label")
                if not seat_lbl and seat_to_grid_label_fn:
                    seat_lbl = seat_to_grid_label_fn(sid)
                elif not seat_lbl:
                    seat_lbl = f"S{sid}"

                processed_faces.append({
                    "id": sid,
                    "seat_label": seat_lbl,
                    "bbox": [x1, y1, x2, y2],
                    "headPose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
                    "attentionScore": st.attn_ema or 0.0,
                    "attentionRaw": st.attn_ema or 0.0,
                    "avgAttention30": st.avg,
                    "attentionFactors": st.last_factors,
                    "attentionBreakdown": st.attention_breakdown(),
                    "emotion": "neutral",
                    "expressions": {},
                    "activity": st.prev_act,
                    "reconciledActivity": st.reconciled_activity()[0],
                    "activityNote": st.reconciled_activity()[1],
                    "cropUsed": "ghost",
                    "trend": tr,
                    "thumbnail": None,
                    "phoneDetected": False,
                    "laptopDetected": False,
                    "earphoneDetected": False,
                    "eyeOpenness": 1.0,
                    "mouthOpen": 0.0,
                    "gridCell": seat_lbl,
                    "faceMatchDue": False,
                    "matchedStudentId": None,
                    "studentName": None,
                    "matchConfidence": 0.0,
                    "ghost": True,
                    "track_id": det.get("track_id"),
                    "geometric_match": det.get("geometric_match", det.get("seat_confidence", 1.0)),
                    "match_type": det.get("match_type", "GHOST"),
                    "status": det.get("status", "GHOST"),
                    "cost": det.get("cost", 0.0),
                    "prev_seat": det.get("prev_seat", seat_lbl),
                })
                continue

            # ── Active Detection Crop & MediaPipe ─────────────────────────────
            bbox_area = (det["bbox"][2] - det["bbox"][0]) * (det["bbox"][3] - det["bbox"][1])
            small_det = frame_area > 0 and (bbox_area / frame_area) < SMALL_BBOX_FRAC

            mp_res = None
            used_crop = "none"
            crop_candidates = det.get("crops", [])
            if not crop_candidates:
                crop_candidates = [{"name": "tight", "bbox": det["bbox"]}]

            if small_det:
                upper = [c for c in crop_candidates if c["name"] == "upper_body"]
                crop_candidates = upper if upper else []

            for crop_info in crop_candidates:
                cx1, cy1, cx2, cy2 = crop_info["bbox"]
                crop = frame[max(0, cy1):max(0, cy2), max(0, cx1):max(0, cx2)]
                if crop.size == 0:
                    continue
                mp_res = analyze_face_crop(crop)
                if mp_res:
                    used_crop = crop_info["name"]
                    break

            # ── Extract signals ───────────────────────────────────────────────
            if mp_res:
                head_pose = mp_res["head_pose"]
                eye_openness = mp_res["eye_openness"]
                mouth_open = mp_res["mouth_open"]
                emotion = mp_res["emotion"]
                blendshapes = mp_res["blendshapes"]
            else:
                head_pose = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
                eye_openness = 1.0
                mouth_open = 0.0
                emotion = "neutral"
                blendshapes = {"neutral": 1.0}

            st = get_state(sid)
            sig = st.extract_signals(
                det["bbox"],
                det["person_bbox"],
                phone_detected=phone_detected,
                eye_openness=eye_openness,
                mouth_open=mouth_open,
                head_pose=head_pose,
            )
            raw = st.compute_attn(sig)
            attn_s = st.smooth(raw)
            prev = st.prev_act
            act = st.classify_act(
                sig, attn_s,
                laptop_detected=laptop_detected,
                earphone_detected=earphone_detected
            )

            if act != prev:
                get_tracker(sid).record(act, time.time())

            # ── Grid cell label ───────────────────────────────────────────────
            grid_cell_label = ""
            if spatial_grid is not None:
                pcx = (px1 + px2) / 2
                pcy = (py1 + py2) / 2
                cell = nearest_grid_cell(spatial_grid, pcx, pcy)
                if cell:
                    grid_cell_label = cell.label
            elif seat_to_grid_label_fn:
                grid_cell_label = seat_to_grid_label_fn(sid)
            else:
                grid_cell_label = det.get("seat_label", f"S{sid}")

            tr = compute_trend(st.attn_hist)
            x1, y1, x2, y2 = det["bbox"]

            # ── Face matching throttle ────────────────────────────────────────
            face_match_counters[sid] = face_match_counters.get(sid, 0) + 1
            face_match_due = face_match_counters[sid] >= FACE_MATCH_INTERVAL
            if face_match_due:
                face_match_counters[sid] = 0

            matched_student_id = None
            matched_student_name = None
            match_confidence = 0.0

            if face_match_due and registered_faces and mp_res:
                cx1, cy1, cx2, cy2 = det["bbox"]
                face_crop = frame[max(0, cy1):max(0, cy2), max(0, cx1):max(0, cx2)]
                enc = face_rec_engine.extract_encoding(face_crop)
                if enc is not None:
                    reg_list = [(s_id, s_enc) for s_id, s_name, s_enc in registered_faces]
                    matched_student_id, match_confidence = face_rec_engine.find_best_match(enc, reg_list)
                    if matched_student_id is not None:
                        for s_id, s_name, _ in registered_faces:
                            if s_id == matched_student_id:
                                matched_student_name = s_name
                                break
                        new_attendance_events.append({
                            "student_id": matched_student_id,
                            "student_name": matched_student_name,
                            "seat_id": sid,
                            "confidence": match_confidence,
                            "timestamp": time.time(),
                        })

            processed_faces.append({
                "id": sid,
                "bbox": [x1, y1, x2, y2],
                "headPose": head_pose,
                "attentionScore": attn_s,
                "attentionRaw": round(raw, 1),
                "avgAttention30": st.avg,
                "attentionFactors": st.last_factors,
                "attentionBreakdown": st.attention_breakdown(),
                "emotion": emotion,
                "expressions": blendshapes,
                "activity": act,
                "reconciledActivity": st.reconciled_activity()[0],
                "activityNote": st.reconciled_activity()[1],
                "cropUsed": used_crop,
                "trend": tr,
                "thumbnail": crop_to_b64(frame, x1, y1, x2, y2),
                "phoneDetected": phone_detected,
                "laptopDetected": laptop_detected,
                "earphoneDetected": earphone_detected,
                "eyeOpenness": eye_openness,
                "gridCell": det.get("seat_label", grid_cell_label),
                "seat_label": det.get("seat_label", grid_cell_label),
                "faceMatchDue": face_match_due,
                "matchedStudentId": matched_student_id,
                "studentName": matched_student_name,
                "matchConfidence": match_confidence,
                "status": det.get("status", "OK"),
                "track_id": det.get("track_id", det.get("id")),
                "geometric_match": det.get("geometric_match", det.get("seat_confidence", 1.0)),
                "seat_confidence": det.get("geometric_match", det.get("seat_confidence", 1.0)),
                "match_type": det.get("match_type", "EXACT"),
                "ghost": det.get("ghost", False),
                "cost": det.get("cost", 0.0),
                "prev_seat": det.get("prev_seat", det.get("seat_label", grid_cell_label)),
                # Decoupled architecture sub-structures
                "topology": det.get("topology", {
                    "seat_label": det.get("seat_label", grid_cell_label),
                    "row": det.get("row"),
                    "col": det.get("col"),
                    "seat_confidence": det.get("geometric_match", 1.0),
                    "geometric_match": det.get("geometric_match", 1.0),
                    "match_type": det.get("match_type", "EXACT"),
                }),
                "state": det.get("state", {
                    "track_id": det.get("track_id", det.get("id")),
                    "status": det.get("status", "OK"),
                    "cost": det.get("cost", 0.0),
                    "ghost": det.get("ghost", False),
                }),
                "identity": det.get("identity", {
                    "student_id": matched_student_id,
                    "student_name": matched_student_name,
                    "confidence": match_confidence,
                }),
            })

        return processed_faces, new_attendance_events
