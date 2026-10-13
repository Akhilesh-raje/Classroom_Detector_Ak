"""
SmartClass AI — Attendance Decision Engine v1.1
Architecture:
1. Time-Based Temporal Confirmation Window + Track Continuity
   (Requires duration >= 1.5s AND observations >= 5; resets if track gap > 1.5s)
2. Identity Resolution Tiers & Biometric Distance Metric Calibration
   - Tier 1: Verified Biometric (Cosine Similarity >= 0.70 / Distance <= 0.30 with reliable backend)
   - Tier 2: Seat-Based Candidate (Occupant mapped to desk roster or borderline biometric)
   - Tier 3: Unknown Occupant (Nullable student_id, explicit UNKNOWN_OCCUPANT state)
3. Strict Idempotency & Thread-Safe Concurrency (no duplicates under parallel requests)
4. Dry-Run Isolation (is_dry_run=1 never touches official student.total_attendance)
5. Audit Trail & Seat Dynamics (Seat changes tracked; conflicts flagged for teacher review)
"""

import json
import logging
import threading
from datetime import datetime, date
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from core.database import Student, AttendanceRecord, Session as SessionModel
from core.face_recognition_engine import face_rec_engine

log = logging.getLogger("attendance_engine")


class AttendanceDecisionEngine:
    # Biometric calibration constants (InsightFace / Buffalo_SC 512D Cosine embeddings)
    # Cosine Similarity = 1 - Cosine Distance
    # Threshold 0.70 similarity = 0.30 distance (Strict Verification)
    # Threshold 0.58 similarity = 0.42 distance (Candidate Evidence)
    MIN_BIOMETRIC_CONFIDENCE = 0.70
    CANDIDATE_BIOMETRIC_CONFIDENCE = 0.58

    def __init__(
        self,
        confirmation_duration_sec: float = 1.5,
        min_observations: int = 5,
        max_gap_sec: float = 1.5,
        min_confidence: float = 0.70
    ):
        self.confirmation_duration_sec = confirmation_duration_sec
        self.min_observations = min_observations
        self.max_gap_sec = max_gap_sec
        self.min_confidence = min_confidence
        
        # Thread lock for concurrent request race conditions
        self._lock = threading.Lock()

        # In-memory temporal presence tracker:
        # key: str -> dict(observation_count=int, first_seen=datetime, last_seen=datetime, ...)
        self._presence_window: Dict[str, Dict[str, Any]] = {}
        
        # Active session state
        self.active_session_id: Optional[str] = None
        self.dry_run: bool = False

    def set_session(self, session_id: Optional[str]):
        self.active_session_id = session_id
        self.dry_run = session_id is None

    def reset_window(self):
        """Clears the in-memory temporal confirmation window."""
        with self._lock:
            self._presence_window.clear()

    def process_frame_observations(
        self,
        observations: List[Dict[str, Any]],
        db: Session,
        dry_run: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Processes a batch of detections from a real frame or simulated stream.
        Applies time-based temporal confirmation, track continuity,
        3-tier identity resolution, and idempotent database persistence.
        """
        with self._lock:
            effective_dry_run = self.dry_run if dry_run is None else dry_run
            today = date.today().isoformat()
            now_default = datetime.utcnow()
            results = []

            # Pre-fetch enrolled students mapped by seat and ID
            all_students = db.query(Student).all()
            student_by_seat = {s.grid_cell: s for s in all_students if s.grid_cell}
            student_by_id = {s.id: s for s in all_students}

            for obs in observations:
                # Support simulated timestamps for reproducible testing of variable FPS / elapsed time
                if "timestamp" in obs:
                    ts_val = obs["timestamp"]
                    if isinstance(ts_val, (int, float)):
                        now = datetime.utcfromtimestamp(ts_val)
                    elif isinstance(ts_val, str):
                        now = datetime.fromisoformat(ts_val)
                    else:
                        now = ts_val
                else:
                    now = now_default

                seat_id = obs.get("seat_id") or obs.get("grid_cell") or obs.get("gridCell") or ""
                track_id = str(obs.get("track_id") or obs.get("id") or seat_id or "unknown")
                matched_id = obs.get("matched_student_id") or obs.get("matchedStudentId")
                face_conf = float(obs.get("match_confidence") or obs.get("confidence") or 0.0)
                thumbnail = obs.get("thumbnail")
                is_calibrated = obs.get("is_calibrated", True)

                # Composite tracking key (seat + track)
                tracker_key = f"{seat_id}_{track_id}" if seat_id else track_id
                
                # ── 1. Temporal Presence Tracking with Track Continuity ───────
                if tracker_key not in self._presence_window:
                    self._presence_window[tracker_key] = {
                        "observation_count": 1,
                        "first_seen": now,
                        "last_seen": now,
                        "seat_id": seat_id,
                        "matched_ids": [matched_id] if matched_id else [],
                        "confidence_history": [face_conf],
                    }
                else:
                    track = self._presence_window[tracker_key]
                    gap_sec = (now - track["last_seen"]).total_seconds()
                    
                    if gap_sec > self.max_gap_sec:
                        # Track was lost and reappeared after a significant gap -> reset window
                        track["observation_count"] = 1
                        track["first_seen"] = now
                        track["matched_ids"] = [matched_id] if matched_id else []
                        track["confidence_history"] = [face_conf]
                    else:
                        # Continuous track
                        track["observation_count"] += 1
                        if matched_id:
                            track["matched_ids"].append(matched_id)
                        track["confidence_history"].append(face_conf)

                    track["last_seen"] = now

                track = self._presence_window[tracker_key]
                obs_count = track["observation_count"]
                elapsed_sec = max(0.0, (now - track["first_seen"]).total_seconds())

                # Check time-based duration AND observation count
                time_ok = (elapsed_sec >= self.confirmation_duration_sec) or obs.get("force_time_confirm", False)
                count_ok = (obs_count >= self.min_observations) or obs.get("force_time_confirm", False)

                if not (time_ok and count_ok):
                    results.append({
                        "track_id": track_id,
                        "seat_id": seat_id,
                        "state": "DETECTED",
                        "status": "PENDING_CONFIRMATION",
                        "observation_count": obs_count,
                        "min_observations": self.min_observations,
                        "elapsed_seconds": round(elapsed_sec, 2),
                        "duration_threshold_sec": self.confirmation_duration_sec,
                        "message": f"Temporal window: {obs_count}/{self.min_observations} obs, {elapsed_sec:.2f}/{self.confirmation_duration_sec:.2f}s"
                    })
                    continue

                # ── 2. Three-Tier Identity Resolution ────────────────────────
                decision_tier = None
                resolved_student_id = None
                candidate_student_id = None
                presence_state = "UNKNOWN_OCCUPANT"
                status = "PENDING_REVIEW"
                method = "unresolved_occupant"

                # Check backend reliability
                backend_reliable = face_rec_engine.is_reliable or is_calibrated

                # Tier 1: Verified Biometric Identity
                if matched_id and face_conf >= self.min_confidence and matched_id in student_by_id:
                    if not backend_reliable:
                        # Fallback backend cannot be trusted for autonomous Tier 1 verification
                        decision_tier = "TIER_2_CANDIDATE_UNRELIABLE_BIOMETRICS"
                        candidate_student_id = matched_id
                        presence_state = "CANDIDATE"
                        status = "PENDING_REVIEW"
                        method = "unreliable_biometrics_fallback"
                    else:
                        decision_tier = "TIER_1_VERIFIED_BIOMETRIC"
                        resolved_student_id = matched_id
                        presence_state = "IDENTIFIED"
                        status = "VERIFIED"
                        method = "biometric_verified"

                # Tier 2: Seat-Based Candidate Evidence OR Borderline Biometric
                elif matched_id and face_conf >= self.CANDIDATE_BIOMETRIC_CONFIDENCE and matched_id in student_by_id:
                    decision_tier = "TIER_2_BORDERLINE_BIOMETRIC"
                    candidate_student_id = matched_id
                    presence_state = "CANDIDATE"
                    status = "PENDING_REVIEW"
                    method = "borderline_biometric_candidate"

                elif seat_id and seat_id in student_by_seat:
                    decision_tier = "TIER_2_SEAT_CANDIDATE"
                    candidate = student_by_seat[seat_id]
                    candidate_student_id = candidate.id
                    presence_state = "CANDIDATE"
                    status = "PENDING_REVIEW"
                    method = "seat_candidate"

                # Tier 3: Unknown Occupant (Nullable student_id)
                else:
                    decision_tier = "TIER_3_UNKNOWN_OCCUPANT"
                    presence_state = "UNKNOWN_OCCUPANT"
                    status = "PENDING_REVIEW"
                    method = "unresolved_occupant"

                # ── 3. Idempotent Attendance Persistence & Seat Dynamics ─────
                audit_log = {
                    "decision_tier": decision_tier,
                    "observation_count": obs_count,
                    "elapsed_sec": round(elapsed_sec, 2),
                    "face_confidence": face_conf,
                    "seat_id": seat_id,
                    "track_id": track_id,
                    "timestamp": now.isoformat(),
                    "is_dry_run": effective_dry_run,
                    "history": [f"Confirmed at {now.isoformat()} (Tier: {decision_tier})"]
                }

                # Check if this verified student already has a record today
                existing_record = None
                if resolved_student_id is not None:
                    query = db.query(AttendanceRecord).filter(
                        AttendanceRecord.student_id == resolved_student_id,
                        AttendanceRecord.date == today,
                        AttendanceRecord.is_dry_run == (1 if effective_dry_run else 0)
                    )
                    if self.active_session_id:
                        query = query.filter(AttendanceRecord.session_id == self.active_session_id)
                    existing_record = query.first()

                # For candidate / unknown occupant, check if this seat already has an unverified record
                if not existing_record and seat_id:
                    existing_record = db.query(AttendanceRecord).filter(
                        AttendanceRecord.seat_id == seat_id,
                        AttendanceRecord.date == today,
                        AttendanceRecord.status == status,
                        AttendanceRecord.is_dry_run == (1 if effective_dry_run else 0)
                    ).first()

                if not existing_record:
                    # Create attendance record (student_id is NULL for unknown occupants)
                    new_record = AttendanceRecord(
                        student_id=resolved_student_id,
                        date=today,
                        marked_at=now,
                        confidence=face_conf if decision_tier == "TIER_1_VERIFIED_BIOMETRIC" else (0.85 if candidate_student_id else 0.50),
                        captured_image=thumbnail,
                        method=method,
                        session_id=self.active_session_id,
                        seat_id=seat_id,
                        status=status,
                        presence_state=presence_state,
                        detection_count=obs_count,
                        candidate_student_id=candidate_student_id,
                        audit_trail=json.dumps(audit_log),
                        is_dry_run=1 if effective_dry_run else 0
                    )
                    db.add(new_record)
                    
                    # Strictly only update official aggregate attendance if verified AND NOT a dry run
                    if resolved_student_id is not None and not effective_dry_run:
                        stud = student_by_id[resolved_student_id]
                        stud.total_attendance = (stud.total_attendance or 0) + 1
                        
                    db.commit()
                    db.refresh(new_record)
                    record_id = new_record.id
                    action_taken = "CREATED"
                else:
                    # Idempotent update: update counter, check seat migration
                    existing_record.detection_count = obs_count
                    
                    # Detect seat reassignment without silent identity substitution
                    if seat_id and existing_record.seat_id != seat_id:
                        try:
                            trail = json.loads(existing_record.audit_trail) if existing_record.audit_trail else {}
                        except Exception:
                            trail = {}
                        history = trail.get("history", [])
                        history.append(f"Seat changed from {existing_record.seat_id} to {seat_id} at {now.isoformat()}")
                        trail["history"] = history
                        trail["previous_seat"] = existing_record.seat_id
                        existing_record.seat_id = seat_id
                        existing_record.audit_trail = json.dumps(trail)
                        action_taken = "SEAT_UPDATED"
                    else:
                        action_taken = "EXISTING_IDEMPOTENT"
                        
                    db.commit()
                    record_id = existing_record.id

                results.append({
                    "record_id": record_id,
                    "track_id": track_id,
                    "seat_id": seat_id,
                    "student_id": resolved_student_id,
                    "candidate_student_id": candidate_student_id,
                    "student_name": student_by_id[resolved_student_id].name if resolved_student_id else None,
                    "candidate_name": student_by_id[candidate_student_id].name if candidate_student_id else None,
                    "presence_state": presence_state,
                    "status": status,
                    "decision_tier": decision_tier,
                    "action": action_taken,
                    "is_dry_run": effective_dry_run,
                    "marked_at": now.isoformat(),
                })

            return results

    def get_attendance_summary(self, db: Session, target_date: Optional[str] = None) -> Dict[str, Any]:
        """Returns verified, review, and detected counts matching the UI architecture."""
        today = target_date or date.today().isoformat()
        
        all_students = db.query(Student).all()
        records = db.query(AttendanceRecord).filter(AttendanceRecord.date == today).all()

        # Separate by status and tier
        verified_records = [r for r in records if r.status == "VERIFIED" and r.student_id is not None]
        review_records   = [r for r in records if r.status == "PENDING_REVIEW"]
        verified_ids     = {r.student_id for r in verified_records if r.student_id is not None}

        # Build full student records
        student_rows = []
        for s in all_students:
            rec = next((r for r in verified_records if r.student_id == s.id), None)
            cand_rec = next((r for r in review_records if r.candidate_student_id == s.id), None)
            
            if rec:
                current_status = "VERIFIED"
                record_info = rec
            elif cand_rec:
                current_status = "PENDING_REVIEW"
                record_info = cand_rec
            else:
                current_status = "ABSENT"
                record_info = None

            student_rows.append({
                "studentId": s.id,
                "name": s.name,
                "rollNo": s.roll_no,
                "section": s.section,
                "assignedSeat": s.grid_cell,
                "detectedSeat": record_info.seat_id if record_info else None,
                "status": current_status,
                "markedAt": record_info.marked_at.isoformat() if record_info else None,
                "confidence": record_info.confidence if record_info else None,
                "method": record_info.method if record_info else None,
                "isDryRun": bool(record_info.is_dry_run) if record_info else False,
                "faceImage": s.face_image,
            })

        # Recent events for ticker / timeline
        student_name_map = {s.id: s.name for s in all_students}
        recent_events = []
        for r in sorted(records, key=lambda x: x.marked_at or datetime.min, reverse=True)[:15]:
            stud_name = None
            if r.student_id and r.student_id in student_name_map:
                stud_name = student_name_map[r.student_id]
            elif r.candidate_student_id and r.candidate_student_id in student_name_map:
                stud_name = f"Candidate: {student_name_map[r.candidate_student_id]}"

            recent_events.append({
                "id": r.id,
                "time": r.marked_at.strftime("%H:%M:%S") if r.marked_at else "",
                "seat": r.seat_id or "—",
                "studentName": stud_name or "Unknown Occupant",
                "status": r.status,
                "presenceState": r.presence_state,
                "method": r.method,
                "confidence": r.confidence,
                "isDryRun": bool(r.is_dry_run)
            })

        return {
            "date": today,
            "counts": {
                "detected": len(records),
                "verified": len(verified_records),
                "review": len(review_records),
                "absent": max(0, len(all_students) - len(verified_ids)),
                "totalEnrolled": len(all_students),
            },
            "records": student_rows,
            "reviewQueue": [
                {
                    "id": r.id,
                    "seatId": r.seat_id,
                    "presenceState": r.presence_state,
                    "confidence": r.confidence,
                    "detectedAt": r.marked_at.isoformat() if r.marked_at else None,
                    "candidateStudentId": r.candidate_student_id,
                    "candidateName": next((s.name for s in all_students if s.id == r.candidate_student_id), "None"),
                    "auditTrail": json.loads(r.audit_trail) if r.audit_trail else {},
                }
                for r in review_records
            ],
            "recentEvents": recent_events,
            "dryRunMode": self.dry_run
        }


# Singleton instance
attendance_engine = AttendanceDecisionEngine()
