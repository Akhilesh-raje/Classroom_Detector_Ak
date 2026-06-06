"""
SmartClass AI — Report Generator
Builds per-student session reports and persists them to the database.
All report fields use JSON-native types (str, int, float, list, dict, None).
"""
from __future__ import annotations
import json
from datetime import datetime


class ReportGenerator:
    """
    Generates StudentReport dicts from StudentState + BehaviorTracker data
    and persists them to the SessionReport table.
    """

    def generate_student_report(
        self,
        student_state,
        behavior_tracker,
        session_id: str,
        session_date: str,
        session_duration: float,
        frame_skip: int,
        source_fps: float,
        student_name: str = "Unknown",
        roll_no: str | None = None,
        grid_label: str = "",
        attendance_status: str = "absent",
        student_id: int | None = None,
    ) -> dict:
        """
        Build a complete StudentReport dict.

        Args:
            student_state:      StudentState instance for this seat.
            behavior_tracker:   BehaviorTracker instance for this seat.
            session_id:         UUID string of the session.
            session_date:       Date string in YYYY-MM-DD format.
            session_duration:   Total session length in seconds.
            frame_skip:         Frame skip value used during processing.
            source_fps:         Source video/camera FPS.
            student_name:       Registered student name or "Unknown".
            roll_no:            Student roll number or None.
            grid_label:         GridCell label (e.g. "R1C2") or seat ID string.
            attendance_status:  "present" or "absent".
            student_id:         Database student ID or None.

        Returns:
            dict with all required fields, JSON-serialisable without custom encoders.
        """
        fps = max(source_fps, 1.0)  # guard against zero fps

        # Electronics durations
        phone_dur    = round(student_state.phone_frame_count    * frame_skip / fps, 3)
        laptop_dur   = round(student_state.laptop_frame_count   * frame_skip / fps, 3)
        earphone_dur = round(student_state.earphone_frame_count * frame_skip / fps, 3)

        # Attention stats
        avg_attn  = round(float(student_state.avg),  1)
        peak_attn = round(float(student_state.peak), 1)
        low_attn  = round(float(student_state.low),  1)

        # Activity timeline
        timeline = behavior_tracker.finalize(session_duration)

        return {
            "student_name":             str(student_name),
            "roll_no":                  str(roll_no) if roll_no is not None else None,
            "grid_label":               str(grid_label),
            "session_id":               str(session_id),
            "session_date":             str(session_date),
            "session_duration_seconds": round(float(session_duration), 3),
            "avg_attention":            avg_attn,
            "peak_attention":           peak_attn,
            "low_attention":            low_attn,
            "attention_methodology": (
                "Observable engagement proxy. Weights: head_pose=65%, "
                "motion_stability=27%, eye_openness=8%. "
                "Not a measure of cognitive comprehension."
            ),
            "confidence":               round(min(1.0, len(student_state.attn_hist) / 30), 2),
            "frames_analyzed":          len(student_state.attn_hist),
            "sample_confidence":        round(min(1.0, len(student_state.attn_hist) / 90), 2),
            "sample_confidence_note": (
                "Low confidence \u2014 fewer than 90 frames analyzed"
                if len(student_state.attn_hist) < 90 else
                "Good sample size"
            ),
            "attendance_status":        str(attendance_status),
            "activity_timeline":        timeline,
            "electronics": {
                "phone_duration_seconds":    phone_dur,
                "laptop_duration_seconds":   laptop_dur,
                "earphone_duration_seconds": earphone_dur,
            },
            # Internal — used for DB persistence, not exposed in API response
            "_student_id": student_id,
        }

    def persist_reports(
        self,
        reports: list[dict],
        session_id: str,
        db_session,
    ) -> None:
        """
        Persist a list of StudentReport dicts to the SessionReport table.

        Args:
            reports:    List of report dicts from generate_student_report().
            session_id: UUID string of the session.
            db_session: SQLAlchemy database session.
        """
        from core.database import SessionReport

        for report in reports:
            student_id = report.get("_student_id")
            # Strip internal key before storing
            clean = {k: v for k, v in report.items() if not k.startswith("_")}
            row = SessionReport(
                session_id=session_id,
                student_id=student_id,
                report_json=json.dumps(clean),
                created_at=datetime.utcnow(),
            )
            db_session.add(row)

        db_session.commit()
