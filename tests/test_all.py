"""
SmartClass AI — Abrupt Integration Test
Tests all new modules without needing camera or video hardware.
"""
import sys
sys.path.insert(0, 'backend')

print("=== Testing imports ===")

# ── Config ────────────────────────────────────────────────────────────────────
from core.config import (
    YOLO_CONF, SEAT_PROXIMITY_THRES, SMALL_BBOX_FRAC,
    FACE_MATCH_THRESHOLD, FACE_MATCH_INTERVAL, HYSTERESIS_FRAMES
)
assert YOLO_CONF in (0.15, 0.18),      f"YOLO_CONF wrong: {YOLO_CONF}"
assert SEAT_PROXIMITY_THRES == 0.25,   f"PROX wrong: {SEAT_PROXIMITY_THRES}"
assert SMALL_BBOX_FRAC == 0.02,        f"SMALL wrong: {SMALL_BBOX_FRAC}"
assert FACE_MATCH_THRESHOLD == 0.75,   f"MATCH wrong: {FACE_MATCH_THRESHOLD}"
assert FACE_MATCH_INTERVAL == 30,      f"INTERVAL wrong: {FACE_MATCH_INTERVAL}"
assert HYSTERESIS_FRAMES in (3, 5),    f"HYST wrong: {HYSTERESIS_FRAMES}"
print("  [OK] config constants")

# ── Database models ───────────────────────────────────────────────────────────
from core.database import Student, Session, SessionReport, AttendanceRecord
assert hasattr(Student, 'grid_cell'), "Student missing grid_cell"
assert hasattr(Session, 'session_id'), "Session model missing"
assert hasattr(SessionReport, 'report_json'), "SessionReport model missing"
print("  [OK] database models")

# ── SeatStabilizer validation ─────────────────────────────────────────────────
from core.stabilizer import SeatStabilizer
s = SeatStabilizer(max_seats=10)
assert s.max_seats == 10
try:
    SeatStabilizer(max_seats=0)
    raise AssertionError("Should have raised ValueError for max_seats=0")
except ValueError:
    pass
try:
    SeatStabilizer(max_seats=21)
    raise AssertionError("Should have raised ValueError for max_seats=21")
except ValueError:
    pass
print("  [OK] stabilizer max_seats validation")

# ── BehaviorTracker ───────────────────────────────────────────────────────────
from core.behavior_tracker import BehaviorTracker
bt = BehaviorTracker(session_start_time=0.0)
bt.record("studying", 1.0)
bt.record("studying", 2.0)   # same — no new segment
bt.record("phone",    3.0)   # transition
bt.record("phone",    4.0)   # same
tl = bt.finalize(5.0)
assert len(tl) == 2, f"Expected 2 segments, got {len(tl)}"
assert tl[0]["activity"] == "studying"
assert tl[1]["activity"] == "phone"
assert abs(tl[0]["duration"] - 2.0) < 0.01, f"studying duration wrong: {tl[0]['duration']}"
assert abs(tl[1]["duration"] - 2.0) < 0.01, f"phone duration wrong: {tl[1]['duration']}"
# JSON round-trip
import json
j = json.dumps(tl)
tl2 = json.loads(j)
assert tl2[0]["activity"] == "studying"
print("  [OK] behavior_tracker segments + JSON round-trip")

# ── StudentState ──────────────────────────────────────────────────────────────
from core.student_state import StudentState
st = StudentState(seat_id=1)
assert st.phone_frame_count == 0
assert st.laptop_frame_count == 0
assert st.earphone_frame_count == 0
assert st._act_candidate == "neutral"

sig_base = {
    "vel": 0.0, "osc": 0.0, "av": 0.0, "drift": 0.0,
    "turn_dev": 0.0, "slump_dev": 0.0,
    "phone_detected": False, "eye_openness": 1.0,
    "mouth_open": 0.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0
}

# Test 3-frame (or 5-frame) hysteresis: studying should commit after HYSTERESIS_FRAMES frames
results = []
for _ in range(7):
    act = st.classify_act(sig_base, 80.0, laptop_detected=False, earphone_detected=False)
    results.append(act)
# After HYSTERESIS_FRAMES frames of "studying" candidate, prev_act should be "studying"
assert "studying" in results, f"Hysteresis failed — studying never committed: {results}"
print(f"  [OK] student_state hysteresis: {results}")

# Test laptop label
st2 = StudentState(seat_id=2)
for _ in range(4):
    act = st2.classify_act(sig_base, 50.0, laptop_detected=True, earphone_detected=False)
assert st2.laptop_frame_count == 4
print(f"  [OK] student_state laptop counter={st2.laptop_frame_count}, last_act={act}")

# Test phone counter
st3 = StudentState(seat_id=3)
sig_phone = dict(sig_base, phone_detected=True)
for _ in range(3):
    st3.classify_act(sig_phone, 10.0)
assert st3.phone_frame_count == 3
print(f"  [OK] student_state phone counter={st3.phone_frame_count}")

# ── ClassroomMapper ───────────────────────────────────────────────────────────
import numpy as np
from core.classroom_mapper import ClassroomMapper, SpatialGrid, GridCell

mapper = ClassroomMapper()

# None frame raises ValueError
try:
    mapper.detect_layout(None)
    raise AssertionError("Should raise ValueError for None frame")
except ValueError:
    pass

# Empty frame raises ValueError
try:
    mapper.detect_layout(np.zeros((0, 0, 3), dtype=np.uint8))
    raise AssertionError("Should raise ValueError for empty frame")
except ValueError:
    pass

# Fallback grid (no YOLO available in test env)
frame = np.zeros((480, 640, 3), dtype=np.uint8)
grid = mapper.detect_layout(frame, rows_hint=2, cols_hint=3)
assert grid.rows == 2, f"Expected 2 rows, got {grid.rows}"
assert grid.cols == 3, f"Expected 3 cols, got {grid.cols}"
assert len(grid.cells) == 6, f"Expected 6 cells, got {len(grid.cells)}"
labels = [c.label for c in grid.cells]
assert "R1C1" in labels and "R2C3" in labels, f"Labels wrong: {labels}"
print(f"  [OK] classroom_mapper fallback 2x3 grid: {labels}")

# Default fallback (no hints)
grid2 = mapper.detect_layout(frame)
assert len(grid2.cells) == 6  # default 2x3
print(f"  [OK] classroom_mapper default fallback: {len(grid2.cells)} cells")

# _fallback_grid left-to-right ordering
for cell in grid.cells:
    assert cell.label == f"R{cell.row+1}C{cell.col+1}", f"Label mismatch: {cell}"
print("  [OK] classroom_mapper cell labels correct")

# ── _is_within_padded_bbox ────────────────────────────────────────────────────
from core.detector import _is_within_padded_bbox

# Center of inner exactly at center of outer → True
assert _is_within_padded_bbox([45, 45, 55, 55], [0, 0, 100, 100]) == True
# Inner center far outside → False
assert _is_within_padded_bbox([200, 200, 210, 210], [0, 0, 100, 100]) == False
# Inner center just inside padded boundary (10% pad = 10px on 100px box)
assert _is_within_padded_bbox([105, 50, 115, 60], [0, 0, 100, 100]) == True   # cx=110 <= 110
assert _is_within_padded_bbox([116, 50, 126, 60], [0, 0, 100, 100]) == False  # cx=121 > 110
print("  [OK] _is_within_padded_bbox logic")

# ── ReportGenerator ───────────────────────────────────────────────────────────
from core.report_generator import ReportGenerator

rg = ReportGenerator()
bt2 = BehaviorTracker(session_start_time=0.0)
bt2.record("studying", 1.0)
bt2.record("phone", 10.0)

report = rg.generate_student_report(
    student_state=st,
    behavior_tracker=bt2,
    session_id="test-session-uuid",
    session_date="2025-01-01",
    session_duration=60.0,
    frame_skip=3,
    source_fps=25.0,
    student_name="Alice",
    roll_no="R001",
    grid_label="R1C2",
    attendance_status="present",
    student_id=42,
)

# Required fields
required_keys = [
    "student_name", "roll_no", "grid_label", "session_id", "session_date",
    "session_duration_seconds", "avg_attention", "peak_attention", "low_attention",
    "attendance_status", "activity_timeline", "electronics"
]
for k in required_keys:
    assert k in report, f"Missing key: {k}"

assert report["student_name"] == "Alice"
assert report["roll_no"] == "R001"
assert report["grid_label"] == "R1C2"
assert report["attendance_status"] == "present"
assert isinstance(report["activity_timeline"], list)
assert isinstance(report["electronics"], dict)
assert "phone_duration_seconds" in report["electronics"]
assert "laptop_duration_seconds" in report["electronics"]
assert "earphone_duration_seconds" in report["electronics"]

# JSON round-trip
j2 = json.dumps(report)
r2 = json.loads(j2)
for k in ["avg_attention", "peak_attention", "low_attention", "session_duration_seconds"]:
    assert abs(report[k] - r2[k]) < 0.01, f"Round-trip mismatch on {k}"
assert len(r2["activity_timeline"]) == len(report["activity_timeline"])
print("  [OK] report_generator all required fields present")
print("  [OK] report_generator JSON round-trip")

# ── ClassroomEngine session_id ────────────────────────────────────────────────
from core.engine import ClassroomEngine
engine = ClassroomEngine(max_seats=6)
assert len(engine.session_id) == 36, f"session_id not UUID: {engine.session_id}"
assert engine.spatial_grid is None
assert engine.behavior_trackers == {}
old_sid = engine.session_id
engine.reset()
assert engine.session_id != old_sid, "reset() should generate new session_id"
print(f"  [OK] engine session_id={engine.session_id[:8]}... (new after reset)")

# ── Engine max_seats forwarding ───────────────────────────────────────────────
for n in [1, 5, 10, 20]:
    e = ClassroomEngine(max_seats=n)
    assert e.stabilizer.max_seats == n, f"max_seats not forwarded for n={n}"
print("  [OK] engine max_seats forwarding [1,5,10,20]")

print()
print("=" * 50)
print("ALL TESTS PASSED")
print("=" * 50)
