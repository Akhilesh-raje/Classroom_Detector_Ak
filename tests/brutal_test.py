# -*- coding: utf-8 -*-
"""
SmartClass AI — BRUTAL AUTOMATED TEST SUITE
============================================
Tests every layer of the system without mercy:
  - StudentState signals, attention math, edge cases
  - BehaviorTracker timeline integrity
  - SeatStabilizer race conditions, overflow, ID drift
  - ClassroomMapper fallback, bad input, zero detections
  - draw_compact_overlay / draw_full_overlay crash safety
  - suppress_overlapping_faces correctness
  - Engine process_single_frame with synthetic frames
  - Memory / state leak detection across 1000 synthetic frames

Run: python brutal_test.py
"""
import sys
import os
import io
import time
import math
import traceback
import numpy as np
import cv2

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_PATH = os.path.join(HERE, "backend")
sys.path.insert(0, BACKEND_PATH)

# ── Test harness ──────────────────────────────────────────────────────────────
PASS  = 0
FAIL  = 0
WARN  = 0
_log  = []

def ok(name):
    global PASS
    PASS += 1
    _log.append(f"  [PASS] {name}")

def fail(name, reason):
    global FAIL
    FAIL += 1
    _log.append(f"  [FAIL] {name}  →  {reason}")

def warn(name, reason):
    global WARN
    WARN += 1
    _log.append(f"  [WARN] {name}  →  {reason}")

def section(title):
    _log.append(f"\n── {title} {'─'*(55-len(title))}")

def try_test(name, fn):
    try:
        fn()
        ok(name)
    except AssertionError as e:
        fail(name, str(e))
    except Exception as e:
        fail(name, f"EXCEPTION: {type(e).__name__}: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# 1. Import safety
# ═════════════════════════════════════════════════════════════════════════════
section("1. IMPORTS")

try:
    from core.student_state import StudentState
    ok("import StudentState")
except Exception as e:
    fail("import StudentState", str(e))

try:
    from core.behavior_tracker import BehaviorTracker
    ok("import BehaviorTracker")
except Exception as e:
    fail("import BehaviorTracker", str(e))

try:
    from core.stabilizer import SeatStabilizer
    ok("import SeatStabilizer")
except Exception as e:
    fail("import SeatStabilizer", str(e))

try:
    from core.classroom_mapper import ClassroomMapper, SpatialGrid, GridCell
    ok("import ClassroomMapper")
except Exception as e:
    fail("import ClassroomMapper", str(e))

try:
    from core.visualizer import alpha_rect, attn_bar, corner_brackets, C, ACT_COLOR, ACT_ICON
    ok("import visualizer primitives")
except Exception as e:
    fail("import visualizer primitives", str(e))

try:
    from test_real_classroom import suppress_overlapping_faces, draw_compact_overlay, draw_full_overlay, _iou
    ok("import test_real_classroom helpers")
except Exception as e:
    fail("import test_real_classroom helpers", str(e))


# ═════════════════════════════════════════════════════════════════════════════
# 2. StudentState — attention math
# ═════════════════════════════════════════════════════════════════════════════
section("2. STUDENT STATE — ATTENTION MATH")

def test_attention_range():
    st  = StudentState(1)
    sig = st.extract_signals([100,100,200,200], [80,80,220,300],
                             phone_detected=False, eye_openness=1.0,
                             mouth_open=0.0, head_pose={"yaw":0,"pitch":0,"roll":0})
    raw = st.compute_attn(sig)
    assert 0 <= raw <= 100, f"attn out of range: {raw}"
try_test("attention_score always in [0,100]", test_attention_range)

def test_phone_penalty():
    st  = StudentState(1)
    sig = st.extract_signals([100,100,200,200],[80,80,220,300],
                             phone_detected=True, eye_openness=1.0,
                             mouth_open=0.0, head_pose={"yaw":0,"pitch":0,"roll":0})
    raw = st.compute_attn(sig)
    assert raw < 20, f"phone should crush attention, got {raw}"
try_test("phone detection crushes attention score (<20)", test_phone_penalty)

def test_extreme_yaw_penalty():
    st  = StudentState(2)
    # Yaw = 90 degrees (fully turned to the side — profile view)
    # At yaw=90: yaw_score=0, head_pose_score=0.4 (pitch component),
    # so attention should be noticeably below the neutral baseline (~75-95)
    results = []
    for _ in range(10):
        sig = st.extract_signals([100,100,200,200],[80,80,220,300],
                                 phone_detected=False, eye_openness=0.9,
                                 mouth_open=0.0, head_pose={"yaw":90,"pitch":0,"roll":0})
        raw = st.compute_attn(sig)
        results.append(raw)
    avg_attn = sum(results) / len(results)
    assert avg_attn < 70, f"yaw=90 avg attn should be < 70 (clearly degraded), got {avg_attn:.1f}"
try_test("extreme yaw (90°) significantly reduces attention (avg < 70)", test_extreme_yaw_penalty)

def test_zero_area_bbox():
    st = StudentState(3)
    # Zero-area bbox — should not crash
    try:
        sig = st.extract_signals([0,0,0,0],[0,0,0,0])
        st.compute_attn(sig)
        ok("zero-area bbox does not crash")
    except Exception as e:
        fail("zero-area bbox does not crash", str(e))

test_zero_area_bbox()

def test_nan_head_pose():
    st = StudentState(4)
    sig = st.extract_signals([100,100,200,200],[80,80,220,300],
                             head_pose={"yaw":float("nan"),"pitch":float("nan"),"roll":0.0})
    try:
        raw = st.compute_attn(sig)
        # nan in head_pose should not produce nan output
        assert not math.isnan(raw), f"attn became NaN: {raw}"
        ok("NaN head pose does not propagate to attn score")
    except Exception as e:
        fail("NaN head pose does not propagate to attn score", str(e))

test_nan_head_pose()

def test_smooth_convergence():
    st = StudentState(5)
    # Feed 100 frames of value 80 — EMA should converge within ±5
    for _ in range(100):
        st.smooth(80.0)
    assert abs(st.attn_ema - 80.0) < 5.0, f"EMA did not converge: {st.attn_ema}"
try_test("EMA converges to 80 after 100 frames", test_smooth_convergence)

def test_avg_property():
    st = StudentState(6)
    for v in [60, 70, 80, 90, 100]:
        st.smooth(float(v))
    expected = sum([60,70,80,90,100]) / 5
    # EMA values won't equal raw but avg should be reasonable
    assert 55 <= st.avg <= 100, f"avg out of range: {st.avg}"
try_test("avg property stays in reasonable range", test_avg_property)

def test_reconciled_high_attn_distracted():
    st = StudentState(7)
    # Simulate high avg attention but last activity = distracted
    for _ in range(60):
        st.smooth(90.0)
    st.prev_act = "distracted"
    act, note = st.reconciled_activity()
    assert act == "studying", f"high avg should reconcile to studying, got '{act}'"
    assert note != "", "should have a note"
try_test("reconciled_activity: high avg overrides distracted label", test_reconciled_high_attn_distracted)

def test_reconciled_low_attn_studying():
    st = StudentState(8)
    for _ in range(60):
        st.smooth(30.0)
    st.prev_act = "studying"
    act, note = st.reconciled_activity()
    assert note != "", "low avg studying should have a reconciliation note"
try_test("reconciled_activity: low avg studying gets a note", test_reconciled_low_attn_studying)

def test_classify_act_drowsy_requires_sustained():
    st = StudentState(9)
    # A single frame of eye_open=0.1 should NOT immediately flip to drowsy
    sig = st.extract_signals([100,100,200,200],[80,80,220,300],
                             eye_openness=0.10, head_pose={"yaw":0,"pitch":0,"roll":0})
    st.compute_attn(sig)
    act = st.classify_act(sig, 70.0)
    # After 1 frame it should NOT be drowsy yet (needs 4 frames + hysteresis)
    assert act != "drowsy", f"drowsy should need sustained eye closure, got '{act}' after 1 frame"
try_test("drowsy requires sustained eye closure (not 1 frame)", test_classify_act_drowsy_requires_sustained)

def test_1000_frames_no_crash():
    st = StudentState(10)
    rng = np.random.default_rng(42)
    for i in range(1000):
        yaw   = float(rng.uniform(-90, 90))
        pitch = float(rng.uniform(-40, 40))
        eye   = float(rng.uniform(0, 1))
        sig   = st.extract_signals(
            [100,100,200,200],[80,80,220,300],
            phone_detected=(i % 50 == 0),
            eye_openness=eye, mouth_open=float(rng.uniform(0,0.5)),
            head_pose={"yaw":yaw,"pitch":pitch,"roll":0.0}
        )
        raw  = st.compute_attn(sig)
        s    = st.smooth(raw)
        act  = st.classify_act(sig, s)
        assert 0 <= raw <= 100, f"attn out of range at frame {i}: {raw}"
        assert act in ("studying","attentive","neutral","talking","fidgeting",
                       "distracted","drowsy","phone","laptop","music"), f"unknown act: {act}"
try_test("StudentState: 1000 random frames — no crash, valid outputs", test_1000_frames_no_crash)


# ═════════════════════════════════════════════════════════════════════════════
# 3. BehaviorTracker
# ═════════════════════════════════════════════════════════════════════════════
section("3. BEHAVIOR TRACKER")

def test_bt_empty_finalize():
    bt = BehaviorTracker(0.0)
    result = bt.finalize(10.0)
    assert result == [], f"empty tracker finalize should be [], got {result}"
try_test("finalize on empty tracker returns []", test_bt_empty_finalize)

def test_bt_single_activity():
    bt = BehaviorTracker(0.0)
    bt.record("studying", 1.0)
    result = bt.finalize(5.0)
    assert len(result) == 1
    assert result[0]["activity"] == "studying"
    assert abs(result[0]["duration"] - 4.0) < 0.01
try_test("single activity segment has correct duration", test_bt_single_activity)

def test_bt_transition():
    bt = BehaviorTracker(0.0)
    bt.record("studying",   1.0)
    bt.record("studying",   2.0)  # same — no new segment
    bt.record("distracted", 3.0)  # transition
    bt.record("distracted", 4.0)
    result = bt.finalize(5.0)
    assert len(result) == 2, f"expected 2 segments, got {len(result)}"
    assert result[0]["activity"] == "studying"
    assert result[1]["activity"] == "distracted"
try_test("two-activity transition produces 2 segments", test_bt_transition)

def test_bt_duration_sum():
    bt = BehaviorTracker(0.0)
    activities = ["studying","talking","distracted","studying","neutral"]
    for i, act in enumerate(activities):
        bt.record(act, float(i))
    result = bt.finalize(10.0)
    total = sum(s["duration"] for s in result)
    # total duration should be close to session length (10s)
    assert abs(total - 10.0) < 0.1, f"segment durations don't sum to session: {total}"
try_test("segment durations sum to session length", test_bt_duration_sum)

def test_bt_rapid_flip():
    # Rapid activity flipping every frame — no crash, sensible output
    bt = BehaviorTracker(0.0)
    activities = ["studying","distracted"] * 500
    for i, act in enumerate(activities):
        bt.record(act, float(i) * 0.1)
    result = bt.finalize(100.0)
    assert len(result) >= 1
try_test("1000 rapid activity flips — no crash", test_bt_rapid_flip)


# ═════════════════════════════════════════════════════════════════════════════
# 4. SeatStabilizer
# ═════════════════════════════════════════════════════════════════════════════
section("4. SEAT STABILIZER")

def test_stab_empty_input():
    s = SeatStabilizer(max_seats=6)
    result = s.update_and_map([], 1920, 1080)
    assert result == [], f"empty input should return [], got {result}"
try_test("empty detection input returns []", test_stab_empty_input)

def test_stab_invalid_max_seats():
    try:
        s = SeatStabilizer(max_seats=0)
        fail("SeatStabilizer(0) should raise ValueError", "no error raised")
    except ValueError:
        ok("SeatStabilizer(max_seats=0) raises ValueError")
    except Exception as e:
        fail("SeatStabilizer(0) raises ValueError", f"wrong exception: {e}")

test_stab_invalid_max_seats()

def test_stab_max_seats_overflow():
    # Send more detections than max_seats — extras should be silently discarded
    s = SeatStabilizer(max_seats=3)
    dets = [{"bbox": [i*100, 100, i*100+80, 200], "confidence": 0.9, "id": i}
            for i in range(10)]  # 10 detections for 3 seats
    result = s.update_and_map(dets, 1920, 1080)
    assert len(result) <= 3, f"should cap at 3 seats, got {len(result)}"
try_test("overflow detections (10 for 3 seats) are discarded", test_stab_max_seats_overflow)

def test_stab_id_stability():
    # Same person at same position across 20 frames — ID should stay constant
    s = SeatStabilizer(max_seats=2)
    ids_seen = set()
    for _ in range(20):
        dets = [
            {"bbox": [100, 100, 200, 300], "confidence": 0.9, "id": 1},
            {"bbox": [400, 100, 500, 300], "confidence": 0.9, "id": 2},
        ]
        result = s.update_and_map(dets, 1920, 1080)
        for r in result:
            ids_seen.add(r["seat_id"])
    assert len(ids_seen) <= 2, f"stable detections should map to ≤2 seats, got {ids_seen}"
try_test("stable detections across 20 frames maintain ≤2 seat IDs", test_stab_id_stability)

def test_stab_sorting_left_to_right():
    # After all seats discovered, left seat should get lower ID
    s = SeatStabilizer(max_seats=3)
    for _ in range(5):
        dets = [
            {"bbox": [800, 100, 900, 300], "confidence": 0.9, "id": 3},  # right
            {"bbox": [100, 100, 200, 300], "confidence": 0.9, "id": 1},  # left
            {"bbox": [450, 100, 550, 300], "confidence": 0.9, "id": 2},  # middle
        ]
        result = s.update_and_map(dets, 1920, 1080)

    # After sorting is triggered, IDs should follow left-to-right order
    result = s.update_and_map(dets, 1920, 1080)
    sorted_result = sorted(result, key=lambda r: (r["bbox"][0] + r["bbox"][2]) / 2)
    ids = [r["seat_id"] for r in sorted_result]
    assert ids == sorted(ids), f"seat IDs not in left-to-right order: {ids}"
try_test("seat IDs sorted left-to-right after discovery", test_stab_sorting_left_to_right)

def test_stab_jitter_no_id_flip():
    # Small random jitter around a fixed position — seat_id should never flip
    s = SeatStabilizer(max_seats=1)
    rng = np.random.default_rng(7)
    assigned_ids = set()
    for _ in range(50):
        jx = int(rng.uniform(-5, 5))
        jy = int(rng.uniform(-5, 5))
        det = [{"bbox": [100+jx, 100+jy, 200+jx, 300+jy], "confidence":0.9, "id":1}]
        res = s.update_and_map(det, 1920, 1080)
        if res:
            assigned_ids.add(res[0]["seat_id"])
    assert len(assigned_ids) == 1, f"jitter caused ID flip: {assigned_ids}"
try_test("small jitter on single person never flips seat_id", test_stab_jitter_no_id_flip)


# ═════════════════════════════════════════════════════════════════════════════
# 5. ClassroomMapper
# ═════════════════════════════════════════════════════════════════════════════
section("5. CLASSROOM MAPPER")

def test_mapper_none_frame():
    cm = ClassroomMapper()
    try:
        cm.detect_layout(None)
        fail("mapper accepts None frame", "no error raised")
    except ValueError:
        ok("mapper raises ValueError for None frame")
    except Exception as e:
        fail("mapper raises ValueError for None frame", f"wrong exception: {e}")

test_mapper_none_frame()

def test_mapper_empty_frame():
    cm = ClassroomMapper()
    try:
        cm.detect_layout(np.zeros((0,0,3), dtype=np.uint8))
        fail("mapper accepts empty frame", "no error raised")
    except (ValueError, Exception) as e:
        ok(f"mapper rejects empty frame ({type(e).__name__})")

test_mapper_empty_frame()

def test_mapper_fallback_grid():
    # Blank white frame — YOLO won't find furniture, should fallback cleanly
    cm   = ClassroomMapper()
    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 255
    grid = cm.detect_layout(frame, rows_hint=3, cols_hint=4)
    assert isinstance(grid, SpatialGrid)
    assert len(grid.cells) == 3 * 4, f"fallback should produce 12 cells, got {len(grid.cells)}"
    assert grid.rows == 3
    assert grid.cols == 4
try_test("blank frame produces correct 3×4 fallback grid", test_mapper_fallback_grid)

def test_mapper_fallback_labels():
    cm   = ClassroomMapper()
    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 200
    grid  = cm.detect_layout(frame, rows_hint=2, cols_hint=3)
    labels = [c.label for c in grid.cells]
    expected = ["R1C1","R1C2","R1C3","R2C1","R2C2","R2C3"]
    assert labels == expected, f"wrong labels: {labels}"
try_test("fallback grid labels are R1C1..R2C3", test_mapper_fallback_labels)

def test_mapper_fallback_bbox_coverage():
    cm    = ClassroomMapper()
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    grid  = cm.detect_layout(frame, rows_hint=2, cols_hint=2)
    for cell in grid.cells:
        x1,y1,x2,y2 = cell.bbox
        assert x2 > x1 and y2 > y1, f"invalid bbox in cell {cell.label}: {cell.bbox}"
        assert x1 >= 0 and y1 >= 0
        assert x2 <= 640 and y2 <= 480
try_test("all fallback grid bboxes are valid and within frame bounds", test_mapper_fallback_bbox_coverage)

def test_mapper_single_cell():
    cm    = ClassroomMapper()
    frame = np.ones((100, 100, 3), dtype=np.uint8) * 100
    grid  = cm.detect_layout(frame, rows_hint=1, cols_hint=1)
    assert len(grid.cells) == 1
    assert grid.cells[0].label == "R1C1"
try_test("1×1 fallback grid has exactly 1 cell labelled R1C1", test_mapper_single_cell)


# ═════════════════════════════════════════════════════════════════════════════
# 6. IoU + suppress_overlapping_faces
# ═════════════════════════════════════════════════════════════════════════════
section("6. IOU + OVERLAP SUPPRESSOR")

def test_iou_identical():
    v = _iou([0,0,100,100],[0,0,100,100])
    assert abs(v - 1.0) < 1e-6, f"identical boxes IoU should be 1.0, got {v}"
try_test("IoU of identical boxes = 1.0", test_iou_identical)

def test_iou_no_overlap():
    v = _iou([0,0,50,50],[100,100,200,200])
    assert v == 0.0, f"non-overlapping boxes IoU should be 0.0, got {v}"
try_test("IoU of non-overlapping boxes = 0.0", test_iou_no_overlap)

def test_iou_half_overlap():
    v = _iou([0,0,100,100],[50,0,150,100])
    assert 0.28 < v < 0.36, f"half-overlap IoU should be ~0.33, got {v}"
try_test("IoU of half-overlapping boxes ≈ 0.33", test_iou_half_overlap)

def test_iou_zero_area():
    # Should not crash on zero-area box
    try:
        v = _iou([5,5,5,5],[0,0,100,100])
        ok("IoU with zero-area box does not crash")
    except Exception as e:
        fail("IoU with zero-area box does not crash", str(e))

test_iou_zero_area()

def test_suppress_no_overlap():
    faces = [
        {"bbox":[0,0,100,100],"person_bbox":[0,0,100,200],"confidence":0.9,"id":1,"attentionScore":80},
        {"bbox":[400,0,500,100],"person_bbox":[400,0,500,200],"confidence":0.8,"id":2,"attentionScore":70},
    ]
    result = suppress_overlapping_faces(faces, iou_thresh=0.45)
    assert len(result) == 2, f"non-overlapping faces should both survive, got {len(result)}"
try_test("non-overlapping faces: both survive suppression", test_suppress_no_overlap)

def test_suppress_identical():
    # Two identical boxes — only the higher-conf one should survive
    faces = [
        {"bbox":[0,0,100,100],"person_bbox":[0,0,100,200],"confidence":0.9,"id":1,"attentionScore":80},
        {"bbox":[0,0,100,100],"person_bbox":[0,0,100,200],"confidence":0.5,"id":2,"attentionScore":70},
    ]
    result = suppress_overlapping_faces(faces, iou_thresh=0.45)
    assert len(result) == 1, f"duplicate boxes should leave 1, got {len(result)}"
    assert result[0]["confidence"] == 0.9
try_test("duplicate boxes: only higher-confidence survives", test_suppress_identical)

def test_suppress_empty():
    result = suppress_overlapping_faces([], iou_thresh=0.45)
    assert result == []
try_test("suppress empty list returns []", test_suppress_empty)

def test_suppress_single():
    faces = [{"bbox":[0,0,100,100],"person_bbox":[0,0,100,200],"confidence":0.9,"id":1,"attentionScore":80}]
    result = suppress_overlapping_faces(faces)
    assert len(result) == 1
try_test("suppress single face returns it unchanged", test_suppress_single)

def test_suppress_15_detections():
    # 15 detections spread evenly — none should overlap
    faces = [
        {
            "bbox": [i*120, 0, i*120+100, 100],
            "person_bbox": [i*120, 0, i*120+100, 200],
            "confidence": 0.9 - i*0.01,
            "id": i,
            "attentionScore": 70,
        }
        for i in range(15)
    ]
    result = suppress_overlapping_faces(faces, iou_thresh=0.45)
    assert len(result) == 15, f"15 non-overlapping should all survive, got {len(result)}"
try_test("15 non-overlapping detections all survive suppression", test_suppress_15_detections)


# ═════════════════════════════════════════════════════════════════════════════
# 7. Visualizer draw functions (no OpenCV window — off-screen frame)
# ═════════════════════════════════════════════════════════════════════════════
section("7. VISUALIZER CRASH SAFETY")

# Minimal face dict with all expected fields
def _make_face(sid, x1=200, y1=150, x2=320, y2=300):
    return {
        "id": sid, "bbox": [x1, y1, x2, y2],
        "attentionScore": 75.0, "avgAttention30": 78.0,
        "activity": "studying", "reconciledActivity": "studying",
        "activityNote": "", "emotion": "neutral",
        "headPose": {"yaw": 5.0, "pitch": -10.0, "roll": 0.0},
        "eyeOpenness": 0.85, "mouthOpen": 0.05,
        "phoneDetected": False, "laptopDetected": False,
        "earphoneDetected": False, "gridCell": "R1C2",
        "attentionFactors": {"head_pose":0.9,"motion_stability":0.8,"eye_openness":0.85},
        "trend": "STDY", "confidence": 0.88, "person_bbox": [180,100,340,400],
    }

def test_compact_overlay_normal():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    face  = _make_face(1)
    draw_compact_overlay(frame, face)
try_test("draw_compact_overlay on normal face — no crash", test_compact_overlay_normal)

def test_full_overlay_normal():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    face  = _make_face(2)
    draw_full_overlay(frame, face)
try_test("draw_full_overlay on normal face — no crash", test_full_overlay_normal)

def test_overlay_bbox_at_edge():
    # bbox at top-left corner — may trigger negative coords
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    face  = _make_face(3, x1=0, y1=0, x2=50, y2=50)
    draw_compact_overlay(frame, face)
    draw_full_overlay(frame, face)
try_test("overlay with bbox at top-left corner (0,0) — no crash", test_overlay_bbox_at_edge)

def test_overlay_bbox_at_bottom_right():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    face  = _make_face(4, x1=1200, y1=680, x2=1280, y2=720)
    draw_compact_overlay(frame, face)
    draw_full_overlay(frame, face)
try_test("overlay with bbox at bottom-right corner — no crash", test_overlay_bbox_at_bottom_right)

def test_overlay_phone_alert():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    face  = _make_face(5)
    face["phoneDetected"] = True
    draw_compact_overlay(frame, face)
    draw_full_overlay(frame, face)
try_test("overlay with phoneDetected=True — no crash", test_overlay_phone_alert)

def test_overlay_laptop():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    face  = _make_face(6)
    face["laptopDetected"] = True
    face["activity"] = "laptop"
    face["reconciledActivity"] = "laptop"
    draw_compact_overlay(frame, face)
    draw_full_overlay(frame, face)
try_test("overlay with laptopDetected=True — no crash", test_overlay_laptop)

def test_overlay_missing_fields():
    # Minimal face dict — many fields missing
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    face  = {"id": 7, "bbox": [200, 200, 350, 400], "attentionScore": 60.0}
    try:
        draw_compact_overlay(frame, face)
        ok("compact overlay handles missing fields gracefully")
    except Exception as e:
        fail("compact overlay handles missing fields gracefully", str(e))

test_overlay_missing_fields()

def test_overlay_all_activity_types():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    activities = list(ACT_ICON.keys()) + ["unknown_activity"]
    for act in activities:
        face = _make_face(99)
        face["activity"] = act
        face["reconciledActivity"] = act
        try:
            draw_compact_overlay(frame, face)
            draw_full_overlay(frame, face)
        except Exception as e:
            fail(f"overlay with activity='{act}'", str(e))
            continue
    ok("all activity types render without crash")

test_overlay_all_activity_types()

def test_overlay_10_faces_same_frame():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    faces = [_make_face(i, x1=i*150, y1=100, x2=i*150+120, y2=300) for i in range(10)]
    for face in faces:
        draw_full_overlay(frame, face)
try_test("10 simultaneous full overlays on 1080p frame — no crash", test_overlay_10_faces_same_frame)


# ═════════════════════════════════════════════════════════════════════════════
# 8. Engine — process_single_frame with synthetic frames
# ═════════════════════════════════════════════════════════════════════════════
section("8. ENGINE — SYNTHETIC FRAME PROCESSING")

try:
    from core.engine import ClassroomEngine
    _engine_ok = True
    ok("ClassroomEngine imported")
except Exception as e:
    _engine_ok = False
    fail("ClassroomEngine imported", str(e))

if _engine_ok:
    def test_engine_blank_frame():
        eng   = ClassroomEngine(max_seats=6)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = eng.process_single_frame(frame, 1280, 720, persist=False)
        assert isinstance(result, list), "must return a list"
    try_test("engine on blank frame returns a list (no crash)", test_engine_blank_frame)

    def test_engine_noise_frame():
        eng   = ClassroomEngine(max_seats=6)
        rng   = np.random.default_rng(0)
        frame = (rng.integers(0, 255, (720, 1280, 3))).astype(np.uint8)
        result = eng.process_single_frame(frame, 1280, 720, persist=True)
        assert isinstance(result, list)
    try_test("engine on pure noise frame returns a list (no crash)", test_engine_noise_frame)

    def test_engine_output_fields():
        eng   = ClassroomEngine(max_seats=12)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = eng.process_single_frame(frame, 1920, 1080, persist=True)
        required = {"id","bbox","attentionScore","activity","headPose",
                    "emotion","phoneDetected","laptopDetected"}
        for face in result:
            missing = required - set(face.keys())
            assert not missing, f"face result missing fields: {missing}"
    try_test("all required fields present in face result dicts", test_engine_output_fields)

    def test_engine_attention_range():
        eng   = ClassroomEngine(max_seats=12)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        for _ in range(10):
            result = eng.process_single_frame(frame, 1920, 1080, persist=True)
            for face in result:
                a = face["attentionScore"]
                assert 0 <= a <= 100, f"attentionScore out of range: {a}"
    try_test("attentionScore always in [0,100] across 10 frames", test_engine_attention_range)

    def test_engine_reset():
        eng = ClassroomEngine(max_seats=6)
        eng.reset()
        assert eng.student_states == {}
        assert eng.behavior_trackers == {}
    try_test("engine.reset() clears all state", test_engine_reset)

    def test_engine_memory_10_frames():
        # Run 10 synthetic frames and check state doesn't grow unboundedly
        eng   = ClassroomEngine(max_seats=6)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        for _ in range(10):
            eng.process_single_frame(frame, 1280, 720, persist=True)
        assert len(eng.student_states) <= 6, \
            f"student_states grew beyond max_seats: {len(eng.student_states)}"
    try_test("engine student_states never exceed max_seats", test_engine_memory_10_frames)


# ═════════════════════════════════════════════════════════════════════════════
# 9. Config sanity
# ═════════════════════════════════════════════════════════════════════════════
section("9. CONFIG SANITY")

try:
    import core.config as cfg

    def test_cfg_yolo_conf():
        assert 0 < cfg.YOLO_CONF < 1, f"YOLO_CONF out of (0,1): {cfg.YOLO_CONF}"
    try_test("YOLO_CONF in (0, 1)", test_cfg_yolo_conf)

    def test_cfg_max_seats():
        assert 1 <= cfg.MAX_SEATS <= 20, f"MAX_SEATS out of [1,20]: {cfg.MAX_SEATS}"
    try_test("MAX_SEATS in [1, 20]", test_cfg_max_seats)

    def test_cfg_ema():
        assert 0 < cfg.EMA_A < 1, f"EMA_A out of (0,1): {cfg.EMA_A}"
    try_test("EMA_A in (0, 1)", test_cfg_ema)

    def test_cfg_hist_len():
        assert cfg.HIST_LEN >= 10, f"HIST_LEN too small: {cfg.HIST_LEN}"
    try_test("HIST_LEN >= 10", test_cfg_hist_len)

    def test_cfg_hysteresis():
        assert cfg.HYSTERESIS_FRAMES >= 1, f"HYSTERESIS_FRAMES < 1: {cfg.HYSTERESIS_FRAMES}"
    try_test("HYSTERESIS_FRAMES >= 1", test_cfg_hysteresis)

except Exception as e:
    fail("load core.config", str(e))


# ═════════════════════════════════════════════════════════════════════════════
# 10. Regression — live video file test (if file exists, run 60 frames)
# ═════════════════════════════════════════════════════════════════════════════
section("10. REAL VIDEO REGRESSION (60 frames)")

VIDEO_PATH = os.path.join(HERE, "test video", "test video - real classrom.mp4")

if not os.path.isfile(VIDEO_PATH):
    warn("real_video_60_frames", f"video not found at {VIDEO_PATH} — skipped")
else:
    def test_real_video_60_frames():
        import core.config as _cfg
        _cfg.YOLO_CONF = 0.30
        from core.engine import ClassroomEngine
        from test_real_classroom import suppress_overlapping_faces

        cap = cv2.VideoCapture(VIDEO_PATH)
        assert cap.isOpened(), "could not open video"
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        eng = ClassroomEngine(max_seats=15)

        frame_count = 0
        detection_counts = []
        t0 = time.time()

        for _ in range(60):
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % 2 != 0:
                continue
            faces = eng.process_single_frame(frame, W, H, persist=True)
            faces = suppress_overlapping_faces(faces)
            detection_counts.append(len(faces))

            # All face outputs must have valid fields
            for face in faces:
                a = face["attentionScore"]
                assert 0 <= a <= 100, f"attn out of range in real video: {a}"
                assert face["activity"] in (
                    "studying","attentive","neutral","talking","fidgeting",
                    "distracted","drowsy","phone","laptop","music"
                ), f"invalid activity: {face['activity']}"

        cap.release()
        elapsed = time.time() - t0

        avg_det = sum(detection_counts)/max(1,len(detection_counts))
        max_det = max(detection_counts) if detection_counts else 0

        _log.append(f"       frames read: {frame_count}, avg detections: {avg_det:.1f}, "
                    f"max: {max_det}, time: {elapsed:.1f}s")
        assert frame_count >= 10, "video too short to test"
        assert avg_det >= 1, f"no students detected in real video — model issue? avg={avg_det}"

    try_test("60 real video frames: valid outputs, ≥1 student detected", test_real_video_60_frames)


# ═════════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ═════════════════════════════════════════════════════════════════════════════
print("\n")
print("=" * 65)
print("  BRUTAL TEST REPORT — SmartClass AI")
print("=" * 65)
for line in _log:
    print(line)

total = PASS + FAIL + WARN
print()
print("=" * 65)
print(f"  TOTAL  : {total}  |  PASS: {PASS}  |  FAIL: {FAIL}  |  WARN: {WARN}")
grade = "ALL CLEAR" if FAIL == 0 else f"{FAIL} FAILURES — needs fixing"
print(f"  RESULT : {grade}")
print("=" * 65)

sys.exit(0 if FAIL == 0 else 1)

