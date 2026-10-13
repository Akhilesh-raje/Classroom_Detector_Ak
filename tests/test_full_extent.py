# -*- coding: utf-8 -*-
"""
SmartClass AI — Full-Extent Comprehensive Test Suite (V2.4)
===========================================================
Performs exhaustive verification across all architectural layers:
  Suite 1: Modular Pipeline Stage Isolation (Scheduler, Detection, Association, Analysis, Output)
  Suite 2: Universal Scene Representation (USR) & Scene Graph Integrity
  Suite 3: Geometry & Topology Mathematical Invariance
  Suite 4: Ground Truth Strict Statistical Alignment (Official GT Audit + Frame-by-Frame validation)
  Suite 5: Stress, Concurrency & Edge Case Boundary Tests
  Suite 6: Multi-Frame Temporal Consistency & 300-Frame Video Stress Test
"""
import os
import sys
import json
import time
import math
import tempfile
import traceback
from collections import defaultdict

# Force UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join(ROOT, "backend")
sys.path.insert(0, ROOT)
sys.path.insert(0, BACKEND)

video_path = os.path.join(ROOT, "classrom video of wit", "videos", "real_classroom.mp4")
if not os.path.exists(video_path):
    video_path = os.path.join(ROOT, "test video", "test video - real classrom.mp4")

import cv2
import numpy as np

# Test runner harness
TOTAL_TESTS = 0
PASSED_TESTS = 0
FAILED_TESTS = 0
FAILURES = []

def record(name: str, passed: bool, detail: str = ""):
    global TOTAL_TESTS, PASSED_TESTS, FAILED_TESTS
    TOTAL_TESTS += 1
    if passed:
        PASSED_TESTS += 1
        icon = "[PASS]"
        print(f"  {icon} {name}" + (f" -> {detail}" if detail else ""))
    else:
        FAILED_TESTS += 1
        icon = "[FAIL]"
        FAILURES.append((name, detail))
        print(f"  {icon} {name} -> {detail}")

def suite_header(title: str):
    print(f"\n{'='*75}")
    print(f"  {title}")
    print(f"{'='*75}")


# ==============================================================================
# SUITE 1: MODULAR PIPELINE STAGE ISOLATION
# ==============================================================================
suite_header("SUITE 1: MODULAR PIPELINE STAGES ISOLATION")

try:
    from core.pipeline.scheduler import PipelineScheduler, SchedulerConfig
    
    # 1.1 Scheduler cadences
    sched_cfg = SchedulerConfig(fast_cadence=1, medium_cadence=2, slow_cadence=5, async_reasoning_interval=10)
    sched = PipelineScheduler(sched_cfg)
    
    c_fast = [sched.should_run_fast(i) for i in range(10)]
    record("Scheduler fast tier runs every frame", all(c_fast), f"count={sum(c_fast)}/10")
    
    c_med = [sched.should_run_medium(i) for i in range(10)]
    record("Scheduler medium tier respects cadence=2", c_med == [True, False, True, False, True, False, True, False, True, False])
    
    c_slow = [sched.should_run_slow(i) for i in range(15)]
    record("Scheduler slow tier respects cadence=5", [i for i, v in enumerate(c_slow) if v] == [0, 5, 10])
    
    c_async = [sched.should_run_async_reasoning(i) for i in range(25)]
    record("Scheduler async tier respects interval=10", [i for i, v in enumerate(c_async) if v] == [0, 10, 20])
except Exception as e:
    record("PipelineScheduler isolation", False, traceback.format_exc())

try:
    from core.pipeline.detection_stage import DetectionStage
    from core.topology import PhysicalSeat, ClassroomTopology, PersonTrack
    
    det_stage = DetectionStage(min_conf=0.25)
    
    # Blank frame
    blank = np.zeros((1080, 1920, 3), dtype=np.uint8)
    raw_dets, phones, laptops, earphones, tracks, t_by_id = det_stage.execute(blank, persist=False, frame_idx=0)
    record("DetectionStage on blank frame returns empty lists without crash", 
           isinstance(raw_dets, list) and len(raw_dets) == 0 and len(tracks) == 0)
    
    # Pure noise frame
    np.random.seed(42)
    noise = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    raw_dets_n, _, _, _, tracks_n, _ = det_stage.execute(noise, persist=False, frame_idx=1)
    record("DetectionStage on random noise frame handles gracefully", isinstance(raw_dets_n, list))
except Exception as e:
    record("DetectionStage isolation", False, traceback.format_exc())

try:
    from core.pipeline.association_stage import AssociationStage
    from core.seat_associator import SeatAssociator
    from core.stabilizer import SeatStabilizer
    
    assoc_stage = AssociationStage()
    associator = SeatAssociator()
    stabilizer = SeatStabilizer(max_seats=12)
    
    # Create synthetic topology: 2 rows of 6 seats (6x6)
    seats = []
    # Row 1 (FRONT): y=660, 6 seats
    x_r1 = [350, 600, 850, 1100, 1350, 1600]
    for idx, x in enumerate(x_r1):
        seats.append(PhysicalSeat(
            seat_id=idx + 1,
            row=0, col=idx,
            label=f"R1C{idx+1}",
            anchor=(float(x), 660.0),
            bbox=(x - 70, 580, x + 70, 740),
            expected_head_region=(x - 50, 500, x + 50, 600)
        ))
    # Row 2 (BACK): y=510, 6 seats
    x_r2 = [300, 550, 800, 1050, 1300, 1550]
    for idx, x in enumerate(x_r2):
        seats.append(PhysicalSeat(
            seat_id=7 + idx,
            row=1, col=idx,
            label=f"R2C{idx+1}",
            anchor=(float(x), 510.0),
            bbox=(x - 65, 440, x + 65, 580),
            expected_head_region=(x - 45, 380, x + 45, 470)
        ))
    topo = ClassroomTopology(seats=seats, rows=2, cols_per_row=[6, 6], source="structure_ai")
    seat_label_to_id = {s.label: s.seat_id for s in seats}
    
    # Test AssociationStage with exact anchor tracks
    synth_tracks = []
    t_by_id = {}
    for s in seats:
        pt = PersonTrack(
            track_id=s.seat_id,
            bbox=list(s.bbox),
            face_bbox=list(s.expected_head_region),
            head_point=((s.expected_head_region[0] + s.expected_head_region[2])/2.0, (s.expected_head_region[1] + s.expected_head_region[3])/2.0),
            bottom_point=(s.anchor[0], s.anchor[1]),
            confidence=0.95,
            frame_idx=0,
            torso_point=(s.anchor[0], s.anchor[1] - 40.0),
            depth=0.5
        )
        synth_tracks.append(pt)
        t_by_id[s.seat_id] = pt
    
    raw_synth_dets = [{"id": s.seat_id, "bbox": list(s.expected_head_region), "confidence": 0.95} for s in seats]
    mapped_dets, bench_rec = assoc_stage.execute(
        tracks=synth_tracks,
        raw_dets=raw_synth_dets,
        topology=topo,
        associator=associator,
        seat_label_to_id=seat_label_to_id,
        stabilizer=stabilizer,
        frame_idx=0,
        W_px=1920,
        H_px=1080,
        t_by_id=t_by_id
    )
    
    assigned_labels = [d["seat_label"] for d in mapped_dets if d.get("status") == "OK"]
    expected_labels = [s.label for s in seats]
    all_matched = sorted(assigned_labels) == sorted(expected_labels)
    record("AssociationStage maps 12 synthetic tracks exactly to 12 physical seats", all_matched, f"matched={len(assigned_labels)}/12")
    
    # Overcrowded scenario: 16 tracks on 12 seats
    extra_tracks = list(synth_tracks)
    for extra_id in [101, 102, 103, 104]:
        extra_tracks.append(PersonTrack(
            track_id=extra_id,
            bbox=[100, 100, 200, 200],
            face_bbox=[120, 100, 180, 160],
            head_point=(150.0, 130.0),
            bottom_point=(150.0, 200.0),
            confidence=0.8,
            frame_idx=1
        ))
    extra_raw = list(raw_synth_dets) + [{"id": eid, "bbox": [120, 100, 180, 160], "confidence": 0.8} for eid in [101, 102, 103, 104]]
    mapped_over, _ = assoc_stage.execute(
        tracks=extra_tracks,
        raw_dets=extra_raw,
        topology=topo,
        associator=associator,
        seat_label_to_id=seat_label_to_id,
        stabilizer=stabilizer,
        frame_idx=1,
        W_px=1920,
        H_px=1080,
        t_by_id={t.track_id: t for t in extra_tracks}
    )
    ok_count = sum(1 for d in mapped_over if d.get("status") == "OK")
    unassigned_count = sum(1 for d in mapped_over if d.get("status") == "UNASSIGNED")
    record("AssociationStage prevents seat over-allocation (max 12 OK)", ok_count == 12, f"OK={ok_count}, UNASSIGNED={unassigned_count}")
except Exception as e:
    record("AssociationStage isolation", False, traceback.format_exc())

try:
    from core.pipeline.output_stage import OutputStage
    from core.scene.scene_state import SceneState, EntityState
    
    out_stage = OutputStage()
    processed_faces = [
        {
            "id": 1, "track_id": 1, "seat_label": "R1C1", "gridCell": "R1C1",
            "status": "OK", "bbox": [400, 500, 500, 600], "attentionScore": 88.5,
            "activity": "attentive", "confidence": 0.92, "row": 0, "col": 0,
            "topology": {"row": 0, "col": 0, "seat_label": "R1C1"}
        },
        {
            "id": 2, "track_id": 2, "seat_label": "R1C2", "gridCell": "R1C2",
            "status": "OK", "bbox": [650, 500, 750, 600], "attentionScore": 72.0,
            "activity": "studying", "confidence": 0.88, "row": 0, "col": 1,
            "topology": {"row": 0, "col": 1, "seat_label": "R1C2"}
        }
    ]
    
    valid_faces, scene_state = out_stage.execute(
        processed_faces=processed_faces,
        topology=topo,
        frame_idx=10,
        timestamp=1000.0
    )
    record("OutputStage produces valid SceneState", isinstance(scene_state, SceneState))
    record("OutputStage computes correct scene-level aggregations", 
           scene_state.total_seats == 12 and scene_state.occupied_seats == 2,
           f"total={scene_state.total_seats}, occ={scene_state.occupied_seats}")
    record("OutputStage generated spatial scene relationships", len(scene_state.relationships) > 0,
           f"count={len(scene_state.relationships)}")
except Exception as e:
    record("OutputStage isolation", False, traceback.format_exc())


# ==============================================================================
# SUITE 2: UNIVERSAL SCENE REPRESENTATION & SCENE GRAPH
# ==============================================================================
suite_header("SUITE 2: UNIVERSAL SCENE REPRESENTATION (USR)")

try:
    from core.scene.scene_state import SceneState, EntityState, SceneRelationship
    
    # 2.1 EntityState serialization & deserialization
    e1 = EntityState(
        entity_id="person_1",
        entity_class="person",
        bbox=[100, 200, 300, 400],
        seat_anchor=(200.0, 400.0),
        depth=0.45,
        pose_state="seated",
        head_direction="FRONT",
        head_pose={"yaw": -2.5, "pitch": 5.1, "roll": 0.2},
        activity="attentive",
        attention_score=85.0,
        seat_id=1,
        seat_label="R1C1",
        row=0, col=0,
        student_id=42,
        student_name="Alice Smith",
        detection_confidence=0.96,
        tracking_confidence=0.90,
        association_confidence=0.95,
        status="OK"
    )
    e1_dict = e1.to_dict()
    e1_rebuilt = EntityState.from_dict(e1_dict)
    
    record("EntityState serializes to dict with all fields", isinstance(e1_dict, dict) and e1_dict["seat_label"] == "R1C1")
    record("EntityState round-trips from dict without field loss", 
           e1_rebuilt.entity_id == e1.entity_id and e1_rebuilt.head_pose == e1.head_pose and e1_rebuilt.student_name == "Alice Smith")
    
    # 2.2 SceneRelationship model
    rel = SceneRelationship(source_id="person_1", relation="adjacent_to", target_id="person_2", confidence=0.95)
    record("SceneRelationship serializes to dict", rel.to_dict()["relation"] == "adjacent_to")
    
    # 2.3 SceneState methods & JSON round-trip
    ss = SceneState(
        timestamp=123456789.0,
        frame_idx=42,
        entities=[e1],
        people_count=1,
        occupied_seats=1,
        total_seats=12,
        average_attention=85.0,
        relationships=[rel]
    )
    
    found_e = ss.get_entity_by_id("person_1")
    record("SceneState.get_entity_by_id finds entity", found_e is not None and found_e.entity_id == "person_1")
    
    found_s = ss.get_entity_by_seat("R1C1")
    record("SceneState.get_entity_by_seat finds entity", found_s is not None and found_s.seat_label == "R1C1")
    
    not_found = ss.get_entity_by_seat("R2C5")
    record("SceneState lookup non-existent seat returns None", not_found is None)
    
    json_str = json.dumps(ss.to_dict())
    loaded = json.loads(json_str)
    record("SceneState serializes cleanly to JSON string", len(loaded["entities"]) == 1 and loaded["frame_idx"] == 42)
except Exception as e:
    record("USR & Scene Graph verification", False, traceback.format_exc())


# ==============================================================================
# SUITE 3: GEOMETRY & TOPOLOGY MATHEMATICAL INVARIANCE
# ==============================================================================
suite_header("SUITE 3: GEOMETRY & TOPOLOGY INVARIANCE")

try:
    from core.structure.geometry import (
        compute_pca_axis, mask_to_polygon, point_in_polygon, bbox_iou,
        cluster_depth_layers_otsu
    )
    
    # 3.1 PCA axis on horizontal line
    horiz_pts = np.array([[10, 50], [30, 50], [80, 50], [150, 50]], dtype=np.float32)
    pca_h = compute_pca_axis(horiz_pts)
    record("PCA on horizontal points has near-zero angle", abs(pca_h["angle_deg"]) < 1.0, f"angle={pca_h['angle_deg']:.2f}")
    
    # 3.2 IoU boundary conditions
    b1 = [0, 0, 100, 100]
    b2 = [0, 0, 100, 100]
    record("IoU identical boxes is 1.0", abs(bbox_iou(b1, b2) - 1.0) < 1e-6)
    
    b_disjoint = [200, 200, 300, 300]
    record("IoU disjoint boxes is 0.0", bbox_iou(b1, b_disjoint) == 0.0)
    
    b_half = [0, 0, 100, 50]
    record("IoU half-overlap is 0.5", abs(bbox_iou(b1, b_half) - 0.5) < 1e-6)
    
    # Zero-area box IoU does not crash / divide by zero
    b_zero = [10, 10, 10, 10]
    record("IoU zero-area box returns 0.0 without crash", bbox_iou(b1, b_zero) == 0.0)
    
    # 3.3 Polygon containment
    square_poly = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int32)
    record("point_in_polygon inside point returns True", point_in_polygon((50.0, 50.0), square_poly) is True)
    record("point_in_polygon outside point returns False", point_in_polygon((150.0, 50.0), square_poly) is False)
    
    # 3.4 Depth Otsu clustering
    depth_layer_front = np.ones((50, 50), dtype=np.float32) * 0.8  # closer
    depth_layer_back = np.ones((50, 50), dtype=np.float32) * 0.2   # farther
    combined_depth = np.vstack([depth_layer_back, depth_layer_front])
    thresh, labels = cluster_depth_layers_otsu(combined_depth)
    record("cluster_depth_layers_otsu splits depth correctly", 0.2 < thresh < 0.8)
    record("Otsu labels partition two layers", set(labels) == {0, 1})
    
    # 3.5 Physical classroom topology ordering invariance
    # In camera view: FRONT row has larger Y than BACK row
    r1_y = np.mean([s.anchor[1] for s in topo.seats if s.row == 0])
    r2_y = np.mean([s.anchor[1] for s in topo.seats if s.row == 1])
    record("Classroom camera geometry: Front row Y > Back row Y", r1_y > r2_y, f"R1_y={r1_y:.1f} > R2_y={r2_y:.1f}")
    
    # Seat monotonic X ordering within rows
    r1_x = [s.anchor[0] for s in topo.seats if s.row == 0]
    r2_x = [s.anchor[0] for s in topo.seats if s.row == 1]
    record("Row 1 seats ordered Left-to-Right monotonically", r1_x == sorted(r1_x))
    record("Row 2 seats ordered Left-to-Right monotonically", r2_x == sorted(r2_x))
except Exception as e:
    record("Geometry & Topology verification", False, traceback.format_exc())


# ==============================================================================
# SUITE 4: GROUND TRUTH STRICT STATISTICAL ALIGNMENT
# ==============================================================================
suite_header("SUITE 4: GROUND TRUTH STRICT STATISTICAL ALIGNMENT")

GT_FILE = os.path.join(BACKEND, "tests", "ground_truth_classroom.json")
engine = None
calib_frames = []

try:
    with open(GT_FILE, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    
    from benchmark_ground_truth import evaluate_ground_truth
    gt_eval = evaluate_ground_truth(gt_data)
    
    record("Ground Truth dataset loaded and verified", gt_eval["total_samples"] == 60)
    record("GT Physical Seat Accuracy >= 95%", gt_eval["seat_accuracy"] >= 95.0, f"accuracy={gt_eval['seat_accuracy']:.2f}%")
    record("GT Row Semantics Accuracy >= 95%", gt_eval["row_accuracy"] >= 95.0, f"accuracy={gt_eval['row_accuracy']:.2f}%")
    record("Zero Double-Booking Errors across GT reference dataset", gt_eval["double_booking_count"] == 0, f"errors={gt_eval['double_booking_count']}")
    record("GT Mean Geometric Match Score >= 0.85", gt_eval["mean_geo_match"] >= 0.85, f"score={gt_eval['mean_geo_match']:.3f}")
    
    from core.engine import ClassroomEngine
    engine = ClassroomEngine(max_seats=20)
    
    video_path = os.path.join(ROOT, "classrom video of wit", "videos", "real_classroom.mp4")
    if not os.path.exists(video_path):
        video_path = os.path.join(ROOT, "test video", "test video - real classrom.mp4")
    
    cap = cv2.VideoCapture(video_path)
    calib_frames = []
    for _ in range(30):
        ret, frame = cap.read()
        if not ret: break
        calib_frames.append(frame)
    
    engine.calibrate_structure(calib_frames, 1920, 1080)
    record("ClassroomEngine calibrated with Structure AI", engine.topology is not None and len(engine.topology.seats) == 12)
    
    # Test Frame 30 full detection alignment
    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
    ret, f30 = cap.read()
    faces_30 = engine.process_single_frame(f30, 1920, 1080, persist=True)
    assigned_30 = {f.get("seat_label"): f for f in faces_30 if f.get("status") == "OK"}
    record("Frame 30 discovers all 12 classroom seats with OK status", len(assigned_30) == 12, f"assigned={len(assigned_30)}/12")
    
    cap.release()

except Exception as e:
    record("Ground Truth Alignment suite", False, traceback.format_exc())


# ==============================================================================
# SUITE 5: STRESS, CONCURRENCY & BOUNDARY TESTS
# ==============================================================================
suite_header("SUITE 5: STRESS, CONCURRENCY & BOUNDARY CONDITIONS")

try:
    # 5.1 Empty / Degenerate frame stress
    empty_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    res_empty = engine.process_single_frame(empty_frame, 1920, 1080, persist=False)
    # On an empty frame with no persons, either 0 detections or ghost holds are returned
    record("process_single_frame on all-black frame handles gracefully without crash", 
           isinstance(res_empty, list) and (len(res_empty) == 0 or all(r.get("ghost") for r in res_empty)))
    
    # 5.2 Tiny frame stress
    tiny_frame = np.zeros((64, 64, 3), dtype=np.uint8)
    res_tiny = engine.process_single_frame(tiny_frame, 64, 64, persist=False)
    record("process_single_frame on tiny 64x64 frame handles gracefully", isinstance(res_tiny, list))
    
    # 5.3 Extreme aspect ratio (1920x200 banner)
    banner_frame = np.zeros((200, 1920, 3), dtype=np.uint8)
    res_banner = engine.process_single_frame(banner_frame, 1920, 200, persist=False)
    record("process_single_frame on extreme 1920x200 aspect ratio handles gracefully", isinstance(res_banner, list))
    
    # 5.4 State Reset Safety
    engine.reset()
    record("engine.reset() cleanly clears student states", len(engine.student_states) == 0)
    record("engine.reset() resets calibration contract for fresh session", engine.topology is None and engine._calibrated is False)
    
    # 5.5 Re-calibration idempotency
    engine.calibrate_structure(calib_frames[:15], 1920, 1080)
    record("Re-calibration produces consistent 12 seats", len(engine.topology.seats) == 12)
    record("Stabilizer is locked after re-calibration", engine.stabilizer._locked is True)

except Exception as e:
    record("Stress & Boundary suite", False, traceback.format_exc())


# ==============================================================================
# SUITE 6: MULTI-FRAME TEMPORAL CONSISTENCY & 300-FRAME VIDEO STRESS TEST
# ==============================================================================
suite_header("SUITE 6: 300-FRAME VIDEO PIPELINE STRESS TEST")

try:
    cap = cv2.VideoCapture(video_path)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_test_frames = min(300, total_video_frames)
    
    print(f"  Streaming {num_test_frames} frames through full unified engine...")
    
    t_start = time.time()
    processed_count = 0
    seat_assignment_history = defaultdict(list)
    track_id_history = defaultdict(list)
    exception_count = 0
    
    for f_idx in range(num_test_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            faces = engine.process_single_frame(frame, 1920, 1080, persist=True)
            processed_count += 1
            
            for f in faces:
                tid = f.get("track_id")
                sl = f.get("seat_label")
                st = f.get("status")
                if tid is not None and st == "OK" and sl:
                    seat_assignment_history[tid].append(sl)
                    track_id_history[sl].append(tid)
        except Exception:
            exception_count += 1
        
        if (f_idx + 1) % 60 == 0 or (f_idx + 1) == num_test_frames:
            elapsed = time.time() - t_start
            fps = (f_idx + 1) / max(0.001, elapsed)
            print(f"    Progress: {f_idx + 1:3d}/{num_test_frames} frames ({fps:4.1f} FPS)")
    
    cap.release()
    total_time = time.time() - t_start
    avg_fps = processed_count / max(0.001, total_time)
    
    record("Completed 300 frames without pipeline exceptions", exception_count == 0, f"exceptions={exception_count}")
    record("Processed all requested frames", processed_count == num_test_frames, f"count={processed_count}/{num_test_frames}")
    record("Processing throughput is stable (> 2.0 FPS on CPU)", avg_fps >= 2.0, f"avg_fps={avg_fps:.1f}")
    
    # Export benchmark CSV
    csv_out = os.path.join(ROOT, "scratch", "benchmark_full_extent_300f.csv")
    engine.export_benchmark_csv(csv_out)
    record("Export benchmark CSV generated", os.path.exists(csv_out) and os.path.getsize(csv_out) > 0, 
           f"size={os.path.getsize(csv_out) if os.path.exists(csv_out) else 0} bytes")

    # Run benchmark metrics
    from benchmark_v2_2 import compute_v2_2_metrics
    bm_metrics = compute_v2_2_metrics(csv_out)
    record("Double-booking errors is strictly 0 across 300 frames", bm_metrics.get("double_booked_frames", -1) == 0)
    record("Unassigned rate is below target (< 10%)", bm_metrics.get("unassigned_rate", 100.0) < 10.0, f"unassigned={bm_metrics.get('unassigned_rate', 0.0):.2f}%")
    record("Physical seats discovered is exactly 12", bm_metrics.get("mapped_seats", 0) == 12)

except Exception as e:
    record("300-Frame Stress Test", False, traceback.format_exc())


# ==============================================================================
# SUMMARY & FINAL VERDICT
# ==============================================================================
print(f"\n{'='*75}")
print(f"  FULL-EXTENT STRESS & SYSTEM VERIFICATION REPORT")
print(f"{'='*75}")
pct = (PASSED_TESTS / TOTAL_TESTS * 100.0) if TOTAL_TESTS > 0 else 0.0
print(f"  Total Validations:  {TOTAL_TESTS}")
print(f"  Passed:             {PASSED_TESTS} ({pct:.1f}%)")
print(f"  Failed:             {FAILED_TESTS}")

if FAILED_TESTS > 0:
    print(f"\n  Failed Validations:")
    for name, detail in FAILURES:
        print(f"    [FAIL] {name} -> {detail}")

print(f"{'='*75}")
if FAILED_TESTS == 0:
    print("  VERDICT: SYSTEM MEETS ALL FULL-EXTENT SPECIFICATIONS (100% PASS) [OK]")
else:
    print(f"  VERDICT: SYSTEM HAS {FAILED_TESTS} FAILING CONDITIONS [FAIL]")
print(f"{'='*75}\n")

sys.exit(0 if FAILED_TESTS == 0 else 1)
