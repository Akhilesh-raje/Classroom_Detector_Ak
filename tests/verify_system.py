# -*- coding: utf-8 -*-
"""
SmartClass System Verification Suite
=====================================
Validates every critical component end-to-end:
  T1  Module imports & class availability
  T2  Structure AI geometry pipeline (detector, segmenter, depth, builder)
  T3  TopologyBuilder → ClassroomTopologyMap synthesis
  T4  ClassroomTopology.from_structure_map conversion
  T5  ClassroomEngine calibrate (baseline 'auto' mode)
  T6  ClassroomEngine calibrate_structure (Structure AI mode)
  T7  process_single_frame produces valid output
  T8  Ground truth seat/row count alignment
  T9  Benchmark report integrity
  T10 Cross-frame topology invariance
"""
import os
import sys
import json
import time
import traceback

# Force UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

import cv2
import numpy as np

# ── Result tracking ───────────────────────────────────────────────────────────
results = []

def test(name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    results.append({"name": name, "passed": passed, "detail": detail})
    icon = "✅" if passed else "❌"
    print(f"  {icon} {name}: {status}" + (f" — {detail}" if detail else ""))

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


topo_map = None
engine_baseline = None
engine_struct = None

# ═══════════════════════════════════════════════════════════════════════════════
# T1: Module Imports & Class Availability
# ═══════════════════════════════════════════════════════════════════════════════
section("T1: Module Imports & Class Availability")
try:
    from core.topology import (
        PhysicalSeat, ClassroomTopology, TopologyConfig,
        TopologyDiscovery, PersonTrack, PersonObservation, RowGeometry
    )
    test("Import core.topology classes", True)
except Exception as e:
    test("Import core.topology classes", False, str(e))

try:
    from core.seat_associator import SeatAssociator, SeatAssignment
    test("Import core.seat_associator", True)
except Exception as e:
    test("Import core.seat_associator", False, str(e))

try:
    from core.engine import ClassroomEngine
    test("Import core.engine.ClassroomEngine", True)
except Exception as e:
    test("Import core.engine.ClassroomEngine", False, str(e))

try:
    from core.structure import (
        StructureEngine, TopologyBuilder, ClassroomTopologyMap,
        Bench, PhysicalSeatGeometry, FurnitureDetector, FurnitureSegmenter,
        MonocularDepthEstimator, compute_pca_axis, mask_to_polygon,
        point_in_polygon, bbox_iou, cluster_depth_layers_otsu
    )
    test("Import core.structure package (all exports)", True)
except Exception as e:
    test("Import core.structure package", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# T2: Structure AI Geometry Pipeline — Unit Checks
# ═══════════════════════════════════════════════════════════════════════════════
section("T2: Structure AI Geometry Utilities")
try:
    from core.structure.geometry import compute_pca_axis, mask_to_polygon, bbox_iou
    # PCA on (N, 2) coordinate array
    pts = np.array([[10, 20], [50, 60], [80, 90], [100, 110]], dtype=np.float32)
    axis = compute_pca_axis(pts)
    test("compute_pca_axis returns dict", isinstance(axis, dict) and "angle_deg" in axis, f"angle={axis.get('angle_deg', 0):.1f}°")
except Exception as e:
    test("compute_pca_axis", False, str(e))

try:
    iou = bbox_iou([0, 0, 100, 100], [50, 50, 150, 150])
    test("bbox_iou computes correctly", abs(iou - (50*50)/(100*100 + 100*100 - 50*50)) < 0.01,
         f"IoU={iou:.4f}")
except Exception as e:
    test("bbox_iou", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# T3: TopologyBuilder → ClassroomTopologyMap synthesis
# ═══════════════════════════════════════════════════════════════════════════════
section("T3: TopologyBuilder Synthesis")
try:
    builder = TopologyBuilder(frame_size=(1920, 1080))
    # build_from_geometry with empty masks should still produce a valid fallback topology
    topo_map = builder.build_from_geometry([], depth_map=None)
    test("TopologyBuilder.build_from_geometry returns ClassroomTopologyMap",
         isinstance(topo_map, ClassroomTopologyMap))
    test("Synthesized topology has seats", len(topo_map.seats) > 0,
         f"{len(topo_map.seats)} seats")
    test("Synthesized topology has benches", len(topo_map.benches) > 0,
         f"{len(topo_map.benches)} benches")
    test("Synthesized topology has rows", topo_map.rows > 0,
         f"{topo_map.rows} rows")
    test("All seats have valid center", all(
        isinstance(s.center, tuple) and len(s.center) == 2 for s in topo_map.seats),
         "center = (x, y) tuples")
    test("All seats have display_label", all(
        isinstance(s.display_label, str) and len(s.display_label) > 0 for s in topo_map.seats))
except Exception as e:
    test("TopologyBuilder synthesis", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════════════
# T4: ClassroomTopology.from_structure_map conversion
# ═══════════════════════════════════════════════════════════════════════════════
section("T4: ClassroomTopology.from_structure_map")
try:
    topo = ClassroomTopology.from_structure_map(topo_map)
    test("from_structure_map returns ClassroomTopology", isinstance(topo, ClassroomTopology))
    test("Seats preserved",
         len(topo.seats) == len(topo_map.seats),
         f"{len(topo.seats)} seats")
    test("Source is 'structure_ai'", topo.source == "structure_ai")
    test("Row count matches benches", topo.rows == topo_map.rows,
         f"topo.rows={topo.rows}, map.rows={topo_map.rows}")
    test("cols_per_row populated", len(topo.cols_per_row) > 0 and all(c > 0 for c in topo.cols_per_row),
         f"cols_per_row={topo.cols_per_row}")
    test("row_geometries populated", len(topo.row_geometries) > 0,
         f"{len(topo.row_geometries)} RowGeometry entries")
    test("All PhysicalSeats have seat_id", all(s.seat_id is not None for s in topo.seats))
    test("All PhysicalSeats have anchor", all(isinstance(s.anchor, tuple) for s in topo.seats))
    test("All PhysicalSeats have bbox", all(len(s.bbox) == 4 for s in topo.seats))
    test("Seats have expected_head_region", all(s.expected_head_region is not None for s in topo.seats))

    # Verify R1/R2 labeling pattern
    labels = [s.label for s in topo.seats]
    r1_seats = [l for l in labels if l.startswith("R1")]
    r2_seats = [l for l in labels if l.startswith("R2")]
    test("R1 row has seats", len(r1_seats) > 0, f"R1: {r1_seats}")
    test("R2 row has seats", len(r2_seats) > 0, f"R2: {r2_seats}")
except Exception as e:
    test("from_structure_map conversion", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════════════
# T5 & T6: ClassroomEngine calibration (both modes)
# ═══════════════════════════════════════════════════════════════════════════════
section("T5: ClassroomEngine Calibration (Baseline auto)")
VIDEO_PATH = os.path.join(ROOT, "classrom video of wit", "videos", "real_classroom.mp4")
video_exists = os.path.isfile(VIDEO_PATH)
test("Video file exists", video_exists, VIDEO_PATH if video_exists else "NOT FOUND")

calib_frames = []
test_frames = {}
if video_exists:
    cap = cv2.VideoCapture(VIDEO_PATH)
    f_idx = 0
    while cap.isOpened() and f_idx < 300:
        ret, frame = cap.read()
        if not ret:
            break
        if f_idx < 30:
            calib_frames.append(frame.copy())
        if f_idx in [30, 90, 150, 210, 270]:
            test_frames[f_idx] = frame.copy()
        f_idx += 1
    cap.release()
    test("Extracted calibration frames", len(calib_frames) >= 10, f"{len(calib_frames)} frames")
    test("Extracted GT test frames", len(test_frames) >= 3, f"{len(test_frames)} frames")

# Baseline calibration
if calib_frames:
    try:
        engine_baseline = ClassroomEngine()
        engine_baseline.calibrate(calib_frames, 1920, 1080, config=TopologyConfig(mode="auto"))
        test("Baseline calibrate() completes", engine_baseline._calibrated)
        test("Baseline discovers seats",
             engine_baseline.topology is not None and len(engine_baseline.topology.seats) > 0,
             f"{len(engine_baseline.topology.seats) if engine_baseline.topology else 0} seats")
    except Exception as e:
        test("Baseline calibrate()", False, traceback.format_exc())
else:
    test("Baseline calibrate() — SKIPPED (no video)", False, "no calibration frames")

# Structure AI calibration
section("T6: ClassroomEngine Calibration (Structure AI)")
if calib_frames:
    try:
        engine_struct = ClassroomEngine()
        engine_struct.calibrate(calib_frames, 1920, 1080, config=TopologyConfig(mode="structure_ai"))
        test("Structure AI calibrate() completes", engine_struct._calibrated)
        test("Structure AI discovers 12 seats",
             engine_struct.topology is not None and len(engine_struct.topology.seats) == 12,
             f"{len(engine_struct.topology.seats) if engine_struct.topology else 0} seats")
        # Verify correct row split: R1=6, R2=6
        if engine_struct.topology:
            r1_count = sum(1 for s in engine_struct.topology.seats if s.row == 0)
            r2_count = sum(1 for s in engine_struct.topology.seats if s.row == 1)
            test("R1 (FRONT) has 6 seats", r1_count == 6, f"R1={r1_count}")
            test("R2 (BACK) has 6 seats", r2_count == 6, f"R2={r2_count}")
        test("Structure AI source is 'structure_ai'",
             engine_struct.topology.source == "structure_ai")
        test("Stabilizer locked", engine_struct.stabilizer._locked)
        test("Seat label maps populated",
             len(engine_struct._seat_label_to_id) == 12 and len(engine_struct._seat_id_to_label) == 12,
             f"label→id: {len(engine_struct._seat_label_to_id)}, id→label: {len(engine_struct._seat_id_to_label)}")
    except Exception as e:
        test("Structure AI calibrate()", False, traceback.format_exc())
else:
    test("Structure AI calibrate() — SKIPPED (no video)", False, "no calibration frames")


# ═══════════════════════════════════════════════════════════════════════════════
# T7: process_single_frame produces valid output
# ═══════════════════════════════════════════════════════════════════════════════
section("T7: process_single_frame Output Validation")
if calib_frames and test_frames and engine_struct is not None and getattr(engine_struct, "_calibrated", False):
    try:
        frame_30 = test_frames.get(30)
        if frame_30 is not None:
            faces = engine_struct.process_single_frame(frame_30, 1920, 1080, persist=True)
            test("process_single_frame returns list", isinstance(faces, list))
            test("Detections found", len(faces) > 0, f"{len(faces)} detections")
            
            if faces:
                f0 = faces[0]
                test("Detection has 'seat_label'", "seat_label" in f0, f"seat_label={f0.get('seat_label')}")
                test("Detection has 'status'", "status" in f0, f"status={f0.get('status')}")
                test("Detection has 'bbox'", "bbox" in f0 and len(f0["bbox"]) == 4)
                test("Detection has 'topology' dict", "topology" in f0 and isinstance(f0["topology"], dict))
                test("Detection has 'identity' dict", "identity" in f0 and isinstance(f0["identity"], dict))
                
                # Count OK vs UNASSIGNED
                ok_count = sum(1 for f in faces if f.get("status") == "OK")
                unassigned_count = sum(1 for f in faces if f.get("status") == "UNASSIGNED")
                test("Some detections assigned to seats", ok_count > 0,
                     f"{ok_count} OK, {unassigned_count} UNASSIGNED")
                
                # Check seat labels are valid R{r}C{c}
                assigned_labels = [f["seat_label"] for f in faces if f.get("status") == "OK"]
                valid_pattern = all(l.startswith("R") and "C" in l for l in assigned_labels)
                test("Seat labels follow R{r}C{c} pattern", valid_pattern,
                     f"labels: {assigned_labels}")
        else:
            test("process_single_frame — SKIPPED (no frame 30)", False)
    except Exception as e:
        test("process_single_frame", False, traceback.format_exc())
else:
    test("process_single_frame — SKIPPED (prerequisites)", False, "engine not calibrated")


# ═══════════════════════════════════════════════════════════════════════════════
# T8: Ground Truth Alignment
# ═══════════════════════════════════════════════════════════════════════════════
section("T8: Ground Truth Alignment")
GT_PATH = os.path.join(ROOT, "backend", "tests", "ground_truth_classroom.json")
gt_exists = os.path.isfile(GT_PATH)
test("Ground truth file exists", gt_exists, GT_PATH if gt_exists else "NOT FOUND")

if gt_exists:
    with open(GT_PATH, "r", encoding="utf-8") as f:
        gt = json.load(f)
    
    gt_seats = gt.get("total_physical_seats", 0)
    gt_rows = len(gt.get("rows", {}))
    gt_frame_count = len(gt.get("frames", {}))
    
    test("GT has 12 total seats", gt_seats == 12, f"gt_seats={gt_seats}")
    test("GT has 2 rows", gt_rows == 2, f"gt_rows={gt_rows}")
    test("GT has annotated frames", gt_frame_count >= 3, f"{gt_frame_count} annotated frames")
    
    # Verify seat labels match between GT and Structure AI topology
    gt_seat_labels = set()
    for row_data in gt.get("rows", {}).values():
        gt_seat_labels.update(row_data.get("seats", []))
    
    if engine_struct is not None and getattr(engine_struct, "_calibrated", False) and engine_struct.topology:
        struct_labels = set(s.label for s in engine_struct.topology.seats)
        overlap = gt_seat_labels & struct_labels
        missing_from_struct = gt_seat_labels - struct_labels
        extra_in_struct = struct_labels - gt_seat_labels
        test("GT seat labels covered by Structure AI",
             len(missing_from_struct) == 0,
             f"overlap={len(overlap)}, missing={missing_from_struct or 'none'}, extra={extra_in_struct or 'none'}")
    
    # Check R1 has 6, R2 has 6
    r1_data = gt["rows"].get("R1", {})
    r2_data = gt["rows"].get("R2", {})
    test("R1 (FRONT) has 6 seats", r1_data.get("count") == 6, f"R1 count={r1_data.get('count')}")
    test("R2 (BACK) has 6 seats", r2_data.get("count") == 6, f"R2 count={r2_data.get('count')}")


# ═══════════════════════════════════════════════════════════════════════════════
# T9: Benchmark Report Integrity
# ═══════════════════════════════════════════════════════════════════════════════
section("T9: Benchmark Report Integrity")
REPORT_PATH = os.path.join(ROOT, "outputs", "dual_benchmark", "benchmark_comparison_report.json")
report_exists = os.path.isfile(REPORT_PATH)
test("Benchmark report exists", report_exists)

if report_exists:
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)
    
    test("Report has topology_benchmark", "topology_benchmark" in report)
    test("Report has occupancy_benchmark", "occupancy_benchmark" in report)
    
    topo_bm = report.get("topology_benchmark", {})
    test("GT seats in report = 12", topo_bm.get("gt_seats") == 12)
    test("Structure AI seat count error = 0", 
         topo_bm.get("structure_ai", {}).get("seat_count_error") == 0,
         f"error={topo_bm.get('structure_ai', {}).get('seat_count_error')}")
    
    occ_bm = report.get("occupancy_benchmark", {})
    struct_occ = occ_bm.get("structure_ai", {})
    baseline_occ = occ_bm.get("baseline_v2_2", {})
    
    test("Structure AI seat accuracy ≥ 90%", struct_occ.get("seat_accuracy_pct", 0) >= 90.0,
         f"{struct_occ.get('seat_accuracy_pct')}%")
    test("Structure AI beats baseline seat accuracy",
         struct_occ.get("seat_accuracy_pct", 0) >= baseline_occ.get("seat_accuracy_pct", 0),
         f"struct={struct_occ.get('seat_accuracy_pct')}% vs baseline={baseline_occ.get('seat_accuracy_pct')}%")
    test("Structure AI zero double-bookings", struct_occ.get("double_bookings") == 0)
    test("Structure AI decoupled from people",
         topo_bm.get("structure_ai", {}).get("decoupled_from_people") is True)


# ═══════════════════════════════════════════════════════════════════════════════
# T10: Cross-Frame Topology Invariance
# ═══════════════════════════════════════════════════════════════════════════════
section("T10: Cross-Frame Topology Invariance")
if engine_struct is not None and getattr(engine_struct, "_calibrated", False) and engine_struct.topology and test_frames:
    try:
        initial_seats = [(s.label, s.anchor, s.bbox) for s in engine_struct.topology.seats]
        initial_count = len(initial_seats)
        
        # Process multiple frames and verify topology doesn't change
        for f_idx in sorted(test_frames.keys()):
            frame = test_frames[f_idx]
            engine_struct.process_single_frame(frame, 1920, 1080, persist=True)
        
        current_seats = [(s.label, s.anchor, s.bbox) for s in engine_struct.topology.seats]
        test("Seat count invariant across frames",
             len(current_seats) == initial_count,
             f"before={initial_count}, after={len(current_seats)}")
        
        labels_match = all(a[0] == b[0] for a, b in zip(initial_seats, current_seats))
        test("Seat labels invariant", labels_match)
        
        anchors_match = all(a[1] == b[1] for a, b in zip(initial_seats, current_seats))
        test("Seat anchors invariant (topology locked)", anchors_match)
        
        test("Stabilizer still locked after processing", engine_struct.stabilizer._locked)
    except Exception as e:
        test("Cross-frame invariance", False, traceback.format_exc())
else:
    test("Cross-frame invariance — SKIPPED", False, "prerequisites not met")


# ═══════════════════════════════════════════════════════════════════════════════
# T11: Universal Scene Representation (USR) & Modular Pipeline Validation
# ═══════════════════════════════════════════════════════════════════════════════
section("T11: Universal Scene Representation & Modular Pipeline")

try:
    from core.scene import SceneState, EntityState, SceneRelationship
    from core.pipeline import DetectionStage, AssociationStage, AnalysisStage, OutputStage, PipelineScheduler
    test("Import USR scene models", True)
    test("Import modular pipeline stages", True)

    if engine_struct is not None and engine_struct.last_scene_state is not None:
        ss = engine_struct.last_scene_state
        test("Engine produces SceneState", isinstance(ss, SceneState), f"type={type(ss).__name__}")
        test("SceneState contains EntityState items", len(ss.entities) > 0 and isinstance(ss.entities[0], EntityState), f"entities={len(ss.entities)}")
        test("SceneState has accurate seat counts", ss.total_seats == 12 and ss.occupied_seats > 0, f"total={ss.total_seats}, occ={ss.occupied_seats}")
        test("SceneState inferred relationships", len(ss.relationships) > 0, f"relations={len(ss.relationships)}")
        
        # Test serialization
        ss_dict = ss.to_dict()
        test("SceneState serializes to dict", isinstance(ss_dict, dict) and "entities" in ss_dict and "relationships" in ss_dict)
    else:
        test("Engine produces SceneState", False, "engine_struct or last_scene_state is None")
except Exception as e:
    test("USR & Pipeline verification", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  VERIFICATION SUMMARY")
print(f"{'='*70}")
passed = sum(1 for r in results if r["passed"])
failed = sum(1 for r in results if not r["passed"])
total = len(results)
pct = (passed / total * 100) if total > 0 else 0

print(f"  Total Tests:  {total}")
print(f"  Passed:       {passed} ({'%.1f' % pct}%)")
print(f"  Failed:       {failed}")

if failed > 0:
    print(f"\n  Failed Tests:")
    for r in results:
        if not r["passed"]:
            print(f"    ❌ {r['name']}" + (f" — {r['detail']}" if r['detail'] else ""))

status = "ALL TESTS PASSED ✅" if failed == 0 else f"{failed} TEST(S) FAILED ❌"
print(f"\n  {status}")
print(f"{'='*70}\n")

# Exit with appropriate code
sys.exit(0 if failed == 0 else 1)
