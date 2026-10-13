# -*- coding: utf-8 -*-
"""
Tests for V2.1 Topology and SeatAssociator.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from core.topology import PhysicalSeat, ClassroomTopology, TopologyConfig, TopologyDiscovery, PersonTrack
from core.seat_associator import SeatAssociator, RowClassifier, SeatAssignment

def test_topology_and_associator():
    print("Testing Topology & SeatAssociator...")

    # 1. Create simulated tracks for 2 rows of 6 students
    # Front row: Y ~ 550, X in [300, 500, 700, 900, 1100, 1300]
    # Back row:  Y ~ 450, X in [320, 520, 720, 920, 1120, 1320]
    discovery = TopologyDiscovery(frame_size=(1920, 1080))

    for f in range(20):
        tracks = []
        # Front row (Row 0 -> R1)
        for i, x in enumerate([300, 500, 700, 900, 1100, 1300]):
            tid = 1 + i
            tracks.append(PersonTrack(
                track_id=tid,
                bbox=[x - 50, 480, x + 50, 750],
                face_bbox=[x - 20, 490, x + 20, 550],
                head_point=(float(x), 520.0),
                bottom_point=(float(x), 750.0),
                confidence=0.9,
                frame_idx=f,
            ))
        # Back row (Row 1 -> R2)
        for i, x in enumerate([320, 520, 720, 920, 1120, 1320]):
            tid = 10 + i
            tracks.append(PersonTrack(
                track_id=tid,
                bbox=[x - 40, 400, x + 40, 620],
                face_bbox=[x - 15, 410, x + 15, 460],
                head_point=(float(x), 435.0),
                bottom_point=(float(x), 620.0),
                confidence=0.85,
                frame_idx=f,
            ))
        discovery.accumulate(tracks)

    topo = discovery.estimate(min_observations=5)
    print(f"Topology discovered: rows={topo.rows}, cols_per_row={topo.cols_per_row}, seats={len(topo.seats)}")
    assert topo.rows == 2, f"Expected 2 rows, got {topo.rows}"
    assert topo.cols_per_row == [6, 6], f"Expected [6, 6], got {topo.cols_per_row}"
    assert len(topo.seats) == 12, f"Expected 12 seats, got {len(topo.seats)}"
    
    # Assert V2.2 explicit row semantics
    assert len(topo.row_geometries) == 2, f"Expected 2 row geometries, got {len(topo.row_geometries)}"
    assert topo.row_geometries[0].semantic == "FRONT", f"Expected FRONT, got {topo.row_geometries[0].semantic}"
    assert topo.row_geometries[1].semantic == "BACK", f"Expected BACK, got {topo.row_geometries[1].semantic}"
    
    # Assert V2.2 seat spatial regions
    s0 = topo.seats[0]
    assert s0.expected_head_region is not None, "Expected expected_head_region on seat"
    assert s0.column_boundary_left is not None, "Expected column_boundary_left on seat"
    print("V2.2 Topology semantics & region bounds test PASSED.")

    # 2. Test Associator
    associator = SeatAssociator()
    # Test frame 0
    assignments = associator.associate(tracks, topo, frame_idx=0)
    assert len(assignments) == 12
    for a in assignments:
        assert a.status == "OK", f"Expected OK, got {a.status} for track {a.track_id}"
        assert a.row_match is True
        assert a.seat_confidence > 0.60, f"Expected seat_confidence > 0.60, got {a.seat_confidence}"
        assert a.match_type in ("EXACT_REGION", "OVERLAP_REGION", "NEAREST_FALLBACK"), f"Unexpected match_type {a.match_type}"
    print("Frame 0 association test PASSED.")

    # 3. Test Ghost Hold: drop track 1 (front row seat 1)
    tracks_missing_1 = [t for t in tracks if t.track_id != 1]
    assignments_ghost = associator.associate(tracks_missing_1, topo, frame_idx=1)
    ghosts = [a for a in assignments_ghost if a.status == "GHOST"]
    assert len(ghosts) == 1, f"Expected 1 ghost, got {len(ghosts)}"
    assert ghosts[0].seat_label == "R1C1", f"Expected R1C1 ghost, got {ghosts[0].seat_label}"
    assert ghosts[0].match_type == "GHOST"
    print("Ghost-hold test PASSED.")

    # 4. Test Unassigned (track far away from any seat)
    far_track = PersonTrack(
        track_id=99,
        bbox=[50, 800, 100, 950],
        face_bbox=[50, 800, 100, 850],
        head_point=(75.0, 825.0),
        bottom_point=(75.0, 950.0),
        confidence=0.9,
        frame_idx=2,
    )
    assignments_unassigned = associator.associate(tracks + [far_track], topo, frame_idx=2)
    unassigned = [a for a in assignments_unassigned if a.status == "UNASSIGNED"]
    assert len(unassigned) >= 1, f"Expected unassigned, got {len(unassigned)}"
    assert any(a.track_id == 99 for a in unassigned), "Track 99 should be UNASSIGNED"
    assert unassigned[0].seat_confidence == 0.0, f"Unassigned seat_confidence should be 0.0"
    print("UNASSIGNED test PASSED.")

    # 5. Test No Double-Booking
    seat_labels = [a.seat_label for a in assignments if a.status != "UNASSIGNED"]
    assert len(seat_labels) == len(set(seat_labels)), "Double-booking detected!"
    print("No double-booking test PASSED.")
    print("All tests in test_v2_association.py PASSED!\n")

if __name__ == "__main__":
    test_topology_and_associator()
