import os
import sys
import cv2
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from core.engine import ClassroomEngine
from core.topology import TopologyDiscovery, TopologyConfig

def run_test(vname, label, test_frames=80, save_annotated=True):
    video_path = os.path.join(ROOT, "test video", vname)
    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    calib_frames = []
    for _ in range(60):
        ret, f = cap.read()
        if not ret:
            break
        calib_frames.append(f)
    cap.release()

    engine = ClassroomEngine()
    engine.calibrate(calib_frames, W, H)
    topo = engine.topology

    print(f"\n{'='*70}")
    print(f"[{label}] Video: {vname} ({W}x{H}, {total_frames} total frames)")
    print(f"Discovered: {len(topo.seats)} seats across {topo.rows} rows (Cols: {topo.cols_per_row}, slant={np.degrees(topo.row_slant_theta):.1f} deg)")
    for rg in topo.row_geometries:
        print(f"  Row {rg.label} ({rg.semantic}): y_median={rg.y_median:.1f}, band=[{rg.y_min:.1f}, {rg.y_max:.1f}]")
    for s in topo.seats:
        print(f"  Seat {s.label} (Row {s.row+1}, Col {s.col+1}): anchor=({s.anchor[0]:.1f}, {s.anchor[1]:.1f}), bbox={[round(x,1) for x in s.bbox]}")

    # Process test_frames
    cap = cv2.VideoCapture(video_path)
    unassigned_count = 0
    assigned_count = 0
    ghost_count = 0
    track_to_seats = {}  # track_id -> list of assigned seats
    last_frame_vis = None

    for f_idx in range(test_frames):
        ret, frame = cap.read()
        if not ret:
            break
        faces = engine.process_single_frame(frame, W, H, persist=True)
        
        vis = frame.copy() if (save_annotated and f_idx == test_frames - 1) else None

        for fc in faces:
            cell = fc.get("gridCell", "")
            tid = fc.get("id")
            if tid not in track_to_seats:
                track_to_seats[tid] = []
            track_to_seats[tid].append(cell)

            if cell.startswith("UNASSIGNED") or cell == "UNASSIGNED":
                unassigned_count += 1
            elif "GHOST" in str(fc.get("activityNote", "")):
                ghost_count += 1
            else:
                assigned_count += 1

        if vis is not None:
            # Draw seats
            for s in topo.seats:
                bx = s.bbox
                col = (0, 255, 120) if s.row == 0 else (255, 180, 0) if s.row == 1 else (200, 100, 255)
                cv2.rectangle(vis, (bx[0], bx[1]), (bx[2], bx[3]), col, 2)
                cv2.circle(vis, (int(s.anchor[0]), int(s.anchor[1])), 6, (0, 0, 255), -1)
                cv2.putText(vis, s.label, (int(s.anchor[0]) - 25, int(s.anchor[1]) - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # Draw detected faces
            for fc in faces:
                bb = fc.get("bbox")
                lbl = fc.get("gridCell", "")
                cv2.rectangle(vis, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 255), 2)
                cv2.putText(vis, f"Occupant->{lbl}", (bb[0], max(20, bb[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            last_frame_vis = vis

    cap.release()

    total_detections = assigned_count + unassigned_count
    assign_rate = (assigned_count / total_detections * 100.0) if total_detections > 0 else 0.0

    # Calculate stability (number of switches per track)
    switches = 0
    for tid, s_list in track_to_seats.items():
        valid_seats = [s for s in s_list if s and not s.startswith("UNASSIGNED")]
        for i in range(len(valid_seats) - 1):
            if valid_seats[i] != valid_seats[i+1]:
                switches += 1

    print(f"\n--- Association & Stability Metrics ({test_frames} Frames) ---")
    print(f"  Assigned: {assigned_count}/{total_detections} ({assign_rate:.1f}%)")
    print(f"  Unassigned: {unassigned_count}")
    print(f"  Ghost Hold: {ghost_count}")
    print(f"  Track-to-Seat Switches: {switches} across {len(track_to_seats)} tracks")
    print(f"  Tracked Students in Session: {len(engine.student_states)}")
    for sid, st in sorted(engine.student_states.items()):
        lbl = engine._seat_id_to_label.get(sid, f"S{sid}")
        print(f"    Seat {lbl:<6} (ID {sid:>2}): avgAttn={st.avg:>5.1f}%, peak={st.peak:>5.1f}%, act={st.reconciled_activity()[0]}")

    if last_frame_vis is not None:
        out_img_name = f"eval_{label.lower().replace(' ', '_')}.jpg"
        out_path = os.path.join(ROOT, "outputs", out_img_name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, last_frame_vis)
        print(f"  Annotated visual saved: {out_path}")

    return {
        "video": vname,
        "rows": topo.rows,
        "total_seats": len(topo.seats),
        "cols_per_row": topo.cols_per_row,
        "assigned": assigned_count,
        "unassigned": unassigned_count,
        "assign_rate": assign_rate,
        "switches": switches,
    }

res1 = run_test("Student Sitting in the Class  Royalty Free Video  No Copyright Video_1080p.mp4", "VIDEO 1", 80)
res2 = run_test("test video - real classrom.mp4", "VIDEO 2", 80)

print(f"\n{'='*70}")
print("SUMMARY RESULTS:")
print(f"VIDEO 1: {res1['total_seats']} seats ({res1['rows']} rows, cols {res1['cols_per_row']}) | Assign Rate: {res1['assign_rate']:.1f}% | Unassigned: {res1['unassigned']}")
print(f"VIDEO 2: {res2['total_seats']} seats ({res2['rows']} rows, cols {res2['cols_per_row']}) | Assign Rate: {res2['assign_rate']:.1f}% | Unassigned: {res2['unassigned']}")
