# -*- coding: utf-8 -*-
"""
SmartClass AI — Real Classroom Video Tester (Automated/Headless)
=================================================================
This version runs without UI for automated testing and CI/CD.
Processes first 300 frames for quick validation.
"""
import sys, os, io, time, cv2, numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE         = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH    = os.path.dirname(HERE)
BACKEND_PATH = os.path.join(ROOT_PATH, "backend")
sys.path.insert(0, BACKEND_PATH)

DEFAULT_VIDEO = os.path.join(ROOT_PATH, "test video", "test video - real classrom.mp4")
OUTPUT_REPORT = os.path.join(ROOT_PATH, "test_results_auto.txt")

MAX_SEATS    = 15
YOLO_CONF_OV = 0.30
TEST_FRAMES  = 60  # Process only first 60 frames for quick test

from core.engine import ClassroomEngine

def _iou(a, b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    iw=max(0,min(ax2,bx2)-max(ax1,bx1)); ih=max(0,min(ay2,by2)-max(ay1,by1))
    inter=iw*ih
    return inter/max(1,(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter)

def suppress_overlapping_faces(faces, iou_thresh=0.45):
    if len(faces)<=1: return faces
    faces=sorted(faces, key=lambda f:f.get("confidence",0.0), reverse=True)
    keep=[]; sup=set()
    for i,fi in enumerate(faces):
        if i in sup: continue
        keep.append(fi)
        bi=fi.get("person_bbox",fi["bbox"])
        for j,fj in enumerate(faces):
            if j<=i or j in sup: continue
            if _iou(bi,fj.get("person_bbox",fj["bbox"]))>iou_thresh:
                sup.add(j)
    return keep


def run_test(video_path: str):
    print("="*70)
    print("SmartClass AI — Automated Test (Headless)")
    print("="*70)
    if not os.path.isfile(video_path):
        print(f"ERROR: Not found: {video_path}")
        return False

    import core.config as _cfg
    _cfg.YOLO_CONF = YOLO_CONF_OV

    print("  Pre-loading AI models...", flush=True)
    t_warm = time.time()
    try:
        from core.detector import _get_yolo, _get_mp_detector, analyze_face_crop
        _yolo = _get_yolo()
        _get_mp_detector()
        _blank = np.zeros((320, 320, 3), dtype=np.uint8)
        _yolo.predict(_blank, conf=0.9, verbose=False)
        analyze_face_crop(_blank)
        print(f"  Models ready in {time.time()-t_warm:.1f}s\n", flush=True)
    except Exception as e:
        print(f"  WARNING: Model pre-load issue: {e}\n", flush=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open: {video_path}")
        return False

    fps_src  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur      = total_fr / fps_src

    frames_to_process = min(TEST_FRAMES, total_fr)
    
    print(f"  Video: {os.path.basename(video_path)}")
    print(f"  {W}x{H}  {fps_src:.1f}fps  {dur:.0f}s total")
    print(f"  Testing first {frames_to_process} frames\n")

    engine = ClassroomEngine(max_seats=MAX_SEATS)
    
    frame_times = []
    total_dets = 0
    shown = 0
    t0 = time.time()
    
    print("  Processing frames...")
    for frame_idx in range(frames_to_process):
        ret, frame = cap.read()
        if not ret:
            print(f"\n  Video ended at frame {frame_idx}")
            break
        
        shown += 1
        t_frame = time.perf_counter()
        
        faces = engine.process_single_frame(frame, W, H, persist=True)
        faces = suppress_overlapping_faces(faces)
        total_dets += len(faces)
        
        frame_times.append((time.perf_counter() - t_frame) * 1000)
        
        if shown % 30 == 0:
            avg_ms = sum(frame_times[-30:]) / min(30, len(frame_times))
            pct = shown / frames_to_process * 100
            filled = int(28 * pct / 100)
            bar = "#" * filled + "-" * (28 - filled)
            print(f"  [{bar}] {pct:5.1f}%  fr={shown}  det={len(faces)}  avg={avg_ms:.0f}ms", 
                  end="\r", flush=True)
    
    cap.release()
    total_time = time.time() - t0
    avg_ms = sum(frame_times) / len(frame_times) if frame_times else 0
    fps_achieved = shown / total_time if total_time > 0 else 0
    
    print("\n\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    print(f"  Frames processed : {shown}")
    print(f"  Total detections : {total_dets}")
    print(f"  Avg per frame    : {total_dets/max(1,shown):.1f}")
    print(f"  Avg AI latency   : {avg_ms:.1f}ms")
    print(f"  Processing FPS   : {fps_achieved:.1f}")
    print(f"  Total time       : {total_time:.2f}s")
    
    # Student statistics
    success = True
    if engine.student_states:
        print(f"\n  Students tracked : {len(engine.student_states)}")
        print()
        header = f"  {'Seat':<6} {'Frames':>8} {'Avg':>7} {'Peak':>7} {'Low':>6}  {'Activity':<30}"
        sep = "  " + "-" * 68
        print(header)
        print(sep)
        
        for sid, st in sorted(engine.student_states.items()):
            act, note = st.reconciled_activity()
            label = act + (f" ({note})" if note else "")
            row = f"  {sid:<6} {st.frame_count:>8} {st.avg:>6.1f}%  {st.peak:>6.1f}%  {st.low:>5.1f}%  {label:<30}"
            print(row)
        
        avgs = [st.avg for st in engine.student_states.values()]
        class_avg = sum(avgs) / len(avgs)
        print(sep)
        print(f"\n  Class average    : {class_avg:.1f}%")
        
        # Validation checks
        print("\n  Validation:")
        if total_dets < shown * 0.5:  # Expect at least 0.5 detections per frame
            print(f"    ⚠ Low detection rate: {total_dets/shown:.2f} per frame")
            success = False
        else:
            print(f"    ✓ Detection rate OK: {total_dets/shown:.2f} per frame")
        
        if len(engine.student_states) < 3:
            print(f"    ⚠ Few students tracked: {len(engine.student_states)}")
            success = False
        else:
            print(f"    ✓ Students tracked: {len(engine.student_states)}")
        
        if avg_ms > 1000:  # More than 1 second per frame is too slow
            print(f"    ⚠ Slow processing: {avg_ms:.0f}ms/frame")
            success = False
        else:
            print(f"    ✓ Processing speed OK: {avg_ms:.0f}ms/frame")
        
    else:
        print("\n  ⚠ No students tracked!")
        success = False
    
    print("="*70)
    
    # Write report
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("SmartClass AI — Automated Test Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Video: {os.path.basename(video_path)}\n")
        f.write(f"Resolution: {W}x{H} @ {fps_src:.1f}fps\n")
        f.write(f"Frames processed: {shown}\n")
        f.write(f"Total time: {total_time:.2f}s\n")
        f.write(f"Avg AI latency: {avg_ms:.1f}ms\n")
        f.write(f"Processing FPS: {fps_achieved:.1f}\n\n")
        
        if engine.student_states:
            f.write(f"Students tracked: {len(engine.student_states)}\n\n")
            f.write(header + "\n")
            f.write(sep + "\n")
            for sid, st in sorted(engine.student_states.items()):
                act, note = st.reconciled_activity()
                label = act + (f" ({note})" if note else "")
                row = f"  {sid:<6} {st.frame_count:>8} {st.avg:>6.1f}%  {st.peak:>6.1f}%  {st.low:>5.1f}%  {label:<30}"
                f.write(row + "\n")
            f.write(sep + "\n")
            avgs = [st.avg for st in engine.student_states.values()]
            f.write(f"\nClass average: {sum(avgs)/len(avgs):.1f}%\n")
    
    print(f"\n  Report saved: {OUTPUT_REPORT}\n")
    
    return success


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO
    success = run_test(os.path.abspath(video))
    sys.exit(0 if success else 1)
