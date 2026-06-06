# -*- coding: utf-8 -*-
"""
SmartClass AI — Real Classroom Video Tester
============================================
Pipeline:
  Main thread  — reads and DISPLAYS every frame at exact video fps.
                 Never waits for AI. Uses latest AI result available.
  AI thread    — processes frames in background, updates shared state.

This guarantees:
  - Zero wobble / frame skip (display is never blocked)
  - Overlays are stable (always drawn from last AI result)
  - Speed control works correctly

Keys: D=toggle details  SPACE=pause  +/-=speed  Q=quit
"""
import sys, os, io, time, threading, cv2, numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE         = os.path.dirname(os.path.abspath(__file__))
BACKEND_PATH = os.path.join(HERE, "backend")
sys.path.insert(0, BACKEND_PATH)

DEFAULT_VIDEO = os.path.join(HERE, "test video", "test video - real classrom.mp4")
OUTPUT_VIDEO  = os.path.join(HERE, "real_classroom_annotated.mp4")
OUTPUT_REPORT = os.path.join(HERE, "real_classroom_results.txt")

MAX_SEATS    = 15
YOLO_CONF_OV = 0.30
IOU_SUPPRESS = 0.45
DISPLAY_W    = 1280
AI_EVERY     = 2      # run full AI every N frames; overlay interpolated between calls

from core.engine import ClassroomEngine
from core.visualizer import (
    alpha_rect, attn_bar, corner_brackets,
    smooth, stable_label, draw_global_hud,
    C, ACT_COLOR, ACT_ICON,
)


# ─────────────────────────────────────────────────────────────────────────────
# Overlay renderers
# ─────────────────────────────────────────────────────────────────────────────
def draw_compact(frame, face):
    x1, y1, x2, y2 = face["bbox"]
    sid  = face.get("id", "?")
    act  = stable_label(sid, "act",
           face.get("reconciledActivity", face.get("activity", "neutral")), hold=5)
    attn = smooth(sid, "attn", face.get("attentionScore", 0), 0.10)
    col  = ACT_COLOR.get(act, C["white"])
    icon = ACT_ICON.get(act, act.upper())

    cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
    corner_brackets(frame, x1, y1, x2, y2, col,
                    length=min(18,max(5,(x2-x1)//5),max(5,(y2-y1)//5)), thickness=2)

    badge = f"S{sid} {icon} {attn:.0f}%"
    bw = max(65, len(badge)*8+10); bh = 18
    by1 = max(0, y1-bh-2); by2 = max(bh, y1-2)
    alpha_rect(frame, x1, by1, x1+bw, by2, col, alpha=0.85)
    cv2.putText(frame, badge, (x1+3, by2-3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, C["black"], 1, cv2.LINE_AA)
    attn_bar(frame, x1, y2+2, x2-x1, 5, attn)

    if face.get("phoneDetected"):
        alpha_rect(frame, x1, max(0,by1-16), x2, max(0,by1), C["red"], alpha=0.90)
        cv2.putText(frame, "PHONE!", (x1+3, max(9,by1-3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C["white"], 2, cv2.LINE_AA)


def draw_full(frame, face):
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = face["bbox"]
    sid    = face.get("id", "?")
    act    = stable_label(sid, "act",
             face.get("reconciledActivity", face.get("activity","neutral")), hold=5)
    emo    = stable_label(sid, "emo", face.get("emotion","neutral"), hold=6)
    attn   = smooth(sid, "attn", face.get("attentionScore", 0),          0.10)
    avg_a  = smooth(sid, "avg",  face.get("avgAttention30", attn),       0.06)
    eye    = smooth(sid, "eye",  face.get("eyeOpenness", 1.0),           0.08)
    yaw    = smooth(sid, "yaw",  face.get("headPose",{}).get("yaw",0),   0.10)
    pitch  = smooth(sid, "pitch",face.get("headPose",{}).get("pitch",0), 0.10)
    phone  = face.get("phoneDetected", False)
    laptop = face.get("laptopDetected", False)
    gc     = face.get("gridCell", "")
    note   = face.get("activityNote", "")
    factors= face.get("attentionFactors", {})
    col    = ACT_COLOR.get(act, C["white"])
    icon   = ACT_ICON.get(act, act.upper())

    cv2.rectangle(frame, (x1,y1), (x2,y2), col, 1)
    corner_brackets(frame, x1, y1, x2, y2, col,
                    length=min(15,max(4,(x2-x1)//5),max(4,(y2-y1)//5)))

    badge = f"S{sid}" + (f" {gc}" if gc else "")
    bw_b  = max(46, len(badge)*8+8)
    alpha_rect(frame, x1, max(0,y1-20), x1+bw_b, max(0,y1-2), col, alpha=0.85)
    cv2.putText(frame, badge, (x1+3, max(9,y1-5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, C["black"], 1, cv2.LINE_AA)

    pw, ph = 148, 108
    px = min(x2+3, W-pw-2)
    py = min(max(0,y1), H-ph-2)
    alpha_rect(frame, px, py, px+pw, py+ph, C["panel"], alpha=0.80)
    cv2.rectangle(frame, (px,py), (px+pw,py+ph), (55,55,55), 1)

    cv2.putText(frame, icon, (px+4,py+13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{attn:.0f}%", (px+pw-38,py+13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1, cv2.LINE_AA)
    attn_bar(frame, px+4, py+17, pw-8, 5, attn)

    cv2.putText(frame, f"Avg {avg_a:.0f}%", (px+4,py+35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, C["white"], 1, cv2.LINE_AA)
    attn_bar(frame, px+4, py+39, pw-8, 4, avg_a)

    emo_col = C["green"] if emo=="happy" else C["yellow"] if emo=="neutral" else C["orange"]
    cv2.putText(frame, f"Emo:{emo}", (px+4,py+54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, emo_col, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Y:{yaw:+.0f} P:{pitch:+.0f}", (px+4,py+65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.26, (155,155,155), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Eye:{eye:.2f}", (px+4,py+76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.26, (155,155,155), 1, cv2.LINE_AA)
    attn_bar(frame, px+50, py+71, pw-55, 4, eye*100)

    if factors:
        cv2.putText(frame,
            f"HP:{factors.get('head_pose',1):.2f} Mo:{factors.get('motion_stability',1):.2f}",
            (px+4,py+88), cv2.FONT_HERSHEY_SIMPLEX, 0.22, (105,105,105), 1, cv2.LINE_AA)
    if note:
        cv2.putText(frame, note, (px+4,py+100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.20, (85,165,85), 1, cv2.LINE_AA)

    if phone:
        alpha_rect(frame, x1, max(0,y1-40), x2, max(0,y1-22), C["red"], alpha=0.90)
        cv2.putText(frame, "!! PHONE !!", (x1+3, max(9,y1-26)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, C["white"], 2, cv2.LINE_AA)
    if laptop:
        alpha_rect(frame, x1, y2+2, x1+58, y2+14, (30,60,60), alpha=0.85)
        cv2.putText(frame, "LAPTOP", (x1+3, y2+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, C["cyan"], 1, cv2.LINE_AA)

    bw2 = x2-x1
    fill = int(bw2*min(attn,100)/100)
    cv2.rectangle(frame, (x1,y2+2), (x2,y2+5), (45,45,45), -1)
    if fill > 0:
        cv2.rectangle(frame, (x1,y2+2), (x1+fill,y2+5), col, -1)


# ─────────────────────────────────────────────────────────────────────────────
# Overlap suppressor
# ─────────────────────────────────────────────────────────────────────────────
def _iou(a, b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    iw=max(0,min(ax2,bx2)-max(ax1,bx1)); ih=max(0,min(ay2,by2)-max(ay1,by1))
    inter=iw*ih
    return inter/max(1,(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter)

def suppress_overlapping_faces(faces):
    if len(faces)<=1: return faces
    faces=sorted(faces, key=lambda f:f.get("confidence",0.0), reverse=True)
    keep=[]; sup=set()
    for i,fi in enumerate(faces):
        if i in sup: continue
        keep.append(fi)
        bi=fi.get("person_bbox",fi["bbox"])
        for j,fj in enumerate(faces):
            if j<=i or j in sup: continue
            if _iou(bi,fj.get("person_bbox",fj["bbox"]))>IOU_SUPPRESS:
                sup.add(j)
    return keep


# ─────────────────────────────────────────────────────────────────────────────
# Shared state — with per-seat bbox interpolation
# ─────────────────────────────────────────────────────────────────────────────
class SharedState:
    """
    Holds the latest AI result and exposes interpolated bboxes.

    When AI produces a new result, we record:
      - prev_boxes : the bbox positions at the previous AI result
      - next_boxes : the new bbox positions
      - interp_t   : perf_counter() when the new result arrived
      - ai_period  : how many seconds between AI results (for lerp denominator)

    The display thread calls get_faces(t) with the current time.
    Each face's bbox is linearly interpolated between prev and next so boxes
    glide to their new position over one AI period instead of snapping.
    """
    def __init__(self):
        self._lock      = threading.Lock()
        self._faces     : list  = []          # latest full face dicts from AI
        self._prev_box  : dict  = {}          # seat_id → [x1,y1,x2,y2] (prev AI)
        self._next_box  : dict  = {}          # seat_id → [x1,y1,x2,y2] (next AI)
        self._interp_t  : float = 0.0         # perf_counter at last AI result
        self._ai_period : float = 0.5         # estimated seconds between AI calls
        self.ai_ms      : float = 0.0

    def update(self, faces: list, ai_ms: float):
        with self._lock:
            now = time.perf_counter()
            if self._interp_t > 0:
                self._ai_period = max(0.05, now - self._interp_t)
            self._interp_t = now

            # Promote current next → prev
            self._prev_box = dict(self._next_box)
            # Record new target boxes
            self._next_box = {
                f["id"]: list(f["bbox"]) for f in faces if "bbox" in f
            }
            # For any seat not seen before, seed prev from next (no lerp jump)
            for sid, box in self._next_box.items():
                if sid not in self._prev_box:
                    self._prev_box[sid] = list(box)

            self._faces  = faces
            self.ai_ms   = ai_ms

    def get_faces(self) -> list:
        """Return face dicts with bbox linearly interpolated to current time."""
        with self._lock:
            if not self._faces:
                return []
            now    = time.perf_counter()
            alpha  = min(1.0, (now - self._interp_t) / max(0.001, self._ai_period))
            result = []
            for face in self._faces:
                f = dict(face)
                sid = f.get("id")
                if sid is not None and sid in self._prev_box and sid in self._next_box:
                    p = self._prev_box[sid]
                    n = self._next_box[sid]
                    # Lerp each coord
                    f["bbox"] = [
                        int(p[i] + (n[i] - p[i]) * alpha) for i in range(4)
                    ]
                result.append(f)
            return result

    def get_ai_ms(self) -> float:
        with self._lock:
            return self.ai_ms


# ─────────────────────────────────────────────────────────────────────────────
# AI thread — runs YOLO+MediaPipe on every AI_EVERY-th frame
# ─────────────────────────────────────────────────────────────────────────────
class AIThread(threading.Thread):
    def __init__(self, engine, shared: SharedState, stop_evt: threading.Event,
                 W: int, H: int):
        super().__init__(daemon=True, name="AIThread")
        self.engine  = engine
        self.shared  = shared
        self.stop    = stop_evt
        self.W       = W
        self.H       = H
        # Frames are pushed here by main thread
        self.frame_slot: list = [None]   # [frame] or [None]
        self.frame_evt = threading.Event()

    def submit(self, frame):
        """Main thread calls this to hand a frame to the AI thread."""
        self.frame_slot[0] = frame.copy()
        self.frame_evt.set()

    def run(self):
        times = []
        n     = 0
        while not self.stop.is_set():
            triggered = self.frame_evt.wait(timeout=0.3)
            if not triggered:
                continue
            self.frame_evt.clear()
            frame = self.frame_slot[0]
            if frame is None:
                continue

            n += 1
            if n % AI_EVERY != 0:
                continue   # skip this frame

            t0    = time.perf_counter()
            faces = self.engine.process_single_frame(frame, self.W, self.H, persist=True)
            faces = suppress_overlapping_faces(faces)
            dt    = (time.perf_counter()-t0)*1000

            times.append(dt)
            if len(times) > 30: times.pop(0)
            avg_ms = sum(times)/len(times)

            self.shared.update(faces, avg_ms)


# ─────────────────────────────────────────────────────────────────────────────
# Main — display loop (never blocked by AI)
# ─────────────────────────────────────────────────────────────────────────────
def run(video_path: str):
    print("="*65)
    print("SmartClass AI — Real Classroom (Zero-Block Display)")
    print("="*65)
    if not os.path.isfile(video_path):
        print(f"ERROR: Not found: {video_path}"); return

    import core.config as _cfg
    _cfg.YOLO_CONF = YOLO_CONF_OV

    # ── Pre-warm models BEFORE opening the window ─────────────────────────────
    print("  Pre-loading AI models (one-time, ~10-30s)...", flush=True)
    _t_warm = time.time()
    try:
        from core.detector import _get_yolo, _get_mp_detector, analyze_face_crop
        _yolo_model = _get_yolo()
        _get_mp_detector()
        # Dummy inference — forces XNNPack JIT so first real frame is instant
        _blank = np.zeros((320, 320, 3), dtype=np.uint8)
        _yolo_model.predict(_blank, conf=0.9, verbose=False)
        analyze_face_crop(_blank)
        print(f"  Models ready in {time.time()-_t_warm:.1f}s\n", flush=True)
    except Exception as e:
        print(f"  Model pre-load warning: {e}\n", flush=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open: {video_path}"); return

    fps_src  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur      = total_fr / fps_src
    spf      = 1.0 / fps_src          # seconds per frame (target interval)

    print(f"  {W}x{H}  {fps_src:.1f}fps  {dur:.0f}s  AI every {AI_EVERY} frames\n")
    print(f"  Input : {os.path.basename(video_path)}")
    print(f"  Output: {OUTPUT_VIDEO}")
    print(f"  D=details  SPACE=pause  +/-=speed  Q=quit\n")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps_src, (W, H))

    WIN = "SmartClass AI  [D=details  SPACE=pause  +/-=speed  Q=quit]"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, DISPLAY_W, int(DISPLAY_W*H/W))

    engine   = ClassroomEngine(max_seats=MAX_SEATS)
    shared   = SharedState()
    stop_evt = threading.Event()
    ai_thd   = AIThread(engine, shared, stop_evt, W, H)
    ai_thd.start()

    show_details = True
    paused       = False
    speed_mult   = 1.0
    total_dets   = 0
    shown        = 0
    t0           = time.time()
    fps_ts: list = []   # timestamps of last 30 frames for fps calc

    # Precision frame clock — tracks when next frame should be shown
    t_next = time.perf_counter()

    while True:
        if not paused:
            # ── Pace: sleep until this frame is due ───────────────────────────
            now  = time.perf_counter()
            gap  = t_next - now
            if gap > 0.001:
                time.sleep(gap)
            t_next = time.perf_counter() + spf / max(0.1, speed_mult)

            # ── Read next frame from disk ──────────────────────────────────────
            ret, frame = cap.read()
            if not ret:
                break
            shown += 1

            # ── Give frame to AI thread (non-blocking) ─────────────────────────
            ai_thd.submit(frame)

            # ── Get latest AI result with interpolated bboxes ─────────────────
            faces = shared.get_faces()
            total_dets += len(faces)

            # ── Draw overlays ──────────────────────────────────────────────────
            draw_fn = draw_full if show_details else draw_compact
            for face in faces:
                draw_fn(frame, face)

            # ── FPS calculation ────────────────────────────────────────────────
            fps_ts.append(time.perf_counter())
            if len(fps_ts) > 30: fps_ts.pop(0)
            disp_fps = (len(fps_ts)-1)/max(1e-6, fps_ts[-1]-fps_ts[0]) if len(fps_ts)>1 else 0
            elapsed  = time.time()-t0
            ai_ms    = shared.get_ai_ms()

            # ── HUD ────────────────────────────────────────────────────────────
            mode_str = f"{'DETAIL' if show_details else 'COMPACT'}  {speed_mult:.1f}x  AI:{ai_ms:.0f}ms"
            draw_global_hud(frame, faces, {
                "mode":         mode_str,
                "fps":          disp_fps,
                "elapsed":      elapsed,
                "frame_idx":    shown,
                "total_frames": total_fr,
                "hint":         "D=details  SPACE=pause  +/-=speed  Q=quit",
            })

            # ── Stdout progress ────────────────────────────────────────────────
            if shown % 90 == 0:
                pct    = shown/max(1,total_fr)*100
                filled = int(28*pct/100)
                bar    = "#"*filled+"-"*(28-filled)
                print(f"  [{bar}] {pct:5.1f}%  fr={shown}/{total_fr}"
                      f"  det={len(faces)}  fps={disp_fps:.1f}  AI={ai_ms:.0f}ms  t={elapsed:.0f}s",
                      end="\r", flush=True)

            out.write(frame)
            cv2.imshow(WIN, frame)

        else:
            cv2.waitKey(30)

        # ── Keys ──────────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            print("\n  Stopped by user."); break
        elif key == ord("d"):
            show_details = not show_details
            print(f"\n  [D] → {'DETAIL' if show_details else 'COMPACT'}")
        elif key == ord(" "):
            paused = not paused
            print(f"\n  [SPACE] {'PAUSED' if paused else 'RESUMED'}")
        elif key in (ord("+"), ord("=")):
            speed_mult = min(4.0, speed_mult+0.25)
            print(f"\n  [+] → {speed_mult:.2f}x")
        elif key in (ord("-"), ord("_")):
            speed_mult = max(0.25, speed_mult-0.25)
            print(f"\n  [-] → {speed_mult:.2f}x")

    stop_evt.set()
    ai_thd.join(timeout=5)
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    total_time = time.time()-t0
    print("\n\n"+"="*65)
    print("RESULTS")
    print("="*65)
    print(f"  Frames shown    : {shown}")
    print(f"  Detections      : {total_dets}")
    print(f"  Avg AI latency  : {shared.get_ai_ms():.1f}ms")
    print(f"  Total time      : {total_time:.2f}s")
    print(f"  Output video    : {OUTPUT_VIDEO}")

    lines = [
        "SmartClass AI — Real Classroom\n",
        f"Video      : {os.path.basename(video_path)}\n",
        f"Resolution : {W}x{H} @ {fps_src:.1f}fps\n",
        f"Duration   : {dur:.0f}s ({total_fr} frames)\n",
        f"Run time   : {total_time:.2f}s\n\n",
    ]

    if engine.student_states:
        print()
        header = f"  {'Seat':<6} {'Avg':>7} {'Peak':>7} {'Low':>6}  {'Activity':<32}"
        sep    = "  "+"-"*60
        print(header); print(sep)
        lines += [header+"\n", sep+"\n"]
        for sid, st in sorted(engine.student_states.items()):
            act, note = st.reconciled_activity()
            label = act+(f" ({note})" if note else "")
            row = f"  {sid:<6} {st.avg:>6.1f}%  {st.peak:>6.1f}%  {st.low:>5.1f}%  {label:<32}"
            print(row); lines.append(row+"\n")
        avgs = [st.avg for st in engine.student_states.values()]
        avg  = sum(avgs)/len(avgs)
        print(sep); print(f"\n  Class avg : {avg:.1f}%")
        lines += [sep+"\n", f"\n  Class avg : {avg:.1f}%\n"]
    print("="*65)
    with open(OUTPUT_REPORT,"w",encoding="utf-8") as f: f.writelines(lines)
    print(f"\n  Report: {OUTPUT_REPORT}")


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO
    run(os.path.abspath(video))
