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
from core.topology import TopologyConfig
from core.visualizer import (
    alpha_rect, attn_bar, corner_brackets,
    smooth, stable_label, draw_global_hud,
    C, ACT_COLOR, ACT_ICON,
)
from core.behavior_tracker import BehaviorTracker
from core.pipeline.comprehensive_analyzer import ComprehensiveExporter


# ─────────────────────────────────────────────────────────────────────────────
# Overlay renderers
# ─────────────────────────────────────────────────────────────────────────────
def draw_compact(frame, face):
    x1, y1, x2, y2 = face["bbox"]
    sid  = face.get("id", "?")
    tid  = face.get("track_id", sid)
    act  = stable_label(sid, "act",
           face.get("reconciledActivity", face.get("activity", "neutral")), hold=5)
    attn = smooth(sid, "attn", face.get("attentionScore", 0), 0.10)
    geo_match = face.get("geometric_match", face.get("seat_confidence", 1.0))
    seat_lbl = face.get("seat_label") or face.get("gridCell") or f"S{sid}"

    # Distinct Row Colors: Cyan for Row 1 (Front), Magenta for Row 2 (Back)
    is_r1 = seat_lbl.startswith("R1")
    is_r2 = seat_lbl.startswith("R2")
    if is_r1:
        row_col = (255, 215, 0)      # Cyan BGR for Front Row (R1)
    elif is_r2:
        row_col = (200, 100, 255)    # Magenta BGR for Back Row (R2)
    else:
        row_col = C["orange"]

    cv2.rectangle(frame, (x1,y1), (x2,y2), row_col, 2)
    corner_brackets(frame, x1, y1, x2, y2, row_col,
                    length=min(18,max(5,(x2-x1)//5),max(5,(y2-y1)//5)), thickness=2)

    badge = f"{seat_lbl} (trk#{tid}) | {attn:.0f}%"
    bw = max(75, len(badge)*7+8); bh = 18
    by1 = max(0, y1-bh-2); by2 = max(bh, y1-2)
    alpha_rect(frame, x1, by1, x1+bw, by2, row_col, alpha=0.88)
    cv2.putText(frame, badge, (x1+3, by2-3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, C["black"], 1, cv2.LINE_AA)
    attn_bar(frame, x1, y2+2, x2-x1, 5, attn)

    if face.get("phoneDetected"):
        alpha_rect(frame, x1, max(0,by1-16), x2, max(0,by1), C["red"], alpha=0.90)
        cv2.putText(frame, "PHONE!", (x1+3, max(9,by1-3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C["white"], 2, cv2.LINE_AA)


def draw_full(frame, face):
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = face["bbox"]
    sid    = face.get("id", "?")
    tid    = face.get("track_id", sid)  # actual YOLO track ID, not seat_id sentinel
    act    = stable_label(sid, "act",
             face.get("reconciledActivity", face.get("activity","neutral")), hold=5)
    col    = ACT_COLOR.get(act, C["green"])
    icon   = ACT_ICON.get(act, act[:8].upper())
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
    geo_match  = face.get("geometric_match", face.get("seat_confidence", 1.0))
    match_type = face.get("match_type", face.get("topology", {}).get("match_type", "EXACT"))
    seat_lbl = face.get("seat_label") or face.get("gridCell") or f"S{sid}"
    is_r1 = seat_lbl.startswith("R1")
    is_r2 = seat_lbl.startswith("R2")
    if is_r1:
        row_col = (255, 215, 0)      # Cyan BGR for Front Row (R1)
    elif is_r2:
        row_col = (200, 100, 255)    # Magenta BGR for Back Row (R2)
    else:
        row_col = C["orange"]

    cv2.rectangle(frame, (x1,y1), (x2,y2), row_col, 2)
    corner_brackets(frame, x1, y1, x2, y2, row_col,
                    length=min(15,max(4,(x2-x1)//5),max(4,(y2-y1)//5)), thickness=2)

    badge = f"{seat_lbl} (trk#{tid})"
    badge_col = row_col
    if face.get("status") == "UNASSIGNED":
        badge = f"UNASSIGNED (trk#{tid})"
        badge_col = C["orange"]
    elif face.get("ghost"):
        badge = f"{seat_lbl} (GHOST)"
        badge_col = (120, 120, 120)

    bw_b  = max(60, len(badge)*7+8)
    alpha_rect(frame, x1, max(0,y1-20), x1+bw_b, max(0,y1-2), badge_col, alpha=0.88)
    cv2.putText(frame, badge, (x1+3, max(9,y1-5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, C["black"], 1, cv2.LINE_AA)

    pw, ph = 152, 115
    px = min(x2+3, W-pw-2)
    py = min(max(0,y1), H-ph-2)
    alpha_rect(frame, px, py, px+pw, py+ph, C["panel"], alpha=0.80)
    cv2.rectangle(frame, (px,py), (px+pw,py+ph), (55,55,55), 1)

    cv2.putText(frame, icon, (px+4,py+13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{attn:.0f}%", (px+pw-38,py+13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1, cv2.LINE_AA)
    attn_bar(frame, px+4, py+17, pw-8, 5, attn)

    cv2.putText(frame, f"Geo: {geo_match:.2f} ({match_type[:8]})", (px+4,py+32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, badge_col, 1, cv2.LINE_AA)

    cv2.putText(frame, f"Avg {avg_a:.0f}%", (px+4,py+44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, C["white"], 1, cv2.LINE_AA)
    attn_bar(frame, px+4, py+48, pw-8, 4, avg_a)

    emo_col = C["green"] if emo=="happy" else C["yellow"] if emo=="neutral" else C["orange"]
    cv2.putText(frame, f"Emo:{emo}", (px+4,py+62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, emo_col, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Y:{yaw:+.0f} P:{pitch:+.0f}", (px+4,py+73),
                cv2.FONT_HERSHEY_SIMPLEX, 0.26, (155,155,155), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Eye:{eye:.2f}", (px+4,py+84),
                cv2.FONT_HERSHEY_SIMPLEX, 0.26, (155,155,155), 1, cv2.LINE_AA)
    attn_bar(frame, px+50, py+79, pw-55, 4, eye*100)

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
# Overlap suppressor & Classroom System View
# ─────────────────────────────────────────────────────────────────────────────
def _iou(a, b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    iw=max(0,min(ax2,bx2)-max(ax1,bx1)); ih=max(0,min(ay2,by2)-max(ay1,by1))
    inter=iw*ih
    return inter/max(1,(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter)

def suppress_overlapping_faces(faces):
    """
    Only suppress true duplicate detections of the SAME person's face.
    Never suppress using full-body person_bbox, which naturally overlaps
    between front and back rows in a classroom.
    """
    if len(faces) <= 1:
        return faces
    faces = sorted(faces, key=lambda f: f.get("confidence", 0.0), reverse=True)
    keep = []
    sup = set()
    for i, fi in enumerate(faces):
        if i in sup:
            continue
        keep.append(fi)
        bi = fi["bbox"]
        for j, fj in enumerate(faces):
            if j <= i or j in sup:
                continue
            if _iou(bi, fj["bbox"]) > 0.55:
                sup.add(j)
    return keep


def draw_classroom_system_overlay(frame, engine, faces):
    """
    Renders Classroom System layout on screen:
    - Physical seat boundaries & anchor crosshairs
    - Live 2x6 Classroom Topology mini-grid in bottom-right corner
    """
    H, W = frame.shape[:2]
    if not engine.topology or not engine.topology.seats:
        return

    # Draw subtle seat boundary rectangles and anchor crosses
    for s in engine.topology.seats:
        sx1, sy1, sx2, sy2 = s.bbox
        ax, ay = s.anchor
        is_r1 = (s.row == 0 or "R1" in s.label)
        scol = (255, 215, 0) if is_r1 else (200, 100, 255)

        overlay = frame.copy()
        cv2.rectangle(overlay, (int(sx1), int(sy1)), (int(sx2), int(sy2)), scol, -1)
        cv2.addWeighted(overlay, 0.05, frame, 0.95, 0, frame)
        cv2.rectangle(frame, (int(sx1), int(sy1)), (int(sx2), int(sy2)), scol, 1, cv2.LINE_AA)

        cv2.drawMarker(frame, (int(ax), int(ay)), scol, cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA)
        cv2.putText(frame, s.label, (int(ax) - 14, int(sy1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, scol, 1, cv2.LINE_AA)

    # Mini-grid in bottom-right corner
    gw, gh = 280, 105
    gx = W - gw - 15
    gy = H - gh - 15
    alpha_rect(frame, gx, gy, gx + gw, gy + gh, (20, 20, 20), alpha=0.88)
    cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (90, 90, 90), 1)

    cv2.putText(frame, "CLASSROOM TOPOLOGY (2x6)", (gx + 10, gy + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, C["cyan"], 1, cv2.LINE_AA)

    occ_set = {f.get("seat_label") for f in faces if f.get("status") == "OK"}

    # Back row (R2) line at top
    for c in range(1, 7):
        lbl = f"R2C{c}"
        col = (200, 100, 255) if lbl in occ_set else (70, 70, 70)
        cx_cell = gx + 15 + (c - 1) * 42
        cy_cell = gy + 32
        cv2.rectangle(frame, (cx_cell, cy_cell), (cx_cell + 36, cy_cell + 20), col, -1 if lbl in occ_set else 1)
        cv2.putText(frame, f"C{c}", (cx_cell + 8, cy_cell + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, C["black"] if lbl in occ_set else C["white"], 1, cv2.LINE_AA)

    # Front row (R1) line at bottom
    for c in range(1, 7):
        lbl = f"R1C{c}"
        col = (255, 215, 0) if lbl in occ_set else (70, 70, 70)
        cx_cell = gx + 15 + (c - 1) * 42
        cy_cell = gy + 58
        cv2.rectangle(frame, (cx_cell, cy_cell), (cx_cell + 36, cy_cell + 20), col, -1 if lbl in occ_set else 1)
        cv2.putText(frame, f"C{c}", (cx_cell + 8, cy_cell + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, C["black"] if lbl in occ_set else C["white"], 1, cv2.LINE_AA)

    cv2.putText(frame, f"Occupancy: {len(occ_set)}/12 | Double Booking: 0", (gx + 10, gy + 94),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 180, 180), 1, cv2.LINE_AA)


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
            # Record new target boxes keyed on track_id
            self._next_box = {
                f.get("track_id", f.get("id")): list(f["bbox"]) for f in faces if "bbox" in f
            }
            # For any track not seen before, seed prev from next
            for key, box in self._next_box.items():
                if key not in self._prev_box:
                    self._prev_box[key] = list(box)

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
                key = f.get("track_id", f.get("id"))
                if key is not None and key in self._prev_box and key in self._next_box:
                    p = self._prev_box[key]
                    n = self._next_box[key]
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
        self.behavior_trackers = {sid: BehaviorTracker(session_start_time=time.time()) for sid in range(1, 25)}
        self.student_crops = {}
        self.best_crop_confs = {}

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

            # Record behaviors and extract best crops
            now_ts = time.time()
            for f in faces:
                sid = f.get("id")
                slabel = f.get("seat_label") or f"S{sid}"
                if sid in self.behavior_trackers:
                    self.behavior_trackers[sid].record(f.get("activity", "neutral"), now_ts)
                if f.get("status") == "OK" and "bbox" in f:
                    pbox = f.get("person_bbox", f["bbox"])
                    x1, y1, x2, y2 = [int(v) for v in pbox]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(self.W, x2), min(self.H, y2)
                    conf = float(f.get("confidence", 0.8))
                    if (x2 - x1) > 60 and (y2 - y1) > 80:
                        if slabel not in self.best_crop_confs or conf > self.best_crop_confs[slabel]:
                            self.student_crops[slabel] = frame[y1:y2, x1:x2].copy()
                            self.best_crop_confs[slabel] = conf

            times.append(dt)
            if len(times) > 30: times.pop(0)
            avg_ms = sum(times)/len(times)

            self.shared.update(faces, avg_ms)


# ─────────────────────────────────────────────────────────────────────────────
# Main — display loop (never blocked by AI)
# ─────────────────────────────────────────────────────────────────────────────
def run(video_path: str, structure_ai: bool = False, headless: bool = False, max_frames: int = 0):
    print("="*65)
    print(f"SmartClass AI — Real Classroom ({'Structure AI' if structure_ai else 'Baseline'} Mode)")
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
    if not headless:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, DISPLAY_W, int(DISPLAY_W*H/W))

    engine   = ClassroomEngine(max_seats=MAX_SEATS)

    # ── Topology Calibration ──────────────────────────────────────────────────
    calib_mode = "structure_ai" if structure_ai else "auto"
    CALIB_FRAMES = 5 if structure_ai else 60
    print(f"  Running Classroom Topology Calibration (mode={calib_mode}, {CALIB_FRAMES} frames)...", flush=True)
    calib_frames = []
    for _ in range(CALIB_FRAMES):
        ret, cf = cap.read()
        if not ret:
            break
        calib_frames.append(cf)
    if calib_frames:
        engine.calibrate(calib_frames, W, H, config=TopologyConfig(mode=calib_mode))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind video to frame 0
    print("  Topology Calibration complete — seating arrangement locked.\n", flush=True)

    shared   = SharedState()
    stop_evt = threading.Event()
    ai_thd   = AIThread(engine, shared, stop_evt, W, H)
    ai_thd.start()

    show_details = True
    show_classroom_view = True if structure_ai else False
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
            if max_frames > 0 and shown >= max_frames:
                break

            # ── Give frame to AI thread (non-blocking) ─────────────────────────
            ai_thd.submit(frame)

            # ── Get latest AI result with interpolated bboxes ─────────────────
            faces = shared.get_faces()
            total_dets += len(faces)

            # ── Draw Classroom System view if enabled ─────────────────────────
            if show_classroom_view:
                draw_classroom_system_overlay(frame, engine, faces)

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
            mode_str = f"{'DETAIL' if show_details else 'COMPACT'} {'[SYS]' if show_classroom_view else ''}  {speed_mult:.1f}x  AI:{ai_ms:.0f}ms"
            draw_global_hud(frame, faces, {
                "mode":         mode_str,
                "fps":          disp_fps,
                "elapsed":      elapsed,
                "frame_idx":    shown,
                "total_frames": total_fr,
                "hint":         "C=system  D=details  SPACE=pause  +/-=speed  Q=quit",
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
            if not headless:
                cv2.imshow(WIN, frame)

        else:
            if not headless:
                cv2.waitKey(30)
            else:
                time.sleep(0.03)

        # ── Keys ──────────────────────────────────────────────────────────────
        key = (cv2.waitKey(1) & 0xFF) if not headless else 255
        if key in (ord("q"), 27):
            print("\n  Stopped by user."); break
        elif key == ord("c"):
            show_classroom_view = not show_classroom_view
            print(f"\n  [C] → Classroom System View: {'ON' if show_classroom_view else 'OFF'}")
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

    # Export V2.2 Association Benchmark CSV and print report
    csv_path = os.path.join(HERE, "benchmark_association_log.csv")
    engine.export_benchmark_csv(csv_path)
    print(f"  Benchmark CSV   : {csv_path}")

    try:
        from benchmark_v2_2 import compute_v2_2_metrics, print_v2_2_report
        v2_2_metrics = compute_v2_2_metrics(csv_path)
        print_v2_2_report(v2_2_metrics)
    except Exception as e:
        print(f"  V2.2 Benchmark audit warning: {e}")

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

    # ── Export Complete Structure & Longitudinal Student Data ───────────────────
    export_dir = os.path.join(HERE, "outputs", "structure_experiment")
    print(f"\n  Exporting comprehensive structure & student analytics to: {export_dir} ...")
    try:
        exporter = ComprehensiveExporter(out_dir=export_dir)
        session_meta = {
            "session_id": f"session_real_classroom_{int(time.time())}",
            "video_path": video_path,
            "width": W,
            "height": H,
            "fps": fps_src,
            "duration": round(shown / fps_src, 2),
            "frame_skip": AI_EVERY,
            "processed_frames": shown,
        }
        exporter.export_all(
            engine=engine,
            behavior_trackers=ai_thd.behavior_trackers,
            student_crops=ai_thd.student_crops,
            session_metadata=session_meta,
        )
        print(f"  Interactive Dashboard: {os.path.join(export_dir, 'classroom_dashboard.html')}")
    except Exception as e:
        print(f"  Export warning: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SmartClass AI Real Classroom Video Tester")
    parser.add_argument("video", nargs="?", default=DEFAULT_VIDEO, help="Video path")
    parser.add_argument("--structure-ai", action="store_true", help="Use Structure AI (furniture + depth geometry) calibration")
    parser.add_argument("--headless", action="store_true", help="Run without UI window")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum frames to process (0=all)")
    parser.add_argument("--export-dir", default=os.path.join(HERE, "outputs", "structure_experiment"), help="Directory for full exports")
    args = parser.parse_args()
    run(os.path.abspath(args.video), structure_ai=args.structure_ai, headless=args.headless, max_frames=args.max_frames)
