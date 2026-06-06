"""
SmartClass AI — Unified Classroom Engine
Handles both batch video analysis and frame-by-frame live camera feeds.
Extended with: spatial grid, laptop/earphone detection, behavior tracking,
session management, and face-match throttling.
"""
import os
import sys
import time
import math
import uuid
import cv2
import base64
import logging
from io import BytesIO
from PIL import Image
from collections import deque

# ── Force UTF-8 Output for Windows Console ────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── Logger Setup ──────────────────────────────────────────────────────────────
log = logging.getLogger("classroom_engine")
if not log.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(h)
    log.setLevel(logging.INFO)


def get_logger(name: str):
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


# ── Import Core Modules ───────────────────────────────────────────────────────
from core.config import (
    MAX_SEATS, DASH_EVERY, TW, YOLO_CONF, PERSIST,
    SMALL_BBOX_FRAC, FACE_MATCH_INTERVAL,
)
from core.stabilizer import SeatStabilizer
from core.student_state import StudentState
from core.behavior_tracker import BehaviorTracker
from core.detector import detect_faces_and_phones, analyze_face_crop, _is_within_padded_bbox
from core.face_recognition_engine import face_rec_engine
from core.ollama_labeler import ollama_labeler

# ── Safe Colorama Setup ───────────────────────────────────────────────────────
try:
    from colorama import Fore, Style, init as _ci
    _ci(autoreset=True)
    Cy, Gn, Yw, Rd, Mg, Bl, Wh = Fore.CYAN, Fore.GREEN, Fore.YELLOW, Fore.RED, Fore.MAGENTA, Fore.BLUE, Fore.WHITE
    Rs, Bd, Dm = Style.RESET_ALL, Style.BRIGHT, Style.DIM
except ImportError:
    Cy = Gn = Yw = Rd = Mg = Bl = Wh = Rs = Bd = Dm = ""

# ── ASCII Display Helpers ─────────────────────────────────────────────────────
def _bar(v, w=10, f="#", e="-"):
    n = max(0, min(w, round(v / 100 * w)))
    return f * n + e * (w - n)

def _cbar(v, w=10):
    c = Gn if v >= 75 else Yw if v >= 55 else Mg if v >= 35 else Rd
    return f"{c}[{_bar(v, w)}]{Rs}"

def _pbar(done, total, w=20):
    pct = done / max(1, total)
    n   = int(pct * w)
    bar = Gn + "=" * n + Dm + "-" * (w - n) + Rs
    return f"[{bar}] {Bd}{pct*100:5.1f}%{Rs}"

def _act_col(act):
    return {
        "attentive": Gn, "studying": Gn, "talking": Yw, "distracted": Rd,
        "fidgeting": Mg, "drowsy": Bl, "neutral": Wh, "phone": Rd,
        "laptop": Cy, "music": Mg,
    }.get(act, Wh)

def _tier(avg):
    if avg >= 80: return f"{Bd+Gn}HIGH    {Rs}"
    if avg >= 60: return f"{Bd+Yw}MEDIUM  {Rs}"
    if avg >= 40: return f"{Bd+Mg}LOW     {Rs}"
    return              f"{Bd+Rd}CRITICAL{Rs}"

def _trend(hist):
    if len(hist) < 12: return f"{Dm}  ~   {Rs}"
    r = sum(list(hist)[-6:])   / 6
    o = sum(list(hist)[-12:-6]) / 6
    d = r - o
    if d >  2.5: return f"{Gn}  UP  {Rs}"
    if d < -2.5: return f"{Rd}  DN  {Rs}"
    return              f"{Wh} STDY {Rs}"

def _sep(c="-"):
    print(Dm + c * TW + Rs)

def _dsep():
    print(Bd + Cy + "=" * TW + Rs)

def _banner(lines):
    print(Bd + Cy + "+" + "-" * (TW - 2) + "+" + Rs)
    for line in lines:
        cleaned = (line.replace(Cy,"").replace(Gn,"").replace(Yw,"")
                       .replace(Rd,"").replace(Mg,"").replace(Bl,"")
                       .replace(Wh,"").replace(Rs,"").replace(Bd,"").replace(Dm,""))
        pad = max(0, TW - 4 - len(cleaned))
        print(Bd + Cy + "| " + Rs + line + " " * pad + Bd + Cy + " |" + Rs)
    print(Bd + Cy + "+" + "-" * (TW - 2) + "+" + Rs)

def crop_to_b64(frame, x1, y1, x2, y2, size=128):
    c = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if c.size == 0:
        return None
    img = Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)).resize((size, size))
    buf = BytesIO()
    img.save(buf, "JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


# ── Grid overlay helper ───────────────────────────────────────────────────────
def _draw_grid_overlay(frame, spatial_grid, student_name_map: dict | None = None):
    """Draw semi-transparent grid cell boundaries on the frame in-place."""
    if spatial_grid is None:
        return
    overlay = frame.copy()
    for cell in spatial_grid.cells:
        x1, y1, x2, y2 = cell.bbox
        # Clamp to frame
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (200, 200, 200), -1)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    # Draw labels
    for cell in spatial_grid.cells:
        x1, y1, x2, y2 = cell.bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        label = cell.label
        if student_name_map and cell.label in student_name_map:
            label = student_name_map[cell.label]
        cv2.putText(frame, label, (x1 + 4, min(h - 4, y1 + 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def _nearest_cell(spatial_grid, cx: float, cy: float):
    """Return the GridCell nearest to (cx, cy). Tie-break: lower row, then lower col."""
    if spatial_grid is None or not spatial_grid.cells:
        return None
    best = None
    best_dist = float("inf")
    for cell in spatial_grid.cells:
        bx1, by1, bx2, by2 = cell.bbox
        ccx = (bx1 + bx2) / 2
        ccy = (by1 + by2) / 2
        dist = math.hypot(cx - ccx, cy - ccy)
        if dist < best_dist or (dist == best_dist and best is not None and
                                (cell.row < best.row or
                                 (cell.row == best.row and cell.col < best.col))):
            best_dist = dist
            best = cell
    return best


# ── Unified Engine ────────────────────────────────────────────────────────────
class ClassroomEngine:
    def __init__(self, max_seats=MAX_SEATS):
        self.stabilizer = SeatStabilizer(max_seats=max_seats)
        self.student_states: dict[int, StudentState] = {}

        # Session tracking
        self.session_id:         str   = str(uuid.uuid4())
        self.session_start_time: float = time.time()
        self.spatial_grid              = None   # SpatialGrid | None
        self.behavior_trackers:  dict  = {}     # seat_id → BehaviorTracker
        self.face_match_counters: dict = {}     # seat_id → int (frames since last match)

        # Pending attendance events for /api/camera/detections
        self.pending_attendance_events: list = []

        # Registered faces for recognition: list of (student_id, name, encoding)
        self.registered_faces: list = []

        log.info(f"ClassroomEngine initialized (max_seats={max_seats}, session={self.session_id[:8]}…)")
        if not face_rec_engine.is_reliable:
            log.warning(
                "⚠ Face recognition using HISTOGRAM fallback — identity matching is unreliable. "
                "Install insightface or face_recognition for real face recognition."
            )

    def reset(self):
        """Reset state for a new session."""
        self.stabilizer = SeatStabilizer(max_seats=self.stabilizer.max_seats)
        self.student_states      = {}
        self.session_id          = str(uuid.uuid4())
        self.session_start_time  = time.time()
        self.spatial_grid        = None
        self.behavior_trackers   = {}
        self.face_match_counters = {}
        self.pending_attendance_events = []
        # Keep registered_faces across resets

    def load_registered_faces(self, students: list):
        """Load student face encodings into memory for recognition."""
        self.registered_faces = []
        for s in students:
            if s.face_encoding:
                enc = face_rec_engine.bytes_to_encoding(s.face_encoding)
                if enc is not None:
                    self.registered_faces.append((s.id, s.name, enc))
        log.info(f"Loaded {len(self.registered_faces)} registered face(s) for recognition")

    def _state(self, seat_id: int) -> StudentState:
        if seat_id not in self.student_states:
            self.student_states[seat_id] = StudentState(seat_id)
        return self.student_states[seat_id]

    def _tracker(self, seat_id: int) -> BehaviorTracker:
        if seat_id not in self.behavior_trackers:
            self.behavior_trackers[seat_id] = BehaviorTracker(self.session_start_time)
        return self.behavior_trackers[seat_id]

    def process_single_frame(self, frame, W_px, H_px, persist=True) -> list:
        """
        Process a single image frame (live webcam or video).
        Returns a list of structured face prediction dicts.
        """
        # ── Detection ─────────────────────────────────────────────────────────
        raw_dets, phones, laptops, earphones = detect_faces_and_phones(
            frame, min_conf=YOLO_CONF, persist=persist
        )
        mapped_dets = self.stabilizer.update_and_map(raw_dets, W_px, H_px)

        # ── Draw grid overlay ─────────────────────────────────────────────────
        if self.spatial_grid is not None:
            _draw_grid_overlay(frame, self.spatial_grid)

        frame_area = W_px * H_px
        processed_faces = []

        for det in mapped_dets:
            sid = det.get("seat_id")
            if sid is None:
                continue

            is_ghost = det.get("ghost", False)
            px1, py1, px2, py2 = det["person_bbox"]

            # ── Phone detection ───────────────────────────────────────────────
            phone_detected = False
            for ph in phones:
                if _is_within_padded_bbox(ph["bbox"], det["person_bbox"], 0.10):
                    phone_detected = True
                    break

            # ── Laptop detection ──────────────────────────────────────────────
            laptop_detected = False
            for lp in laptops:
                if _is_within_padded_bbox(lp["bbox"], det["person_bbox"], 0.10):
                    laptop_detected = True
                    break

            # ── Earphone detection ────────────────────────────────────────────
            earphone_detected = False
            person_idx = None
            for i, p in enumerate(raw_dets):
                if p.get("id") == det.get("id"):
                    person_idx = i
                    break
            if person_idx is not None:
                for ep in earphones:
                    if ep["person_idx"] == person_idx:
                        earphone_detected = True
                        break

            # ── MediaPipe face analysis ───────────────────────────────────────
            # Skip expensive MediaPipe analysis on ghost frames — use cached state
            if is_ghost:
                st   = self._state(sid)
                tr   = _trend(st.attn_hist)
                x1, y1, x2, y2 = det["bbox"]
                processed_faces.append({
                    "id":               sid,
                    "bbox":             [x1, y1, x2, y2],
                    "headPose":         {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
                    "attentionScore":   st.attn_ema or 0.0,
                    "attentionRaw":     st.attn_ema or 0.0,
                    "avgAttention30":   st.avg,
                    "attentionFactors": st.last_factors,
                    "attentionBreakdown": st.attention_breakdown(),
                    "emotion":          "neutral",
                    "expressions":      {},
                    "activity":         st.prev_act,
                    "reconciledActivity": st.reconciled_activity()[0],
                    "activityNote":     st.reconciled_activity()[1],
                    "cropUsed":         "ghost",
                    "trend":            tr,
                    "thumbnail":        None,
                    "phoneDetected":    False,
                    "laptopDetected":   False,
                    "earphoneDetected": False,
                    "eyeOpenness":      1.0,
                    "mouthOpen":        0.0,
                    "gridCell":         "",
                    "faceMatchDue":     False,
                    "matchedStudentId": None,
                    "studentName":      None,
                    "matchConfidence":  0.0,
                    "ghost":            True,
                })
                continue

            bbox_area = (det["bbox"][2] - det["bbox"][0]) * (det["bbox"][3] - det["bbox"][1])
            small_det = frame_area > 0 and (bbox_area / frame_area) < SMALL_BBOX_FRAC

            mp_res    = None
            used_crop = "none"
            crop_candidates = det.get("crops", [])
            if not crop_candidates:
                crop_candidates = [{"name": "tight", "bbox": det["bbox"]}]

            if small_det:
                # Only try upper_body crop for small detections
                upper = [c for c in crop_candidates if c["name"] == "upper_body"]
                crop_candidates = upper if upper else []
            
            for crop_info in crop_candidates:
                cx1, cy1, cx2, cy2 = crop_info["bbox"]
                crop = frame[max(0, cy1):max(0, cy2), max(0, cx1):max(0, cx2)]
                if crop.size == 0:
                    continue
                mp_res = analyze_face_crop(crop)
                if mp_res:
                    used_crop = crop_info["name"]
                    break

            # ── Extract metrics ───────────────────────────────────────────────
            if mp_res:
                head_pose    = mp_res["head_pose"]
                eye_openness = mp_res["eye_openness"]
                mouth_open   = mp_res["mouth_open"]
                emotion      = mp_res["emotion"]
                blendshapes  = mp_res["blendshapes"]
            else:
                head_pose    = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
                eye_openness = 1.0
                mouth_open   = 0.0
                emotion      = "neutral"
                blendshapes  = {"neutral": 1.0}

            st  = self._state(sid)
            sig = st.extract_signals(
                det["bbox"],
                det["person_bbox"],
                phone_detected=phone_detected,
                eye_openness=eye_openness,
                mouth_open=mouth_open,
                head_pose=head_pose,
            )
            raw    = st.compute_attn(sig)
            attn_s = st.smooth(raw)
            prev   = st.prev_act
            act    = st.classify_act(sig, attn_s,
                                     laptop_detected=laptop_detected,
                                     earphone_detected=earphone_detected)

            # ── Behavior tracking ─────────────────────────────────────────────
            if act != prev:
                self._tracker(sid).record(act, time.time())

            # ── Grid cell assignment ──────────────────────────────────────────
            grid_cell_label = ""
            if self.spatial_grid is not None:
                pcx = (px1 + px2) / 2
                pcy = (py1 + py2) / 2
                cell = _nearest_cell(self.spatial_grid, pcx, pcy)
                if cell:
                    grid_cell_label = cell.label

            tr = _trend(st.attn_hist)
            x1, y1, x2, y2 = det["bbox"]

            # ── Face match throttle ───────────────────────────────────────────
            self.face_match_counters[sid] = self.face_match_counters.get(sid, 0) + 1
            face_match_due = self.face_match_counters[sid] >= FACE_MATCH_INTERVAL
            if face_match_due:
                self.face_match_counters[sid] = 0

            # ── Real face recognition ─────────────────────────────────────────
            matched_student_id   = None
            matched_student_name = None
            match_confidence     = 0.0

            if face_match_due and self.registered_faces and mp_res:
                # Extract encoding from the face crop
                cx1, cy1, cx2, cy2 = det["bbox"]
                face_crop = frame[max(0,cy1):max(0,cy2), max(0,cx1):max(0,cx2)]
                enc = face_rec_engine.extract_encoding(face_crop)
                if enc is not None:
                    reg_list = [(s_id, s_enc) for s_id, s_name, s_enc in self.registered_faces]
                    matched_student_id, match_confidence = face_rec_engine.find_best_match(enc, reg_list)
                    if matched_student_id is not None:
                        for s_id, s_name, _ in self.registered_faces:
                            if s_id == matched_student_id:
                                matched_student_name = s_name
                                break
                        # Queue attendance event
                        self.pending_attendance_events.append({
                            "student_id":   matched_student_id,
                            "student_name": matched_student_name,
                            "seat_id":      sid,
                            "confidence":   match_confidence,
                            "timestamp":    time.time(),
                        })

            processed_faces.append({
                "id":               sid,
                "bbox":             [x1, y1, x2, y2],
                "headPose":         head_pose,
                "attentionScore":   attn_s,
                "attentionRaw":     round(raw, 1),
                "avgAttention30":   st.avg,
                "attentionFactors": st.last_factors,
                "attentionBreakdown": st.attention_breakdown(),
                "emotion":          emotion,
                "expressions":      blendshapes,
                "activity":         act,
                "reconciledActivity": st.reconciled_activity()[0],
                "activityNote":     st.reconciled_activity()[1],
                "cropUsed":         used_crop,
                "trend":            tr,
                "thumbnail":        crop_to_b64(frame, x1, y1, x2, y2),
                "phoneDetected":    phone_detected,
                "laptopDetected":   laptop_detected,
                "earphoneDetected": earphone_detected,
                "eyeOpenness":      eye_openness,
                "mouthOpen":        mouth_open,
                "gridCell":         grid_cell_label,
                "faceMatchDue":     face_match_due,
                "matchedStudentId": matched_student_id,
                "studentName":      matched_student_name,
                "matchConfidence":  match_confidence,
            })

        return processed_faces

    def process_video_file(self, video_path: str, frame_skip: int = 3):
        """
        Batch process a local video file.
        Yields metadata, frame, and complete event dicts.
        """
        self.reset()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            yield {"error": f"Cannot open video file: {video_path}"}
            return

        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W_px    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H_px    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dur     = round(total / fps_src, 2)

        print()
        _banner([
            f"{Bd+Wh}SmartClass AI  ·  YOLO Classroom Engine v6.0{Rs}",
            f"{Dm}{os.path.basename(video_path)}{Rs}",
            f"{Cy}{W_px}x{H_px}{Rs} @ {Cy}{fps_src:.0f} fps{Rs}  "
            f"|  {Cy}{total} frames{Rs}  |  {Cy}{dur}s{Rs}",
        ])
        print()

        yield {
            "type": "metadata",
            "fps": fps_src, "totalFrames": total,
            "width": W_px, "height": H_px, "duration": dur,
        }

        frame_idx  = 0
        processed  = 0
        detections = 0
        fps_window = deque(maxlen=30)
        t0         = time.time()
        t_frame    = t0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue

            t_now = time.time()
            fps_window.append(1.0 / max(1e-6, t_now - t_frame))
            t_frame    = t_now
            fps_live   = sum(fps_window) / len(fps_window)
            processed += 1

            faces = self.process_single_frame(frame, W_px, H_px, persist=True)
            detections += len(faces)

            if processed % DASH_EVERY == 0:
                self._print_dashboard(frame_idx, frame_idx / fps_src, fps_live, faces, total, processed)

            yield {
                "type":       "frame",
                "frameIndex":  frame_idx,
                "timestamp":   round(frame_idx / fps_src, 3),
                "faceCount":   len(faces),
                "faces":       faces,
            }

        cap.release()
        elapsed = time.time() - t0
        self._print_report(processed, detections, elapsed)

        yield {
            "type":               "complete",
            "framesProcessed":     processed,
            "totalSourceFrames":   total,
            "frameCoverage":       round(processed / max(1, total) * frame_skip * 100, 1),
            "effectiveDuration":   round(processed * frame_skip / fps_src, 2),
            "totalFaceDetections": detections,
            "elapsedSeconds":      round(elapsed, 2),
            "sessionId":           self.session_id,
            "ollamaAvailable":     ollama_labeler.is_available(),
            "aiAnalysisNote": (
                "AI analysis powered by " + (ollama_labeler.get_model() or "N/A")
                if ollama_labeler.is_available()
                else "AI analysis unavailable \u2014 Ollama offline. Using rule-based fallback."
            ),
            "faceRecBackend":      face_rec_engine.backend,
            "faceRecReliable":     face_rec_engine.is_reliable,
            "studentStats": {
                str(sid): {
                    "avgAttention30": st.avg,
                    "peakAttention":  st.peak,
                    "lastActivity":   st.reconciled_activity()[0],
                    "activityNote":   st.reconciled_activity()[1],
                }
                for sid, st in self.student_states.items()
            },
        }

    # ── Console Display ───────────────────────────────────────────────────────
    def _print_dashboard(self, frame_idx, ts, fps_live, faces, total_frames, processed):
        print()
        n_stu = len(faces)
        print(
            f"----  Frame {Bd+Cy}{frame_idx:04d}{Rs}  t={ts:.2f}s  "
            f"{Bd+Wh}{n_stu} student{'s' if n_stu != 1 else ''}{Rs}  "
            f"@ {fps_live:.1f} fps"
        )
        if not faces:
            print(f"    {Rd}-- no active detections --{Rs}")
            print()
            return
        print()
        _sep()
        for face in sorted(faces, key=lambda f: f["id"]):
            tid  = face["id"]
            attn = face["attentionScore"]
            act  = face["activity"]
            avg  = face["avgAttention30"]
            tr   = face["trend"]
            gc   = face.get("gridCell", "")
            ac   = _act_col(act)
            print(
                f"  {Bd+Wh}{tid:<2}{Rs}  {_cbar(attn, 10)}  "
                f"{Bd}{attn:>5.1f}%{Rs}   {ac}{act:<12}{Rs} "
                f"{avg:>6.1f}%  {Dm}{gc}{Rs}  {tr}"
            )
        print()

    def _print_report(self, processed, detections, elapsed):
        print()
        _dsep()
        _banner([f"{Bd+Wh}SmartClass AI  ·  Session Complete{Rs}"])
        _dsep()
        print(f"\n  Frames Processed :  {Bd+Cy}{processed}{Rs}")
        print(f"  Total Detections :  {Bd+Cy}{detections}{Rs}")
        print(f"  Execution Time   :  {Bd+Cy}{elapsed:.2f}s{Rs}")
        if not self.student_states:
            print(f"\n  {Rd}No tracked data collected.{Rs}\n")
            return
        print(f"\n  Seat Analytics")
        _sep()
        for tid, st in sorted(self.student_states.items()):
            tr_sym = _trend(st.attn_hist)
            rec_act, note = st.reconciled_activity()
            display_act = f"{rec_act}" + (f" ({note})" if note else "")
            print(
                f"  {Bd+Wh}{tid:<2}{Rs} {_cbar(st.avg, 12)}  "
                f"{Bd}{st.avg:>6.1f}%{Rs}  {st.peak:>5.1f}%  {st.low:>5.1f}%  "
                f"{_act_col(rec_act)}{display_act:<28}{Rs}"
                f"{_tier(st.avg)} {tr_sym}"
            )
        avgs      = [st.avg for st in self.student_states.values()]
        class_avg = sum(avgs) / len(avgs)
        _sep()
        print(f"\n  Mean Attention   :  {_cbar(class_avg, 20)}  {Bd}{class_avg:.1f}%{Rs}")
        _dsep()
        print()
