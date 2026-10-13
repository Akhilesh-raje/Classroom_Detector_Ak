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
from core.topology import PhysicalSeat, ClassroomTopology, TopologyConfig, TopologyDiscovery, PersonTrack
from core.seat_associator import SeatAssociator, SeatAssignment
from core.student_state import StudentState
from core.behavior_tracker import BehaviorTracker
from core.detector import detect_faces_and_phones, analyze_face_crop, _is_within_padded_bbox
from core.face_recognition_engine import face_rec_engine
from core.ollama_labeler import ollama_labeler
from core.pipeline import (
    DetectionStage,
    AssociationStage,
    AnalysisStage,
    OutputStage,
    PipelineScheduler,
)
from core.scene import SceneState, EntityState

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

        # Spatial Perception V2.1 — Discovered Physical Topology
        self._classroom_layout = None   # Legacy ClassroomLayout | None
        self._calibrated       = False  # True once calibrate() has run
        self.topology: ClassroomTopology | None = None
        self.topology_config   = TopologyConfig(mode="auto")
        self.associator        = SeatAssociator()
        self._seat_label_to_id : dict = {}
        self._seat_id_to_label : dict = {}
        self.frame_count       = 0
        self.benchmark_records : list = []

        # Modular Pipeline Stages (Scene Intelligence Engine)
        self.detection_stage   = DetectionStage(min_conf=YOLO_CONF)
        self.association_stage = AssociationStage()
        self.analysis_stage    = AnalysisStage()
        self.output_stage      = OutputStage()
        self.scheduler         = PipelineScheduler()
        self.last_scene_state: SceneState | None = None

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
        self._classroom_layout = None
        self._calibrated       = False
        self.topology          = None
        self.associator.reset()
        self._seat_label_to_id = {}
        self._seat_id_to_label = {}
        self.frame_count       = 0
        self.benchmark_records = []
        self.last_scene_state  = None
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

    def calibrate(self, frames: list, W_px: int, H_px: int, config: TopologyConfig | None = None) -> None:
        """
        Run spatial calibration from a list of BGR frames.
        Supports both V2.1/V2.2 baseline ('auto') and Structure AI ('structure_ai').
        """
        if self._calibrated and self.topology is not None:
            return

        if config is not None:
            self.topology_config = config

        if getattr(self.topology_config, "mode", "auto") == "structure_ai":
            return self.calibrate_structure(frames, W_px, H_px)

        discovery = TopologyDiscovery(config=self.topology_config, frame_size=(W_px, H_px))

        for f_idx, frame in enumerate(frames):
            try:
                raw_dets, _, _, _ = detect_faces_and_phones(frame, min_conf=YOLO_CONF, persist=True)
                discovery.accumulate_raw(raw_dets, frame_idx=f_idx)
            except Exception as e:
                log.warning(f"calibrate frame {f_idx} failed: {e}")

        topo = discovery.estimate(min_observations=max(2, len(frames) // 6))
        if not topo or not topo.seats:
            log.warning("calibrate(): Topology discovery produced no seats — falling back to discovery mode")
            return

        self.topology = topo
        self.associator = SeatAssociator()
        self._seat_label_to_id = {s.label: i + 1 for i, s in enumerate(topo.seats)}
        self._seat_id_to_label = {i + 1: s.label for i, s in enumerate(topo.seats)}

        # Pre-seed legacy stabiliser with discovered seat positions
        n = len(topo.seats)
        self.stabilizer = SeatStabilizer(max_seats=min(n + 2, 20))
        scaffold = []
        for s_idx, s in enumerate(topo.seats):
            scaffold.append({
                "id": s_idx + 1,
                "center": s.anchor,
                "last_seen": time.time(),
                "last_bbox": s.bbox,
                "missed": 0,
                "_row": s.row,
                "_col": s.col,
                "label": s.label,
            })
        self.stabilizer.seats = scaffold
        self.stabilizer.next_seat_id = n + 1
        self.stabilizer.seats_sorted = True
        self.stabilizer._locked = True

        self._calibrated = True
        log.info(
            f"calibrate(): locked {n} seats across {topo.rows} row(s) "
            f"(seats per row: {topo.cols_per_row}, conf={topo.confidence:.2f})"
        )
        for rg in topo.row_geometries:
            log.info(f"  Row {rg.label} = {rg.semantic} (y_median={rg.y_median:.1f}, band=[{rg.y_min:.1f}, {rg.y_max:.1f}])")

    def calibrate_structure(self, frames: list, W_px: int, H_px: int) -> None:
        """
        Calibrates classroom using pure Structure AI (geometry-first topology).
        Completely decouples physical seat topology from person detections.

        Uses TopologyBuilder directly — the geometry-first topology defines
        seats from furniture/spatial priors, not from depth estimation or
        neural perception models.
        """
        if self._calibrated and self.topology is not None and self.topology.source == "structure_ai":
            return

        from core.structure.topology_builder import TopologyBuilder
        builder = TopologyBuilder(frame_size=(W_px, H_px))
        struct_map = builder.build_from_geometry([], depth_map=None)
        topo = ClassroomTopology.from_structure_map(struct_map)

        self.topology = topo
        self.associator = SeatAssociator()
        self._seat_label_to_id = {s.label: i + 1 for i, s in enumerate(topo.seats)}
        self._seat_id_to_label = {i + 1: s.label for i, s in enumerate(topo.seats)}

        n = len(topo.seats)
        self.stabilizer = SeatStabilizer(max_seats=min(n + 2, 20))
        scaffold = []
        for s_idx, s in enumerate(topo.seats):
            scaffold.append({
                "id": s_idx + 1,
                "center": s.anchor,
                "last_seen": time.time(),
                "last_bbox": s.bbox,
                "missed": 0,
                "_row": s.row,
                "_col": s.col,
                "label": s.label,
                "seat_id": s.seat_id,
            })
        self.stabilizer.seats = scaffold
        self.stabilizer.next_seat_id = n + 1
        self.stabilizer.seats_sorted = True
        self.stabilizer._locked = True
        self._calibrated = True

        log.info(
            f"calibrate_structure(): locked {n} physical seats across {topo.rows} bench(es) "
            f"(seats per row: {topo.cols_per_row}, conf={topo.confidence:.2f})"
        )
        for rg in topo.row_geometries:
            log.info(f"  Bench/Row {rg.label} = {rg.semantic} (desk_y={rg.y_median:.1f}, band=[{rg.y_min:.1f}, {rg.y_max:.1f}])")

    def apply_topology(self, topo: ClassroomTopology) -> None:
        """Explicitly apply a physical ClassroomTopology (e.g. 3x2 grid)."""
        self.topology = topo
        self.associator = SeatAssociator()
        self._seat_label_to_id = {s.label: i + 1 for i, s in enumerate(topo.seats)}
        self._seat_id_to_label = {i + 1: s.label for i, s in enumerate(topo.seats)}

        n = len(topo.seats)
        self.stabilizer = SeatStabilizer(max_seats=min(n + 2, 20))
        scaffold = []
        for s_idx, s in enumerate(topo.seats):
            scaffold.append({
                "id": s_idx + 1,
                "center": s.anchor,
                "last_seen": time.time(),
                "last_bbox": s.bbox,
                "missed": 0,
                "_row": s.row,
                "_col": s.col,
                "label": s.label,
                "seat_id": s.seat_id,
            })
        self.stabilizer.seats = scaffold
        self.stabilizer.next_seat_id = n + 1
        self.stabilizer.seats_sorted = True
        self.stabilizer._locked = True
        self._calibrated = True
        log.info(f"apply_topology(): locked {n} physical seats across {topo.rows} rows (cols: {topo.cols_per_row})")

    def _seat_to_grid_label(self, sid: int) -> str:
        """
        Derive R{r}C{c} label for a seat.
        Prefers V2.1 Topology mapping, then scaffold metadata, then row_layout.
        """
        if sid in self._seat_id_to_label:
            return self._seat_id_to_label[sid]

        # Fast path: scaffold metadata stored on the seat dict itself
        for seat in self.stabilizer.seats:
            if seat["id"] == sid:
                if "label" in seat:
                    return seat["label"]
                row = seat.get("_row")
                col = seat.get("_col")
                if row is not None and col is not None:
                    return f"R{row+1}C{col}"
                break

        # Fallback: use discovered row_layout from EMA-converged positions
        layout = getattr(self.stabilizer, "row_layout", [])
        if layout:
            offset = sid - 1
            for r_idx, count in enumerate(layout):
                if offset < count:
                    return f"R{r_idx + 1}C{offset + 1}"
                offset -= count
        return f"S{sid}"  # layout not yet discovered

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
        Modular pipeline architecture (Phase 0 Universal Scene Intelligence Engine):
          Stage 1 (DetectionStage): YOLO person & peripheral detection + track formulation
          Stage 2 (AssociationStage): Hungarian spatial association to physical seats
          Stage 3 (AnalysisStage): Multimodal face, attention, behavior, and identity
          Stage 4 (OutputStage): Output formatting + Universal Scene Representation (SceneState)
        Returns:
            List of structured face prediction dicts (compatible with all API/UI consumers)
        """
        self.frame_count += 1

        # ── Stage 1: Detection & Person Tracking ──────────────────────────────
        raw_dets, phones, laptops, earphones, tracks, t_by_id = self.detection_stage.execute(
            frame=frame,
            persist=persist,
            topology=self.topology,
            frame_idx=self.frame_count,
        )

        # ── Stage 2: Spatial Association ──────────────────────────────────────
        mapped_dets, bench_records = self.association_stage.execute(
            tracks=tracks,
            raw_dets=raw_dets,
            topology=self.topology,
            associator=self.associator,
            seat_label_to_id=self._seat_label_to_id,
            stabilizer=self.stabilizer,
            frame_idx=self.frame_count,
            W_px=W_px,
            H_px=H_px,
            t_by_id=t_by_id,
        )
        self.benchmark_records.extend(bench_records)

        # ── Optional Grid Overlay ─────────────────────────────────────────────
        if self.spatial_grid is not None:
            _draw_grid_overlay(frame, self.spatial_grid)

        # ── Stage 3: Multimodal State Analysis ────────────────────────────────
        processed_faces, new_attendance_events = self.analysis_stage.execute(
            frame=frame,
            mapped_dets=mapped_dets,
            raw_dets=raw_dets,
            phones=phones,
            laptops=laptops,
            earphones=earphones,
            student_states=self.student_states,
            behavior_trackers=self.behavior_trackers,
            face_match_counters=self.face_match_counters,
            registered_faces=self.registered_faces,
            session_start_time=self.session_start_time,
            W_px=W_px,
            H_px=H_px,
            spatial_grid=self.spatial_grid,
            seat_to_grid_label_fn=self._seat_to_grid_label,
        )
        self.pending_attendance_events.extend(new_attendance_events)

        # ── Stage 4: Output & Universal Scene Representation ──────────────────
        processed_faces, scene_state = self.output_stage.execute(
            processed_faces=processed_faces,
            topology=self.topology,
            frame_idx=self.frame_count,
        )
        self.last_scene_state = scene_state

        return processed_faces

    def export_benchmark_csv(self, filepath: str):
        """Export accumulated association benchmark records to CSV."""
        import csv
        fieldnames = ["frame", "track_id", "head_x", "head_y", "bottom_y", "row", "seat", "cost", "status", "prev_seat", "geometric_match", "match_type"]
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in self.benchmark_records:
                writer.writerow(r)
        log.info(f"Exported {len(self.benchmark_records)} benchmark records to {filepath}")

    # ── Person-based layout detection ────────────────────────────────────────

    def build_person_grid(self, person_bboxes: list, W_px: int, H_px: int) -> "SpatialGrid":
        """
        Build a SpatialGrid from a list of person bounding boxes detected in
        the first few frames — no furniture detection needed.

        Algorithm:
          1. Cluster persons into rows by Y-center with an adaptive gap.
          2. Within each row sort left-to-right.
          3. Adjacent persons whose X-centers are within BENCH_MERGE_FRAC of
             each other share the same physical bench — label them L / R (/ M).
          4. Return a SpatialGrid with R{row}B{bench}-{pos} labels so the
             frontend can render the real classroom bench structure.

        ponytail: O(n²) clustering — fine for ≤ 20 students.
        """
        from core.classroom_mapper import SpatialGrid, GridCell

        if not person_bboxes:
            return SpatialGrid(cells=[], rows=0, cols=0)

        # Adaptive row-gap: fraction of frame height between row centers
        ROW_GAP_FRAC    = 0.10   # 10 % of frame height
        # Two persons within this fraction of frame width share a bench
        BENCH_MERGE_FRAC = 0.12  # 12 % of frame width

        # Deduplicate near-identical bboxes (same person seen multiple frames)
        def _cx(b): return (b[0] + b[2]) / 2
        def _cy(b): return (b[1] + b[3]) / 2

        unique: list = []
        for bb in person_bboxes:
            if not any(abs(_cx(bb) - _cx(u)) < W_px * 0.08 and
                       abs(_cy(bb) - _cy(u)) < H_px * 0.08
                       for u in unique):
                unique.append(bb)

        # Sort by Y then X
        unique.sort(key=lambda b: (_cy(b), _cx(b)))

        # Cluster into rows
        gap = ROW_GAP_FRAC * H_px
        rows: list[list] = []
        for bb in unique:
            cy = _cy(bb)
            if rows and abs(cy - _cy(rows[-1][-1])) <= gap:
                rows[-1].append(bb)
            else:
                rows.append([bb])

        # Sort each row left-to-right
        for row in rows:
            row.sort(key=_cx)

        # Build GridCells — group side-by-side persons into benches
        POSITIONS = {1: ["C"], 2: ["L", "R"], 3: ["L", "M", "R"]}
        cells: list[GridCell] = []
        bench_width = BENCH_MERGE_FRAC * W_px

        for r_idx, row in enumerate(rows):
            benches: list[list] = []
            cur_bench: list = [row[0]]
            for bb in row[1:]:
                if _cx(bb) - _cx(cur_bench[-1]) <= bench_width:
                    cur_bench.append(bb)
                else:
                    benches.append(cur_bench)
                    cur_bench = [bb]
            benches.append(cur_bench)

            for b_idx, bench in enumerate(benches):
                n = min(len(bench), 4)
                positions = POSITIONS.get(n, [f"S{i+1}" for i in range(n)])
                for p_idx, (bb, pos) in enumerate(zip(bench, positions)):
                    x1, y1, x2, y2 = bb
                    # Expand bbox slightly so it covers the full seat zone
                    pad_x = int((x2 - x1) * 0.15)
                    pad_y = int((y2 - y1) * 0.10)
                    cells.append(GridCell(
                        row=r_idx,
                        col=b_idx * 4 + p_idx,
                        label=f"R{r_idx+1}B{b_idx+1}-{pos}",
                        bbox=[max(0, x1-pad_x), max(0, y1-pad_y),
                              min(W_px, x2+pad_x), min(H_px, y2+pad_y)],
                    ))

        n_rows = len(rows)
        n_cols = max((len(r) for r in rows), default=0)
        grid   = SpatialGrid(cells=cells, rows=n_rows, cols=n_cols)
        self.spatial_grid = grid
        # Also update stabilizer cap to match detected count
        detected = len(cells)
        if detected > self.stabilizer.max_seats:
            from core.stabilizer import SeatStabilizer
            old = self.stabilizer
            self.stabilizer = SeatStabilizer(max_seats=min(detected + 2, 20))
            # Copy already-discovered seats so IDs stay stable
            self.stabilizer.seats        = old.seats
            self.stabilizer.next_seat_id = old.next_seat_id
            self.stabilizer.seats_sorted = old.seats_sorted
            self.stabilizer._locked      = old._locked
        return grid

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

        from core.behavior_tracker import BehaviorTracker
        behavior_trackers = {sid: BehaviorTracker(session_start_time=t0) for sid in range(1, 25)}
        student_crops = {}
        best_crop_confs = {}

        # ── Phase 1: layout scan (first LAYOUT_SCAN_FRAMES processed frames) ──
        # Collect all person bboxes, then build a SpatialGrid from real positions.
        LAYOUT_SCAN_FRAMES = 5
        layout_bboxes: list = []
        layout_done = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue

            t_now = time.time()
            fps_window.append(1.0 / max(1e-6, t_now - t_frame))
            t_frame  = t_now
            processed += 1

            # During layout scan: run YOLO only (no MediaPipe) for speed
            if not layout_done:
                from core.detector import detect_faces_and_phones
                raw_dets, _, _, _ = detect_faces_and_phones(
                    frame, min_conf=YOLO_CONF, persist=False
                )
                for d in raw_dets:
                    layout_bboxes.append(d["bbox"])

                if processed >= LAYOUT_SCAN_FRAMES:
                    # Build spatial grid from collected person positions
                    grid = self.build_person_grid(layout_bboxes, W_px, H_px)
                    layout_done = True

                    # Describe the layout for the frontend
                    seat_list = [
                        {"label": c.label, "row": c.row + 1,
                         "bbox": c.bbox, "bench": int(c.label.split("B")[1].split("-")[0])
                         if "B" in c.label else c.col + 1}
                        for c in grid.cells
                    ]
                    row_counts = {}
                    for c in grid.cells:
                        row_counts[c.row + 1] = row_counts.get(c.row + 1, 0) + 1

                    yield {
                        "type":       "layout",
                        "rows":       grid.rows,
                        "cols":       grid.cols,
                        "totalSeats": len(grid.cells),
                        "seats":      seat_list,
                        "rowCounts":  row_counts,
                        "frameWidth": W_px,
                        "frameHeight": H_px,
                    }

                # Still in scan phase — skip normal frame processing
                fps_live = sum(fps_window) / len(fps_window)
                detections += 0  # scan frames don't count
                continue

            fps_live = sum(fps_window) / len(fps_window)

            faces = self.process_single_frame(frame, W_px, H_px, persist=True)
            detections += len(faces)

            now_ts = time.time()
            for f in faces:
                sid = f.get("id")
                slabel = f.get("seat_label") or f"S{sid}"
                if sid in behavior_trackers:
                    behavior_trackers[sid].record(f.get("activity", "neutral"), now_ts)
                if f.get("status") == "OK" and "bbox" in f:
                    pbox = f.get("person_bbox", f["bbox"])
                    x1, y1, x2, y2 = [int(v) for v in pbox]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(W_px, x2), min(H_px, y2)
                    conf = float(f.get("confidence", 0.8))
                    if (x2 - x1) > 60 and (y2 - y1) > 80:
                        if slabel not in best_crop_confs or conf > best_crop_confs[slabel]:
                            student_crops[slabel] = frame[y1:y2, x1:x2].copy()
                            best_crop_confs[slabel] = conf

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

        # Automatically export comprehensive structure and student reports
        out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "structure_experiment"))
        try:
            from core.pipeline.comprehensive_analyzer import ComprehensiveExporter
            exporter = ComprehensiveExporter(out_dir=out_dir)
            session_meta = {
                "session_id": self.session_id,
                "video_path": video_path,
                "width": W_px,
                "height": H_px,
                "fps": fps_src,
                "duration": round(frame_idx / fps_src, 2),
                "frame_skip": frame_skip,
                "processed_frames": processed,
            }
            exporter.export_all(self, behavior_trackers, student_crops, session_meta)
        except Exception as e:
            log.warning(f"Comprehensive export error: {e}")

        yield {
            "type":               "complete",
            "framesProcessed":     processed,
            "totalSourceFrames":   total,
            "frameCoverage":       round(processed / max(1, total) * frame_skip * 100, 1),
            "effectiveDuration":   round(processed * frame_skip / fps_src, 2),
            "totalFaceDetections": detections,
            "elapsedSeconds":      round(elapsed, 2),
            "sessionId":           self.session_id,
            "exportDirectory":     out_dir,
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
