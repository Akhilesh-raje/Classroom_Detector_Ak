"""
SmartClass AI — Classroom Layout Mapper
Detects tables/chairs in a reference frame using YOLO and builds a named SpatialGrid.
Falls back to a uniform grid when furniture detection fails.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field


@dataclass
class GridCell:
    row:   int          # 0-indexed
    col:   int          # 0-indexed
    label: str          # "R{row+1}C{col+1}"
    bbox:  list         # [x1, y1, x2, y2] pixel coords


@dataclass
class SpatialGrid:
    cells: list = field(default_factory=list)   # list[GridCell]
    rows:  int  = 0
    cols:  int  = 0


class ClassroomMapper:
    """
    Analyses a single BGR reference frame and returns a SpatialGrid that maps
    physical desk positions to named GridCells (R1C1, R1C2, …).

    Usage:
        mapper = ClassroomMapper()
        grid   = mapper.detect_layout(frame, rows_hint=3, cols_hint=4)
    """

    ROW_GAP_FRAC: float = 0.15   # fraction of frame height used as row-gap threshold
    DESK_CONF:    float = 0.20   # YOLO confidence for furniture detection
    DESK_CLASSES: list  = [56, 60]  # COCO: 56=chair, 60=dining table

    # ── Public API ────────────────────────────────────────────────────────────

    def detect_layout(
        self,
        frame,
        rows_hint: int | None = None,
        cols_hint: int | None = None,
    ) -> SpatialGrid:
        """
        Detect classroom layout from a BGR frame.

        Args:
            frame:      BGR numpy array (H×W×3).
            rows_hint:  Expected number of desk rows (guides clustering & fallback).
            cols_hint:  Expected number of desks per row (guides fallback).

        Returns:
            SpatialGrid with one GridCell per detected (or fallback) desk position.

        Raises:
            ValueError: If frame is None or has zero pixels.
        """
        import numpy as np

        if frame is None:
            raise ValueError("frame must not be None")
        if not hasattr(frame, "size") or frame.size == 0:
            raise ValueError("frame must not be empty (zero pixels)")

        frame_h, frame_w = frame.shape[:2]
        default_rows = rows_hint or 2
        default_cols = cols_hint or 3
        required = default_rows * default_cols

        # ── Run YOLO furniture detection ──────────────────────────────────────
        detections = self._detect_furniture(frame)

        # ── Decide: use detections or fallback ────────────────────────────────
        if len(detections) < required:
            return self._fallback_grid(frame_h, frame_w, default_rows, default_cols)

        # ── Cluster into rows and build grid ──────────────────────────────────
        rows = self._cluster_rows(detections, frame_h)
        cells = []
        for r_idx, row_dets in enumerate(rows):
            # Sort left-to-right within each row
            row_dets_sorted = sorted(row_dets, key=lambda d: (d["bbox"][0] + d["bbox"][2]) / 2)
            for c_idx, det in enumerate(row_dets_sorted):
                x1, y1, x2, y2 = det["bbox"]
                cells.append(GridCell(
                    row=r_idx,
                    col=c_idx,
                    label=f"R{r_idx + 1}C{c_idx + 1}",
                    bbox=[x1, y1, x2, y2],
                ))

        num_rows = len(rows)
        num_cols = max((len(r) for r in rows), default=1)
        return SpatialGrid(cells=cells, rows=num_rows, cols=num_cols)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _detect_furniture(self, frame) -> list:
        """Run non-tracking YOLO predict on the frame for furniture classes."""
        try:
            from core.detector import _get_yolo
            model = _get_yolo()
            results = model.predict(
                frame,
                classes=self.DESK_CLASSES,
                conf=self.DESK_CONF,
                verbose=False,
            )
            dets = []
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    dets.append({"bbox": [x1, y1, x2, y2], "confidence": float(box.conf[0])})
            return dets
        except Exception:
            return []

    def _cluster_rows(self, detections: list, frame_h: int) -> list:
        """
        Group detections into rows by Y-center proximity.
        Gap threshold = ROW_GAP_FRAC × frame_h.
        Returns list of lists (each inner list = one row's detections).
        """
        if not detections:
            return []

        gap = self.ROW_GAP_FRAC * frame_h
        # Sort by Y-center
        sorted_dets = sorted(detections, key=lambda d: (d["bbox"][1] + d["bbox"][3]) / 2)

        rows = [[sorted_dets[0]]]
        for det in sorted_dets[1:]:
            cy = (det["bbox"][1] + det["bbox"][3]) / 2
            last_cy = (rows[-1][-1]["bbox"][1] + rows[-1][-1]["bbox"][3]) / 2
            if abs(cy - last_cy) <= gap:
                rows[-1].append(det)
            else:
                rows.append([det])
        return rows

    def _fallback_grid(self, frame_h: int, frame_w: int, rows: int, cols: int) -> SpatialGrid:
        """
        Divide the frame uniformly into rows × cols cells.
        Each cell gets a label R{r+1}C{c+1} and a pixel bbox.
        """
        cell_w = frame_w // cols
        cell_h = frame_h // rows
        cells = []
        for r in range(rows):
            for c in range(cols):
                x1 = c * cell_w
                y1 = r * cell_h
                x2 = x1 + cell_w if c < cols - 1 else frame_w
                y2 = y1 + cell_h if r < rows - 1 else frame_h
                cells.append(GridCell(
                    row=r,
                    col=c,
                    label=f"R{r + 1}C{c + 1}",
                    bbox=[x1, y1, x2, y2],
                ))
        return SpatialGrid(cells=cells, rows=rows, cols=cols)
