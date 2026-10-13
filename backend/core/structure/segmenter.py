# -*- coding: utf-8 -*-
"""
SmartClass Structure AI — Furniture Mask & Polygon Segmenter
============================================================
Converts raw furniture detections into refined binary masks, convex hulls,
and bounding polygons.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np
import cv2

from core.structure.geometry import mask_to_polygon, compute_pca_axis

log = logging.getLogger("structure.segmenter")


@dataclass
class FurnitureMask:
    """Segmented physical furniture item with mask and boundary polygon."""
    id: str
    class_name: str
    confidence: float
    bbox: List[int]                     # [x1, y1, x2, y2]
    mask: np.ndarray                    # Binary mask (H, W) uint8 (0 or 1)
    polygon: List[Tuple[int, int]]      # Boundary polygon vertices
    centerline: Tuple[Tuple[float, float], Tuple[float, float]]  # (p1, p2)
    orientation_deg: float
    area_px: float
    mean_depth: float = 0.5


class FurnitureSegmenter:
    """Refines detections into structured geometric masks and polygons."""

    def __init__(self, min_area_px: float = 500.0):
        self.min_area_px = min_area_px

    def process_detections(
        self,
        frame: np.ndarray,
        detections: list,
        depth_map: Optional[np.ndarray] = None
    ) -> List[FurnitureMask]:
        """
        Takes raw FurnitureDetection objects and computes clean masks, polygons,
        and PCA centerlines.
        """
        H, W = frame.shape[:2]
        masks_out: List[FurnitureMask] = []

        for det in detections:
            # 1. Obtain binary mask (either from detection or synthetic box mask)
            if det.mask is not None:
                mask = (det.mask > 0).astype(np.uint8)
            else:
                x1, y1, x2, y2 = det.bbox
                mask = np.zeros((H, W), dtype=np.uint8)
                mask[y1:y2, x1:x2] = 1

            area = float(np.sum(mask))
            if area < self.min_area_px:
                continue

            # 2. Extract polygon
            polygon = mask_to_polygon(mask, min_area=self.min_area_px)
            if polygon is None or len(polygon) < 3:
                x1, y1, x2, y2 = det.bbox
                polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

            # 3. Compute PCA axis
            pts_y, pts_x = np.where(mask > 0)
            if len(pts_x) >= 10:
                coords = np.column_stack((pts_x, pts_y))
                pca_res = compute_pca_axis(coords)
                cx, cy = pca_res["centroid"]
                vx, vy = pca_res["vector"]
                hl = pca_res["length"] / 2.0
                p1 = (cx - vx * hl, cy - vy * hl)
                p2 = (cx + vx * hl, cy + vy * hl)
                orient = pca_res["angle_deg"]
            else:
                cx, cy = (det.bbox[0] + det.bbox[2]) / 2.0, (det.bbox[1] + det.bbox[3]) / 2.0
                p1, p2 = (float(det.bbox[0]), cy), (float(det.bbox[2]), cy)
                orient = 0.0

            # 4. Sample depth
            mean_d = 0.5
            if depth_map is not None:
                d_vals = depth_map[mask > 0]
                if len(d_vals) > 0:
                    mean_d = float(np.median(d_vals))

            masks_out.append(FurnitureMask(
                id=det.id,
                class_name=det.class_name,
                confidence=det.confidence,
                bbox=det.bbox,
                mask=mask,
                polygon=polygon,
                centerline=(p1, p2),
                orientation_deg=orient,
                area_px=area,
                mean_depth=mean_d
            ))

        return masks_out
