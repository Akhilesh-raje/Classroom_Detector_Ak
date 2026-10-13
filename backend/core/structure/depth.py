# -*- coding: utf-8 -*-
"""
SmartClass Structure AI — Depth Perception Module
=================================================
Wraps monocular depth estimation (Depth Anything V2) to extract continuous
relative depth fields, enabling physical depth-layer reasoning.
"""
from __future__ import annotations
import gc
import logging
from typing import Optional, Tuple, List, Union
import numpy as np
import cv2
import torch
from PIL import Image

log = logging.getLogger("structure.depth")


class MonocularDepthEstimator:
    """
    Relative monocular depth estimator using Depth Anything V2.
    Produces depth maps where higher values correspond to points closer to the camera.
    """

    def __init__(self, model_name: str = "depth-anything/Depth-Anything-V2-Small-hf", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipe = None
        self._last_depth_map: Optional[np.ndarray] = None

    def _ensure_loaded(self):
        if self._pipe is None:
            log.info(f"Loading Depth Anything V2 model ({self.model_name}) on {self.device}...")
            from transformers import pipeline
            self._pipe = pipeline(
                task="depth-estimation",
                model=self.model_name,
                device=0 if self.device == "cuda" else -1
            )
            log.info("Depth Anything V2 model loaded successfully.")

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimates relative depth from a single BGR or RGB frame.
        Returns:
            Normalized 2D float32 array of shape (H, W), values in [0.0, 1.0]
            where 1.0 is closest foreground and 0.0 is farthest background.
        """
        self._ensure_loaded()
        H, W = frame.shape[:2]

        if len(frame.shape) == 3 and frame.shape[2] == 3:
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_img = frame

        pil_img = Image.fromarray(rgb_img)
        res = self._pipe(pil_img)
        raw_depth = np.array(res["depth"], dtype=np.float32)

        # Normalize depth map to [0.0, 1.0]
        d_min, d_max = float(raw_depth.min()), float(raw_depth.max())
        if d_max - d_min > 1e-6:
            norm_depth = (raw_depth - d_min) / (d_max - d_min)
        else:
            norm_depth = np.zeros_like(raw_depth)

        # Resize to original frame resolution
        if norm_depth.shape != (H, W):
            norm_depth = cv2.resize(norm_depth, (W, H), interpolation=cv2.INTER_LINEAR)

        self._last_depth_map = norm_depth
        return norm_depth

    def sample_point(self, point: Tuple[float, float], depth_map: Optional[np.ndarray] = None) -> float:
        """Samples normalized depth at (x, y)."""
        dmap = depth_map if depth_map is not None else self._last_depth_map
        if dmap is None:
            return 0.5
        H, W = dmap.shape
        x = max(0, min(W - 1, int(round(point[0]))))
        y = max(0, min(H - 1, int(round(point[1]))))
        return float(dmap[y, x])

    def sample_region(self, bbox: List[int], depth_map: Optional[np.ndarray] = None) -> float:
        """
        Computes median depth inside a bounding box [x1, y1, x2, y2].
        """
        dmap = depth_map if depth_map is not None else self._last_depth_map
        if dmap is None:
            return 0.5
        H, W = dmap.shape
        x1 = max(0, min(W - 1, int(bbox[0])))
        y1 = max(0, min(H - 1, int(bbox[1])))
        x2 = max(x1 + 1, min(W, int(bbox[2])))
        y2 = max(y1 + 1, min(H, int(bbox[3])))
        
        region = dmap[y1:y2, x1:x2]
        return float(np.median(region)) if region.size > 0 else 0.5

    def sample_mask(self, mask: np.ndarray, depth_map: Optional[np.ndarray] = None) -> float:
        """
        Computes median depth over active pixels in a binary mask.
        """
        dmap = depth_map if depth_map is not None else self._last_depth_map
        if dmap is None:
            return 0.5
        vals = dmap[mask > 0]
        return float(np.median(vals)) if vals.size > 0 else 0.5

    def release(self):
        """Unload pipeline to free GPU memory after calibration."""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            log.info("Depth estimator released from memory.")
