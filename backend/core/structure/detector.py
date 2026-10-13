# -*- coding: utf-8 -*-
"""
SmartClass Structure AI — Furniture Detection Module
====================================================
Detects physical furniture (desks, benches, tables, chairs) using
open-vocabulary Grounding DINO and/or YOLO-seg models.
"""
from __future__ import annotations
import gc
import os
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import cv2
import torch
from PIL import Image

log = logging.getLogger("structure.detector")


@dataclass
class FurnitureDetection:
    """Detection candidate for classroom furniture."""
    id: str
    class_name: str
    confidence: float
    bbox: List[int]                     # [x1, y1, x2, y2]
    source: str                         # "grounding_dino" | "yolo_seg"
    mask: Optional[np.ndarray] = None   # Binary mask if available


class FurnitureDetector:
    """
    Detects physical classroom structures (desks, chairs, benches, tables).
    """

    def __init__(
        self,
        gd_model_id: str = "IDEA-Research/grounding-dino-tiny",
        yolo_model_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        self.gd_model_id = gd_model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = None
        self._gd_model = None
        self._yolo = None

        # Resolve YOLO model path if available
        if yolo_model_path is None:
            # Look in workspace root
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            candidates = [
                os.path.join(root_dir, "models", "yolo11s-seg.pt"),
                os.path.join(root_dir, "yolo11s-seg.pt"),
                os.path.join(root_dir, "models", "yolo11s.pt"),
                "yolo11s-seg.pt",
            ]
            for c in candidates:
                if os.path.exists(c):
                    self.yolo_model_path = c
                    break
            else:
                self.yolo_model_path = "yolo11s-seg.pt"
        else:
            self.yolo_model_path = yolo_model_path

    def _ensure_gd_loaded(self):
        if self._gd_model is None:
            log.info(f"Loading Grounding DINO ({self.gd_model_id}) on {self.device}...")
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            self._processor = AutoProcessor.from_pretrained(self.gd_model_id)
            self._gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.gd_model_id).to(self.device)
            log.info("Grounding DINO loaded successfully.")

    def _ensure_yolo_loaded(self):
        if self._yolo is None:
            from ultralytics import YOLO
            log.info(f"Loading YOLO model ({self.yolo_model_path})...")
            self._yolo = YOLO(self.yolo_model_path)
            log.info("YOLO model loaded.")

    def detect_grounding_dino(
        self,
        frame: np.ndarray,
        text_prompt: str = "classroom desk . student desk . wooden desk . table . chair . bench .",
        box_threshold: float = 0.20,
        text_threshold: float = 0.20,
    ) -> List[FurnitureDetection]:
        """Runs open-vocabulary detection using Grounding DINO."""
        self._ensure_gd_loaded()
        H, W = frame.shape[:2]

        if len(frame.shape) == 3 and frame.shape[2] == 3:
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_img = frame
        pil_img = Image.fromarray(rgb_img)

        inputs = self._processor(images=pil_img, text=text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._gd_model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(H, W)]
        )[0]

        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results.get("text_labels", results.get("labels", []))

        detections = []
        for idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            x1, y1, x2, y2 = [int(v) for v in box]
            detections.append(FurnitureDetection(
                id=f"gd_{idx}",
                class_name=str(label).strip().lower(),
                confidence=float(score),
                bbox=[x1, y1, x2, y2],
                source="grounding_dino",
                mask=None
            ))

        return detections

    def detect_yolo_seg(
        self,
        frame: np.ndarray,
        conf: float = 0.15,
        target_classes: Optional[List[str]] = None
    ) -> List[FurnitureDetection]:
        """Runs instance segmentation using YOLO11-seg."""
        self._ensure_yolo_loaded()
        H, W = frame.shape[:2]
        targets = set(target_classes or ["chair", "dining table", "bench", "table", "desk"])

        results = self._yolo.predict(frame, conf=conf, verbose=False)[0]
        detections = []
        names = results.names

        if results.masks is not None:
            masks_data = results.masks.data.cpu().numpy()
            boxes_data = results.boxes.data.cpu().numpy()

            for idx, (m, b) in enumerate(zip(masks_data, boxes_data)):
                x1, y1, x2, y2, c, cls_id = b
                cls_name = names[int(cls_id)].lower()
                if cls_name in targets:
                    full_mask = cv2.resize((m > 0.5).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                    detections.append(FurnitureDetection(
                        id=f"yolo_{idx}",
                        class_name=cls_name,
                        confidence=float(c),
                        bbox=[int(x1), int(y1), int(x2), int(y2)],
                        source="yolo_seg",
                        mask=full_mask
                    ))

        return detections

    def detect(self, frame: np.ndarray) -> List[FurnitureDetection]:
        """
        Combines YOLO-seg (for precise masks) and Grounding DINO (for open-vocabulary coverage).
        """
        detections = []
        # Run YOLO-seg first as it provides direct segmentation masks
        try:
            yolo_dets = self.detect_yolo_seg(frame)
            detections.extend(yolo_dets)
        except Exception as e:
            log.warning(f"YOLO-seg detection failed: {e}")

        # Run Grounding DINO for extra desk / table boundaries
        try:
            gd_dets = self.detect_grounding_dino(frame)
            detections.extend(gd_dets)
        except Exception as e:
            log.warning(f"Grounding DINO detection failed: {e}")

        return detections

    def release(self):
        """Releases heavy deep learning models from memory."""
        if self._gd_model is not None:
            del self._gd_model
            del self._processor
            self._gd_model = None
            self._processor = None
        if self._yolo is not None:
            del self._yolo
            self._yolo = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        log.info("Furniture detector models released.")
