# -*- coding: utf-8 -*-
"""
SmartClass Structure AI — Structure Engine Facade
=================================================
Top-level coordinator for structural perception.
Runs detection, segmentation, depth estimation, and geometric synthesis
to produce a pure ClassroomTopologyMap.

Knows zero about:
  - student identity
  - face recognition
  - attention / activity
  - tracking IDs
"""
from __future__ import annotations
import os
import json
import logging
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import cv2

from core.structure.detector import FurnitureDetector, FurnitureDetection
from core.structure.segmenter import FurnitureSegmenter, FurnitureMask
from core.structure.depth import MonocularDepthEstimator
from core.structure.topology_builder import (
    TopologyBuilder,
    ClassroomTopologyMap,
    Bench,
    PhysicalSeatGeometry
)

log = logging.getLogger("structure.engine")


class StructureEngine:
    """
    Independent classroom structure perception engine.
    Analyzes raw classroom video frames (during calibration) and outputs
    a pure physical topology map.
    """

    def __init__(
        self,
        frame_size: Tuple[int, int] = (1920, 1080),
        yolo_model_path: Optional[str] = None,
        depth_model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
        device: Optional[str] = None
    ):
        self.frame_size = frame_size
        self.detector = FurnitureDetector(yolo_model_path=yolo_model_path, device=device)
        self.segmenter = FurnitureSegmenter()
        self.depth_estimator = MonocularDepthEstimator(model_name=depth_model_name, device=device)
        self.builder = TopologyBuilder(frame_size=frame_size)
        self._cached_topology: Optional[ClassroomTopologyMap] = None

    def analyze(self, frames: List[np.ndarray], release_after: bool = True) -> ClassroomTopologyMap:
        """
        Analyzes a sequence of calibration frames (e.g. 1-10 keyframes) to produce
        a permanent physical classroom topology.
        """
        if not frames:
            raise ValueError("StructureEngine.analyze requires at least 1 frame.")

        log.info(f"StructureEngine: Beginning spatial calibration over {len(frames)} keyframes...")
        ref_frame = frames[0]

        # 1. Depth Estimation
        log.info("StructureEngine: Estimating monocular depth field...")
        depth_map = self.depth_estimator.estimate(ref_frame)

        # 2. Furniture Detection (sample keyframes to avoid redundant static inference)
        log.info("StructureEngine: Detecting classroom furniture...")
        sample_frames = [frames[0]] if len(frames) <= 2 else [frames[0], frames[len(frames) // 2]]
        all_detections: List[FurnitureDetection] = []
        for f in sample_frames:
            dets = self.detector.detect(f)
            all_detections.extend(dets)

        # 3. Furniture Segmentation & Contour Extraction
        log.info("StructureEngine: Segmenting furniture boundaries & centerlines...")
        masks = self.segmenter.process_detections(ref_frame, all_detections, depth_map=depth_map)
        log.info(f"StructureEngine: Extracted {len(masks)} physical furniture masks.")

        # 4. Synthesize Physical Topology
        log.info("StructureEngine: Synthesizing bench groups and physical seats...")
        topology = self.builder.build_from_geometry(masks, depth_map=depth_map)
        log.info(f"StructureEngine: Successfully synthesized {len(topology.seats)} physical seats across {topology.rows} benches.")

        self._cached_topology = topology

        # 5. Free GPU memory if requested (so runtime Human AI gets full resources)
        if release_after:
            self.detector.release()
            self.depth_estimator.release()

        return topology

    def save_topology(self, topology: ClassroomTopologyMap, file_path: str):
        """Exports pure classroom topology to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(topology.to_dict(), f, indent=2)
        log.info(f"StructureEngine: Saved topology map to {file_path}")

    def load_topology(self, file_path: str) -> ClassroomTopologyMap:
        """Loads classroom topology from JSON."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        benches: List[Bench] = []
        for b_data in data.get("benches", []):
            bbox = b_data.get("bbox", [0, 0, 100, 100])
            poly = [(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])]
            benches.append(Bench(
                id=b_data["id"],
                row_id=b_data["row_id"],
                semantic=b_data.get("semantic", "FRONT"),
                polygon=poly,
                bbox=bbox,
                centerline=((float(bbox[0]), b_data.get("desk_edge_y", 600)), (float(bbox[2]), b_data.get("desk_edge_y", 600))),
                orientation_deg=0.0,
                depth_mean=float(b_data.get("depth_mean", 0.5)),
                desk_edge_y=float(b_data.get("desk_edge_y", 600))
            ))

        seats: List[PhysicalSeatGeometry] = []
        for s_data in data.get("seats", []):
            cx, cy = s_data["center"]
            bbox = s_data["bbox"]
            poly = [(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])]
            seats.append(PhysicalSeatGeometry(
                seat_id=s_data["seat_id"],
                bench_id=s_data.get("bench_id", "BENCH_01"),
                row_id=s_data.get("row_id", "R1"),
                semantic=s_data.get("semantic", "FRONT"),
                display_label=s_data["display_label"],
                center=(float(cx), float(cy)),
                polygon=poly,
                bbox=bbox,
                head_region=s_data.get("head_region", bbox),
                torso_region=s_data.get("torso_region", bbox),
                desk_contact_line=((float(bbox[0]), float(bbox[3])), (float(bbox[2]), float(bbox[3]))),
                expected_depth=float(s_data.get("expected_depth", 0.5)),
                confidence=float(s_data.get("confidence", 1.0)),
                evidence=s_data.get("evidence", {})
            ))

        topology = ClassroomTopologyMap(
            benches=benches,
            seats=seats,
            rows=len(benches),
            frame_size=tuple(data.get("frame_size", [1920, 1080])),
            source="loaded_from_json",
            confidence=float(data.get("confidence", 1.0))
        )
        self._cached_topology = topology
        return topology
