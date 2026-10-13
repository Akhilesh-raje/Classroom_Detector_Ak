# -*- coding: utf-8 -*-
"""
SmartClass Structure AI Package
===============================
Provides furniture perception, monocular depth estimation, geometric reasoning,
and physical seat topology synthesis completely decoupled from person tracking.
"""
from core.structure.geometry import (
    compute_pca_axis,
    mask_to_polygon,
    point_in_polygon,
    point_polygon_distance,
    bbox_iou,
    cluster_depth_layers_otsu,
)
from core.structure.depth import MonocularDepthEstimator
from core.structure.detector import FurnitureDetector, FurnitureDetection
from core.structure.segmenter import FurnitureSegmenter, FurnitureMask
from core.structure.topology_builder import (
    Bench,
    PhysicalSeatGeometry,
    ClassroomTopologyMap,
    TopologyBuilder,
)
from core.structure.structure_engine import StructureEngine

__all__ = [
    "StructureEngine",
    "ClassroomTopologyMap",
    "Bench",
    "PhysicalSeatGeometry",
    "TopologyBuilder",
    "FurnitureDetector",
    "FurnitureDetection",
    "FurnitureSegmenter",
    "FurnitureMask",
    "MonocularDepthEstimator",
    "compute_pca_axis",
    "mask_to_polygon",
    "point_in_polygon",
    "point_polygon_distance",
    "bbox_iou",
    "cluster_depth_layers_otsu",
]
