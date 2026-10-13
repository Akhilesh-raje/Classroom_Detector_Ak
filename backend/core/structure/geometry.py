# -*- coding: utf-8 -*-
"""
SmartClass Structure AI — Computational Geometry & Spatial Reasoning
===================================================================
Provides geometric utilities for:
  - PCA principal axis estimation on desk/table masks
  - Longitudinal axis calculation for benches
  - Polygon containment, distances, and IoU
  - Monocular depth layer clustering
  - Perspective-aware spatial subdivision
"""
from __future__ import annotations
import math
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import cv2


def compute_pca_axis(points: np.ndarray) -> Dict[str, Any]:
    """
    Computes principal orientation axis, centroid, and dimensions via PCA.
    Args:
        points: (N, 2) array of (x, y) coordinates.
    Returns:
        Dict with centroid (cx, cy), principal_vector (vx, vy),
        angle_deg (orientation relative to horizontal),
        length (spread along principal axis), width (spread along orthogonal axis).
    """
    if len(points) < 3:
        return {
            "centroid": (0.0, 0.0),
            "vector": (1.0, 0.0),
            "angle_deg": 0.0,
            "length": 0.0,
            "width": 0.0,
        }

    points = points.astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(points, mean=None)
    
    cx, cy = float(mean[0, 0]), float(mean[0, 1])
    vx, vy = float(eigenvectors[0, 0]), float(eigenvectors[0, 1])
    
    # Project points onto principal and orthogonal axes
    diff = points - np.array([[cx, cy]], dtype=np.float32)
    proj_major = np.dot(diff, np.array([vx, vy], dtype=np.float32))
    ortho_vector = np.array([-vy, vx], dtype=np.float32)
    proj_minor = np.dot(diff, ortho_vector)

    length = float(proj_major.max() - proj_major.min())
    width = float(proj_minor.max() - proj_minor.min())
    angle_deg = float(np.degrees(np.arctan2(vy, vx)))

    return {
        "centroid": (cx, cy),
        "vector": (vx, vy),
        "angle_deg": angle_deg,
        "length": length,
        "width": width,
        "min_proj": float(proj_major.min()),
        "max_proj": float(proj_major.max()),
    }


def mask_to_polygon(mask: np.ndarray, min_area: float = 200.0, epsilon_factor: float = 0.015) -> Optional[List[Tuple[int, int]]]:
    """
    Converts a binary mask into an approximate simplified polygon.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # Take largest contour by area
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None
        
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon_factor * peri, True)
    
    poly = [(int(pt[0][0]), int(pt[0][1])) for pt in approx]
    return poly if len(poly) >= 3 else None


def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[int, int]]) -> bool:
    """
    Tests if (x, y) is inside or on the boundary of polygon.
    """
    if len(polygon) < 3:
        return False
    contour = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    dist = cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), measureDist=False)
    return dist >= 0


def point_polygon_distance(point: Tuple[float, float], polygon: List[Tuple[int, int]]) -> float:
    """
    Signed distance from point to polygon in pixels.
    Positive: inside polygon.
    Negative: outside polygon.
    """
    if len(polygon) < 3:
        return -999.0
    contour = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    return float(cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), measureDist=True))


def bbox_iou(box1: List[int], box2: List[int]) -> float:
    """
    Computes Intersection over Union for two bounding boxes [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    return float(inter_area / union_area) if union_area > 0 else 0.0


def cluster_depth_layers_otsu(depth_values: List[float]) -> Tuple[float, List[int]]:
    """
    Uses 1D variance minimization (Otsu's principle) to partition depth values
    into two distinct layers:
      Cluster 0: Foreground (larger depth / closer)
      Cluster 1: Background (smaller depth / farther)
    Returns:
        (optimal_threshold, list of cluster labels 0 or 1)
    """
    arr = np.asarray(depth_values, dtype=np.float32).ravel()
    if len(arr) < 2:
        return 0.5, [0] * len(arr)

    sorted_vals = np.sort(arr).tolist()
    n = len(sorted_vals)
    total_var = float(np.var(sorted_vals))
    if total_var < 1e-6:
        return float(sorted_vals[n // 2]), [0] * n

    best_thresh = float(sorted_vals[0])
    best_variance_between = 0.0

    for i in range(1, n):
        w0 = i / n
        w1 = (n - i) / n
        mean0 = float(np.mean(sorted_vals[:i]))
        mean1 = float(np.mean(sorted_vals[i:]))
        var_between = w0 * w1 * ((mean0 - mean1) ** 2)

        if var_between > best_variance_between:
            best_variance_between = var_between
            best_thresh = (sorted_vals[i - 1] + sorted_vals[i]) / 2.0

    labels = [0 if float(v) >= best_thresh else 1 for v in arr]
    return best_thresh, labels


def estimate_classroom_perspective_matrix(
    src_corners: Optional[List[Tuple[float, float]]] = None,
    frame_size: Tuple[int, int] = (1920, 1080)
) -> np.ndarray:
    """
    Computes a 3x3 perspective homography matrix mapping oblique image-space
    trapezoid to normalized rectangular classroom coordinates [0, 1000] x [0, 600].
    """
    W, H = frame_size
    if src_corners is None or len(src_corners) != 4:
        # Default perspective calibration trapezoid for classroom:
        # Top-Left, Top-Right, Bottom-Right, Bottom-Left
        src_corners = [
            (160.0, 440.0),   # Back-Left
            (1620.0, 450.0),  # Back-Right
            (1910.0, 750.0),  # Front-Right
            (280.0, 710.0),   # Front-Left
        ]
    src_pts = np.array(src_corners, dtype=np.float32)
    dst_pts = np.array([
        [50.0, 100.0],    # Normalized Back-Left
        [950.0, 100.0],   # Normalized Back-Right
        [950.0, 500.0],   # Normalized Front-Right
        [50.0, 500.0],    # Normalized Front-Left
    ], dtype=np.float32)

    return cv2.getPerspectiveTransform(src_pts, dst_pts)


def transform_points_perspective(points: np.ndarray, H_mat: np.ndarray) -> np.ndarray:
    """
    Transforms (N, 2) image coordinates to normalized perspective space via homography.
    """
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, H_mat)
    return transformed.reshape(-1, 2)


def score_topology_candidate(
    transformed_anchors: np.ndarray,
    candidate_rows: int = 2,
    candidate_cols: Tuple[int, int] = (6, 6)
) -> Dict[str, Any]:
    """
    Scores a candidate topology hypothesis (e.g. 6x6 vs 5x7) based on physical evidence:
      - Spacing regularity: lower variance of seat-to-seat distances along rows
      - Parallelism & row separation
      - Column alignment consistency
    """
    if len(transformed_anchors) < 8:
        return {"confidence": 0.5, "score": 0.5, "candidate": candidate_cols}

    # Split into 2 rows along normalized V (Y) axis
    v_coords = transformed_anchors[:, 1]
    mid_v = float(np.median(v_coords))
    row1_mask = v_coords > mid_v   # Front row (larger V)
    row2_mask = ~row1_mask         # Back row (smaller V)

    r1_pts = transformed_anchors[row1_mask]
    r2_pts = transformed_anchors[row2_mask]

    n_r1 = len(r1_pts)
    n_r2 = len(r2_pts)

    # 1. Count penalty vs candidate shape
    target_r1, target_r2 = candidate_cols
    count_error = abs(n_r1 - target_r1) + abs(n_r2 - target_r2)
    count_score = max(0.0, 1.0 - (count_error / 12.0))

    # 2. Spacing regularity (standard deviation of horizontal delta)
    def _spacing_variance(pts):
        if len(pts) < 3:
            return 1.0
        xs = np.sort(pts[:, 0])
        diffs = np.diff(xs)
        if len(diffs) == 0 or np.mean(diffs) < 1e-4:
            return 1.0
        cv = float(np.std(diffs) / np.mean(diffs)) # Coefficient of variation
        return max(0.0, 1.0 - cv)

    r1_reg = _spacing_variance(r1_pts)
    r2_reg = _spacing_variance(r2_pts)
    regularity_score = (r1_reg + r2_reg) / 2.0

    # 3. Overall composite confidence
    composite = 0.50 * count_score + 0.50 * regularity_score

    return {
        "candidate": candidate_cols,
        "score": round(float(composite), 3),
        "count_score": round(float(count_score), 3),
        "regularity_score": round(float(regularity_score), 3),
        "row_counts": (n_r1, n_r2)
    }

