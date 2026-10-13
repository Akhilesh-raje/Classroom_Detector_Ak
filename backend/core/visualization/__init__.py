"""
SmartClass AI — Computer Vision Visualization Domain Package
Handles HUD overlays, bounding box rendering, seat occupancy markers, and real-time streaming annotations.
"""
from core.visualization.visualizer import (
    draw_hud,
    draw_student_box,
    draw_seating_chart,
    _compute_row_col,
    COLOR_MAP,
)

__all__ = [
    "draw_hud",
    "draw_student_box",
    "draw_seating_chart",
    "_compute_row_col",
    "COLOR_MAP",
]
