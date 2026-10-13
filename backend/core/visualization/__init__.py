"""
SmartClass AI — Computer Vision Visualization Domain Package
Handles HUD overlays, bounding box rendering, seat occupancy markers, and real-time streaming annotations.
"""
from core.visualization.visualizer import (
    draw_student_overlay,
    draw_global_hud,
    draw_compact_overlay,
    draw_full_overlay,
    draw_spatial_overlay,
    draw_behavioral_overlay,
    draw_grid_overlay,
    draw_grid_lines,
    render_classroom_frame,
    draw_hud,
    draw_student_box,
    draw_seating_chart,
    _compute_row_col,
    COLOR_MAP,
    C,
    ACT_COLOR,
    ACT_ICON,
)

__all__ = [
    "draw_student_overlay",
    "draw_global_hud",
    "draw_compact_overlay",
    "draw_full_overlay",
    "draw_spatial_overlay",
    "draw_behavioral_overlay",
    "draw_grid_overlay",
    "draw_grid_lines",
    "render_classroom_frame",
    "draw_hud",
    "draw_student_box",
    "draw_seating_chart",
    "_compute_row_col",
    "COLOR_MAP",
    "C",
    "ACT_COLOR",
    "ACT_ICON",
]

