"""
SmartClass AI — Computer Vision Detection Domain Package
Handles YOLO person detection, ByteTrack tracking, face crop extraction, and phone detection.
"""
from core.detection.detector import (
    detect_faces,
    track_persons,
    detect_faces_and_phones,
    analyze_face_crop,
    _get_yolo,
    _get_mp_detector,
)

__all__ = [
    "detect_faces",
    "track_persons",
    "detect_faces_and_phones",
    "analyze_face_crop",
    "_get_yolo",
    "_get_mp_detector",
]
