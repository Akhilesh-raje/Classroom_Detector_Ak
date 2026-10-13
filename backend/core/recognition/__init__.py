"""
SmartClass AI — Biometric Identity Recognition Domain Package
Handles face embedding extraction, InsightFace recognition, gallery matching, and cosine similarity scoring.
"""
from core.recognition.face_engine import (
    FaceEngine,
    face_engine,
)
from core.recognition.face_recognition_engine import (
    FaceRecognitionEngine,
    face_rec_engine,
)

__all__ = [
    "FaceEngine",
    "face_engine",
    "FaceRecognitionEngine",
    "face_rec_engine",
]
