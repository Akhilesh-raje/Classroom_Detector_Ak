"""
SmartClass AI — Core Enterprise Architecture
Modular Domain Packages:
  - attendance:    Temporal confirmation windows, verification tiers, presence tracking
  - spatial:       Seating topology, coordinate normalization, anchor stabilization
  - detection:     YOLO detection, ByteTrack tracking, face crop extraction
  - recognition:   InsightFace identity matching, cosine embeddings
  - behavior:      Head pose, engagement, activity tracking, student state
  - analytics:     Classroom session reports, Ollama multimodal labeling
  - storage:       SQLite ORM models, migrations, persistence transactions
  - visualization: HUD overlays, bounding box annotations, matrix renderers
  - pipeline:      Multi-stage processing pipeline & comprehensive analyzers
  - structure:     Topology builder & 3D geometric scene structure
  - scene:         Scene state representation & entity relationship tracking
"""
import sys
import os

from core.config import *
from core.engine import ClassroomEngine, get_logger

# Domain re-exports
from core.attendance import AttendanceDecisionEngine, AttendanceEngine, attendance_engine
from core.spatial import ClassroomTopology, TopologyConfig, SeatAssociator, SeatStabilizer
from core.detection import detect_faces, track_persons
from core.recognition import FaceEngine, face_engine, FaceRecognitionEngine, face_rec_engine
from core.behavior import BehaviorTracker, StudentState
from core.analytics import ReportGenerator, ollama_labeler
from core.storage import init_db, get_db, Student, AttendanceRecord, ActivityLog, SessionLocal
from core.visualization import draw_hud, draw_student_box, draw_seating_chart

__all__ = [
    "ClassroomEngine",
    "get_logger",
    "attendance",
    "spatial",
    "detection",
    "recognition",
    "behavior",
    "analytics",
    "storage",
    "visualization",
    "AttendanceDecisionEngine",
    "AttendanceEngine",
    "attendance_engine",
    "ClassroomTopology",
    "TopologyConfig",
    "SeatAssociator",
    "SeatStabilizer",
    "detect_faces",
    "track_persons",
    "FaceEngine",
    "face_engine",
    "FaceRecognitionEngine",
    "face_rec_engine",
    "BehaviorTracker",
    "StudentState",
    "ReportGenerator",
    "ollama_labeler",
    "init_db",
    "get_db",
    "Student",
    "AttendanceRecord",
    "ActivityLog",
    "SessionLocal",
    "draw_hud",
    "draw_student_box",
    "draw_seating_chart",
]
