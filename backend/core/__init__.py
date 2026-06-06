"""
SmartClass AI — Core Package Interface
"""
from core.config import *
from core.student_state import StudentState
from core.stabilizer import SeatStabilizer
from core.detector import detect_faces, track_persons
from core.engine import ClassroomEngine, get_logger
from core.database import init_db, get_db, Student, AttendanceRecord, ActivityLog
