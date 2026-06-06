"""
SmartClass AI — SQLAlchemy Database Models & Connections
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, LargeBinary, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "smartclass.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Student(Base):
    __tablename__ = "students"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    roll_no = Column(String(50), unique=True, nullable=False, index=True)
    section = Column(String(50), nullable=False)
    email = Column(String(100), nullable=True)
    face_encoding = Column(LargeBinary, nullable=True)  # numpy array as bytes
    face_image = Column(Text, nullable=True)  # base64 thumbnail
    grid_cell = Column(String(20), nullable=True)  # e.g. "R1C2"
    registered_at = Column(DateTime, default=datetime.utcnow)
    total_attendance = Column(Integer, default=0)
    avg_attentiveness = Column(Float, default=0.0)

class AttendanceRecord(Base):
    __tablename__ = "attendance"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    student_id = Column(Integer, nullable=False, index=True)
    date = Column(String(10), nullable=False, index=True)  # YYYY-MM-DD
    marked_at = Column(DateTime, default=datetime.utcnow)
    confidence = Column(Float, default=0.0)
    captured_image = Column(Text, nullable=True)  # base64 snapshot
    method = Column(String(20), default="ai")  # ai or manual

class ActivityLog(Base):
    __tablename__ = "activity_logs"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    student_id = Column(Integer, nullable=True)
    student_name = Column(String(100), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    activity = Column(String(50))  # attentive, distracted, drowsy, talking, phone
    attention_score = Column(Float, default=0.0)
    emotion = Column(String(30), nullable=True)
    head_yaw = Column(Float, nullable=True)
    head_pitch = Column(Float, nullable=True)
    eye_openness = Column(Float, nullable=True)

class Session(Base):
    __tablename__ = "sessions"
    session_id          = Column(String(36), primary_key=True)          # UUID string
    session_name        = Column(String(200), nullable=False)
    start_time          = Column(DateTime, nullable=False)
    end_time            = Column(DateTime, nullable=True)
    status              = Column(String(20), default="active")          # "active" or "complete"
    total_frames        = Column(Integer, default=0)
    student_count       = Column(Integer, default=0)
    avg_class_attention = Column(Float, default=0.0)

class SessionReport(Base):
    __tablename__ = "session_reports"
    id         = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), nullable=False, index=True)
    student_id = Column(Integer, nullable=True, index=True)
    report_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
