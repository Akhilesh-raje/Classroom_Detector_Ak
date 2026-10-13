"""
SmartClass AI — SQLAlchemy Database Models & Connections
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, LargeBinary, Text, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "smartclass.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class SeatingConfig(Base):
    """
    Describes the physical layout of one classroom.
    layout_type: "bench" (shared long bench, 2-3 per bench) or "desk" (one desk per student).
    For bench layout every bench can have positions: L / M / R.
    For desk layout positions are ignored and grid_cell (R1C1) is used directly.
    One active config per classroom at a time.
    """
    __tablename__ = "seating_configs"
    id           = Column(Integer, primary_key=True, autoincrement=True)
    name         = Column(String(100), nullable=False, default="Default")   # e.g. "Room 101"
    layout_type  = Column(String(10),  nullable=False, default="bench")     # "bench" | "desk"
    num_rows     = Column(Integer,     nullable=False, default=4)
    benches_per_row = Column(Integer,  nullable=False, default=3)           # benches or desk-cols
    seats_per_bench = Column(Integer,  nullable=False, default=2)           # 1 for desk layout
    is_active    = Column(Integer,     nullable=False, default=1)           # 1 = active
    created_at   = Column(DateTime,    default=datetime.utcnow)


class Student(Base):
    __tablename__ = "students"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    roll_no = Column(String(50), unique=True, nullable=False, index=True)
    section = Column(String(50), nullable=False)
    email = Column(String(100), nullable=True)
    face_encoding = Column(LargeBinary, nullable=True)  # numpy float32 array as raw bytes
    face_image = Column(Text, nullable=True)             # base64 JPEG thumbnail
    # Seat location — canonical format depends on layout_type:
    #   bench layout  → "R1B2-L" (row 1, bench 2, left position)
    #   desk layout   → "R1C2"   (row 1, col 2 — backward-compatible)
    grid_cell    = Column(String(20), nullable=True)
    seat_row     = Column(Integer,  nullable=True)   # 1-indexed row number
    seat_bench   = Column(Integer,  nullable=True)   # 1-indexed bench/col number
    seat_pos     = Column(String(5), nullable=True)  # "L", "M", "R" or None for desk layout
    registered_at = Column(DateTime, default=datetime.utcnow)
    total_attendance = Column(Integer, default=0)
    avg_attentiveness = Column(Float, default=0.0)

class AttendanceRecord(Base):
    __tablename__ = "attendance"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    student_id = Column(Integer, nullable=True, index=True)  # Nullable if unknown occupant pending review
    date = Column(String(10), nullable=False, index=True)  # YYYY-MM-DD
    marked_at = Column(DateTime, default=datetime.utcnow)
    confidence = Column(Float, default=0.0)
    captured_image = Column(Text, nullable=True)  # base64 snapshot
    method = Column(String(30), default="ai")  # biometric_verified, seat_candidate, manual, unresolved_occupant

    # Architectural separation of presence, candidate, and verified identity:
    session_id           = Column(String(36), nullable=True, index=True)
    seat_id              = Column(String(20), nullable=True) # e.g. R1C1
    status               = Column(String(30), default="VERIFIED") # VERIFIED, PENDING_REVIEW, REJECTED, MANUAL
    presence_state       = Column(String(30), default="IDENTIFIED") # DETECTED, CANDIDATE, IDENTIFIED, UNKNOWN_OCCUPANT
    detection_count      = Column(Integer, default=1) # number of consecutive confirmed frames
    candidate_student_id = Column(Integer, nullable=True) # if seat suggests a candidate
    audit_trail          = Column(Text, nullable=True) # JSON evidence log
    is_dry_run           = Column(Integer, default=0) # 1 if dry run replay, 0 if persisted

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
    # ── SQLite column migration ───────────────────────────────────────────────
    # create_all won't add columns to existing tables; do it explicitly.
    _migrate_add_columns()


def _migrate_add_columns():
    """Add any missing columns to existing tables (safe to run repeatedly)."""
    migrations = {
        "students": [
            ("grid_cell",  "VARCHAR(20)"),
            ("seat_row",   "INTEGER"),
            ("seat_bench", "INTEGER"),
            ("seat_pos",   "VARCHAR(5)"),
        ],
        "attendance": [
            ("session_id",           "VARCHAR(36)"),
            ("seat_id",              "VARCHAR(20)"),
            ("status",               "VARCHAR(30) DEFAULT 'VERIFIED'"),
            ("presence_state",       "VARCHAR(30) DEFAULT 'IDENTIFIED'"),
            ("detection_count",      "INTEGER DEFAULT 1"),
            ("candidate_student_id", "INTEGER"),
            ("audit_trail",          "TEXT"),
            ("is_dry_run",           "INTEGER DEFAULT 0"),
        ],
    }
    with engine.connect() as conn:
        for table, cols in migrations.items():
            existing = {row[1] for row in conn.execute(
                text(f"PRAGMA table_info({table})")
            )}
            for col_name, col_type in cols:
                if col_name not in existing:
                    conn.execute(text(
                        f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}"
                    ))
            conn.commit()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
