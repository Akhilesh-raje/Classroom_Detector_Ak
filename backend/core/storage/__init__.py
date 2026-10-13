"""
SmartClass AI — Persistence & Storage Domain Package
Handles SQLite schema, models, session transactions, migrations, and database sessions.
"""
from core.storage.database import (
    Base,
    engine,
    SessionLocal,
    Student,
    AttendanceRecord,
    ActivityLog,
    Session,
    SessionReport,
    SeatingConfig,
    init_db,
    get_db,
)

__all__ = [
    "Base",
    "engine",
    "SessionLocal",
    "Student",
    "AttendanceRecord",
    "ActivityLog",
    "Session",
    "SessionReport",
    "SeatingConfig",
    "init_db",
    "get_db",
]
