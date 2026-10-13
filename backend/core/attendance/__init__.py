"""
SmartClass AI — Attendance Domain Package
Handles temporal confirmation windows, verification tiers, presence tracking, and student attendance persistence.
"""
from core.attendance.attendance_engine import (
    AttendanceDecisionEngine,
    attendance_engine,
)

# Aliases for compatibility
AttendanceEngine = AttendanceDecisionEngine

__all__ = [
    "AttendanceDecisionEngine",
    "AttendanceEngine",
    "attendance_engine",
]
