"""
SmartClass AI — Student Behavior & Engagement Domain Package
Tracks head poses, hand raises, phone usage, posture anomalies, and real-time student temporal state.
"""
from core.behavior.behavior_tracker import BehaviorTracker
from core.behavior.student_state import StudentState

__all__ = [
    "BehaviorTracker",
    "StudentState",
]
