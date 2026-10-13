"""
SmartClass AI — Analytics & Reporting Domain Package
Handles classroom session summary reports, Ollama multimodal semantic labeling, and export metrics.
"""
from core.analytics.report_generator import ReportGenerator
from core.analytics.ollama_labeler import (
    OllamaLabeler,
    OllamaClassroomLabeler,
    ollama_labeler,
)

__all__ = [
    "ReportGenerator",
    "OllamaLabeler",
    "OllamaClassroomLabeler",
    "ollama_labeler",
]

