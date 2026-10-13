"""
SmartClass AI — Perceptual Pipeline Package
Decomposes monolithic frame processing into composable, testable stages.
"""
from core.pipeline.detection_stage import DetectionStage
from core.pipeline.association_stage import AssociationStage
from core.pipeline.analysis_stage import AnalysisStage
from core.pipeline.output_stage import OutputStage
from core.pipeline.scheduler import PipelineScheduler, SchedulerConfig

__all__ = [
    "DetectionStage",
    "AssociationStage",
    "AnalysisStage",
    "OutputStage",
    "PipelineScheduler",
    "SchedulerConfig",
]
