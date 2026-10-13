"""
SmartClass AI — Pipeline Tier Scheduler
Controls cadence and execution policies across perceptual and cognitive stages:
- FAST Tier (every frame): YOLO detection + BoT-SORT tracking + spatial association
- MEDIUM Tier (every k frames): MediaPipe gaze/head pose + object containment checks
- SLOW Tier (every n frames): Face recognition / identification
- ASYNC Tier: VLM scene interpretation / summary reasoning
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class SchedulerConfig:
    """Configures frame intervals for tiered execution."""
    fast_cadence: int = 1       # Run every frame
    medium_cadence: int = 1     # Run face mesh / pose (set >1 to throttle on low-end hardware)
    slow_cadence: int = 30      # Run face recognition match every 30 frames
    async_reasoning_interval: int = 90  # VLM reasoning every 90 frames (~3 sec)


class PipelineScheduler:
    """Decides which stages to execute for a given frame index."""

    def __init__(self, config: SchedulerConfig | None = None):
        self.config = config or SchedulerConfig()

    def should_run_fast(self, frame_idx: int) -> bool:
        return frame_idx % self.config.fast_cadence == 0

    def should_run_medium(self, frame_idx: int) -> bool:
        return frame_idx % self.config.medium_cadence == 0

    def should_run_slow(self, frame_idx: int) -> bool:
        return frame_idx % self.config.slow_cadence == 0

    def should_run_async_reasoning(self, frame_idx: int) -> bool:
        return frame_idx % self.config.async_reasoning_interval == 0
