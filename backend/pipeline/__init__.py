"""
BCI processing pipeline components.
"""

from backend.pipeline.realtime_pipeline import RealtimePipeline
from backend.pipeline.sync_metrics import (
    FeatureFrame,
    SyncState,
    SyncMetricsCalculator,
)
from backend.pipeline.sync_manager import (
    SyncSession,
    SyncSessionConfig,
    SyncSessionPhase,
    SyncSessionState,
    SyncManager,
)

__all__ = [
    "RealtimePipeline",
    "FeatureFrame",
    "SyncState",
    "SyncMetricsCalculator",
    "SyncSession",
    "SyncSessionConfig",
    "SyncSessionPhase",
    "SyncSessionState",
    "SyncManager",
]

