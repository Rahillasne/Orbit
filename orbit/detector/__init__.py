"""Failure detection subsystem for deployment-time anomaly detection."""

# Phase-2 detectors (preferred)
from orbit.detector.heuristic import (
    BaseDetector,
    DetectorPipeline,
    FailureDetection,
    GripperDropConfig,
    GripperDropDetector,
    OutOfBoundsConfig,
    OutOfBoundsDetector,
    PipelineResult,
    RewardThresholdConfig,
    RewardThresholdDetector,
    StallConfig,
    StallDetector,
    TimeoutConfig,
    TimeoutDetector,
    load_pipeline_from_yaml,
)

# Legacy (deprecated -- kept for backward compat)
from orbit.detector.legacy import (
    DetectionResult,
    DetectorConfig,
    HeuristicDetector,
)

__all__ = [
    # Phase 2
    "BaseDetector",
    "DetectorPipeline",
    "FailureDetection",
    "GripperDropConfig",
    "GripperDropDetector",
    "OutOfBoundsConfig",
    "OutOfBoundsDetector",
    "PipelineResult",
    "RewardThresholdConfig",
    "RewardThresholdDetector",
    "StallConfig",
    "StallDetector",
    "TimeoutConfig",
    "TimeoutDetector",
    "load_pipeline_from_yaml",
    # Legacy
    "DetectionResult",
    "DetectorConfig",
    "HeuristicDetector",
]
