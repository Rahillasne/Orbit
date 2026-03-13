"""Orbit profile module.

Score dataset readiness, map coverage gaps, and generate data collection prescriptions.
"""

from orbit.profile.types import (
    ActionStats,
    CapabilityScore,
    CoverageMap,
    DatasetGap,
    DatasetProfile,
    DatasetReportCard,
    EmbeddingIndex,
    EmbeddingStats,
    Prescription,
    QualityMetrics,
    TaskAssessment,
)

_LAZY_IMPORTS = {
    "DatasetProfiler": ("orbit.profile.profiler", "DatasetProfiler"),
    "CapabilityScorer": ("orbit.profile.capability", "CapabilityScorer"),
    "ProfileReporter": ("orbit.profile.report", "ProfileReporter"),
    "DatasetLoader": ("orbit.profile.loaders", "DatasetLoader"),
    "ReportCardGenerator": ("orbit.profile.report_card", "ReportCardGenerator"),
    "DatasetFeatureExtractor": ("orbit.profile.feature_extractor", "DatasetFeatureExtractor"),
    "FeatureScaler": ("orbit.profile.feature_extractor", "FeatureScaler"),
    "DatasetQualityModel": ("orbit.profile.predictor", "DatasetQualityModel"),
    "DatasetQualityModelV2": ("orbit.profile.predictor_v2", "DatasetQualityModelV2"),
    "Prediction": ("orbit.profile.predictor_v2", "Prediction"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module_path, attr = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module 'orbit.profile' has no attribute {name!r}")


__all__ = [
    "DatasetProfiler",
    "DatasetLoader",
    "CapabilityScorer",
    "ProfileReporter",
    "ReportCardGenerator",
    "DatasetFeatureExtractor",
    "FeatureScaler",
    "DatasetProfile",
    "DatasetReportCard",
    "CapabilityScore",
    "CoverageMap",
    "QualityMetrics",
    "EmbeddingIndex",
    "EmbeddingStats",
    "ActionStats",
    "DatasetGap",
    "TaskAssessment",
    "Prescription",
    "DatasetQualityModel",
    "DatasetQualityModelV2",
    "Prediction",
]
