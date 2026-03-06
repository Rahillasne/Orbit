"""Orbit profile module — dataset capability analysis."""

from orbit.profile.types import (
    CapabilityScore,
    CoverageMap,
    DatasetProfile,
    EmbeddingIndex,
    QualityMetrics,
)

_LAZY_IMPORTS = {
    "DatasetProfiler": ("orbit.profile.profiler", "DatasetProfiler"),
    "CapabilityScorer": ("orbit.profile.capability", "CapabilityScorer"),
    "ProfileReporter": ("orbit.profile.report", "ProfileReporter"),
    "DatasetLoader": ("orbit.profile.loaders", "DatasetLoader"),
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
    "DatasetProfile",
    "CapabilityScore",
    "CoverageMap",
    "QualityMetrics",
    "EmbeddingIndex",
]
