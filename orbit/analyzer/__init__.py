"""Orbit analyzer module.

Backward-compatible: ``EmbeddingGapAnalyzer``, ``AnalyzerConfig``, and
``GapAnalysisResult`` still work.  New code should use
:class:`EmbeddingAnalyzer` and the models in :mod:`orbit.analyzer.models`.
"""

# New model exports (lightweight, no heavy deps)
from orbit.analyzer.models import (
    AnalysisReport,
    EmbeddingAnalyzerConfig,
    EpisodeGapSummary,
    FailureCluster,
    FailureClusterReport,
    FrameGapResult,
)

# Legacy + heavy imports are lazy to avoid requiring open_clip / torch at
# import time.
_LAZY_IMPORTS = {
    "EmbeddingGapAnalyzer": ("orbit.analyzer.embedding_gap", "EmbeddingGapAnalyzer"),
    "AnalyzerConfig": ("orbit.analyzer.embedding_gap", "AnalyzerConfig"),
    "GapAnalysisResult": ("orbit.analyzer.embedding_gap", "GapAnalysisResult"),
    "EmbeddingAnalyzer": ("orbit.analyzer.embedding_analyzer", "EmbeddingAnalyzer"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module_path, attr = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module 'orbit.analyzer' has no attribute {name!r}")


__all__ = [
    # Legacy
    "EmbeddingGapAnalyzer",
    "AnalyzerConfig",
    "GapAnalysisResult",
    # New
    "EmbeddingAnalyzer",
    "EmbeddingAnalyzerConfig",
    "AnalysisReport",
    "EpisodeGapSummary",
    "FrameGapResult",
    "FailureCluster",
    "FailureClusterReport",
]
