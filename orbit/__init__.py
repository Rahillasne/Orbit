"""Orbit: Robotics/ML debugging toolkit."""

__version__ = "1.0.0"


def __getattr__(name: str):
    """Lazy-import heavy modules so that ``import orbit`` doesn't require
    the full ML stack (torch, open_clip, etc.) to be installed.
    """
    _imports = {
        "EpisodeLogger": "orbit.logger.episode_logger",
        "HeuristicDetector": "orbit.detector.legacy",
        "DetectorPipeline": "orbit.detector.heuristic",
        "EmbeddingGapAnalyzer": "orbit.analyzer.embedding_gap",
        "EmbeddingAnalyzer": "orbit.analyzer.embedding_analyzer",
        "Prescriber": "orbit.prescriber.prescriber",
        "FailureDescriber": "orbit.vlm.failure_describer",
    }
    if name in _imports:
        import importlib

        module = importlib.import_module(_imports[name])
        return getattr(module, name)
    raise AttributeError(f"module 'orbit' has no attribute {name!r}")


__all__ = [
    "EpisodeLogger",
    "HeuristicDetector",
    "DetectorPipeline",
    "EmbeddingGapAnalyzer",
    "EmbeddingAnalyzer",
    "Prescriber",
    "FailureDescriber",
]
