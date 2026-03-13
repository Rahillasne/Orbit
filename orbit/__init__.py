"""Orbit: Analyze robot datasets to predict task capability and prescribe data collection."""

__version__ = "1.2.0"


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
        "DatasetProfiler": "orbit.profile.profiler",
        "Sim2RealProfiler": "orbit.sim2real_profiler",
        "LeRobotAdapter": "orbit.adapters.lerobot_adapter",
        "RobomimicAdapter": "orbit.adapters.robomimic_adapter",
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
    "DatasetProfiler",
    "Sim2RealProfiler",
    "LeRobotAdapter",
    "RobomimicAdapter",
]
