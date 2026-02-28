from orbit.logger.compat import episode_to_legacy, legacy_to_episode
from orbit.logger.episode_logger import EpisodeLogger
from orbit.logger.schemas import (
    DeploymentSession,
    Episode,
    EpisodeFrame,
    EpisodeRecord,
    LoggerConfig,
    Outcome,
    StepRecord,
    StorageFormat,
)

__all__ = [
    "EpisodeLogger",
    # New models
    "Episode",
    "EpisodeFrame",
    "DeploymentSession",
    "LoggerConfig",
    "Outcome",
    # Legacy (deprecated)
    "EpisodeRecord",
    "StepRecord",
    "StorageFormat",
    # Compat
    "episode_to_legacy",
    "legacy_to_episode",
]
