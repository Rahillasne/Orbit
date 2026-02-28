"""Pydantic models for episode logging data structures."""

from __future__ import annotations

import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Outcome(str, Enum):
    """Outcome of an episode."""

    SUCCESS = "success"
    FAILURE = "failure"
    UNKNOWN = "unknown"


class StorageFormat(str, Enum):
    """Supported storage backend formats (legacy)."""

    HDF5 = "hdf5"
    PARQUET = "parquet"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class LoggerConfig(BaseModel):
    """Configuration for the EpisodeLogger."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    storage_dir: str = "./orbit_data"
    task_name: str = "default_task"
    robot_id: str = "default"
    robot_dof: int = 6
    max_frames_per_episode: int = 10_000
    save_images: bool = True
    image_keys: list[str] = Field(default_factory=lambda: ["front"])
    fps: int = 30
    metadata: dict[str, Any] = Field(default_factory=dict)
    policy_checkpoint: str = ""
    policy_version: str = ""
    environment_description: str = ""
    lock_timeout: float = 10.0


# ---------------------------------------------------------------------------
# New models
# ---------------------------------------------------------------------------


class EpisodeFrame(BaseModel):
    """A single timestep frame within an episode."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    timestamp: float
    joint_positions: list[float]
    gripper_state: float = Field(ge=0.0, le=1.0)
    image_path: str = ""
    action: list[float]
    reward: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Episode(BaseModel):
    """Complete record of a single episode."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    episode_id: UUID = Field(default_factory=uuid4)
    task_name: str = "default_task"
    robot_id: str = "default"
    start_time: datetime.datetime = Field(default_factory=datetime.datetime.now)
    end_time: datetime.datetime | None = None
    frames: list[EpisodeFrame] = Field(default_factory=list)
    outcome: Outcome = Outcome.UNKNOWN
    policy_checkpoint: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def total_reward(self) -> float:
        return sum(f.reward for f in self.frames if f.reward is not None)

    @property
    def duration(self) -> float | None:
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    @property
    def avg_action_magnitude(self) -> float:
        if not self.frames:
            return 0.0
        magnitudes = [float(np.linalg.norm(f.action)) for f in self.frames]
        return float(np.mean(magnitudes))


class DeploymentSession(BaseModel):
    """A deployment session containing multiple episodes."""

    session_id: UUID = Field(default_factory=uuid4)
    episodes: list[Episode] = Field(default_factory=list)
    environment_description: str = ""
    policy_version: str = ""


# ---------------------------------------------------------------------------
# DEPRECATED: Legacy models retained for backward compatibility.
# Use Episode, EpisodeFrame, DeploymentSession instead.
# ---------------------------------------------------------------------------


class StepRecord(BaseModel):
    """A single timestep within an episode (deprecated)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    step_index: int
    timestamp: float
    observation: dict[str, Any]
    action: list[float] | Any
    reward: float = 0.0
    done: bool = False
    info: dict[str, Any] = Field(default_factory=dict)
    images: dict[str, Any] = Field(default_factory=dict)


class EpisodeRecord(BaseModel):
    """Complete record of a single episode (deprecated)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    episode_id: int
    task: str
    steps: list[StepRecord] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    start_time: datetime.datetime = Field(default_factory=datetime.datetime.now)
    end_time: datetime.datetime | None = None
    total_reward: float = 0.0
    success: bool | None = None
    num_steps: int = 0
