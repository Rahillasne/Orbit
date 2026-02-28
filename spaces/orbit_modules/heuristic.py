"""Phase-2 heuristic failure detectors for the new Episode model."""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np
import yaml

from orbit_modules.schemas import Episode

if TYPE_CHECKING:
    from orbit_modules.legacy import DetectionResult

logger = logging.getLogger(__name__)


@dataclass
class FailureDetection:
    """A single failure signal emitted by one detector."""

    detector_name: str
    confidence: float
    frame_idx: int | None
    description: str


class BaseDetector(abc.ABC):
    @abc.abstractmethod
    def detect(self, episode: Episode) -> list[FailureDetection]: ...

    def explain(self, detections: list[FailureDetection]) -> str:
        if not detections:
            return f"{self.__class__.__name__}: No failures detected."
        lines = [f"{self.__class__.__name__} detected {len(detections)} issue(s):"]
        for d in detections:
            frame_info = f" at frame {d.frame_idx}" if d.frame_idx is not None else ""
            lines.append(f"  - [{d.confidence:.0%}]{frame_info}: {d.description}")
        return "\n".join(lines)


@dataclass
class GripperDropConfig:
    closed_threshold: float = 0.8
    open_threshold: float = 0.2
    min_closed_frames: int = 10
    release_zone_pct: float = 0.15
    reclose_window: int = 5
    confidence: float = 0.85


class GripperDropDetector(BaseDetector):
    def __init__(self, config: GripperDropConfig | None = None) -> None:
        self.config = config or GripperDropConfig()

    def detect(self, episode: Episode) -> list[FailureDetection]:
        detections: list[FailureDetection] = []
        if not episode.frames:
            return detections
        n_frames = len(episode.frames)
        release_zone_start = int(n_frames * (1 - self.config.release_zone_pct))
        closed_run = 0
        for i, frame in enumerate(episode.frames):
            if frame.gripper_state > self.config.closed_threshold:
                closed_run += 1
            elif frame.gripper_state < self.config.open_threshold:
                if closed_run >= self.config.min_closed_frames:
                    if i >= release_zone_start:
                        closed_run = 0
                        continue
                    reclose = False
                    for j in range(i + 1, min(i + 1 + self.config.reclose_window, n_frames)):
                        if episode.frames[j].gripper_state > self.config.open_threshold:
                            reclose = True
                            break
                    if reclose:
                        closed_run = 0
                        continue
                    detections.append(FailureDetection(
                        detector_name="GripperDropDetector", confidence=self.config.confidence,
                        frame_idx=i, description=f"Gripper opened at frame {i} after being closed for {closed_run} consecutive frames",
                    ))
                closed_run = 0
        return detections


@dataclass
class StallConfig:
    velocity_threshold: float = 0.001
    min_stall_frames: int = 10
    confidence: float = 0.80


class StallDetector(BaseDetector):
    def __init__(self, config: StallConfig | None = None) -> None:
        self.config = config or StallConfig()

    def detect(self, episode: Episode) -> list[FailureDetection]:
        detections: list[FailureDetection] = []
        frames = episode.frames
        if len(frames) < 2:
            return detections
        stall_run = 0
        stall_start: int | None = None
        for i in range(1, len(frames)):
            dt = frames[i].timestamp - frames[i - 1].timestamp
            dt = max(dt, 1e-6)
            jp_curr = np.array(frames[i].joint_positions)
            jp_prev = np.array(frames[i - 1].joint_positions)
            velocity = (jp_curr - jp_prev) / dt
            speed = float(np.linalg.norm(velocity))
            if speed < self.config.velocity_threshold:
                if stall_run == 0:
                    stall_start = i
                stall_run += 1
            else:
                if stall_run >= self.config.min_stall_frames and stall_start is not None:
                    detections.append(FailureDetection(
                        detector_name="StallDetector", confidence=self.config.confidence,
                        frame_idx=stall_start, description=f"Robot stalled for {stall_run} frames starting at frame {stall_start}",
                    ))
                stall_run = 0
                stall_start = None
        if stall_run >= self.config.min_stall_frames and stall_start is not None:
            detections.append(FailureDetection(
                detector_name="StallDetector", confidence=self.config.confidence,
                frame_idx=stall_start, description=f"Robot stalled for {stall_run} frames starting at frame {stall_start}",
            ))
        return detections


@dataclass
class OutOfBoundsConfig:
    joint_limits_lower: list[float] = field(default_factory=list)
    joint_limits_upper: list[float] = field(default_factory=list)
    confidence: float = 0.95


class OutOfBoundsDetector(BaseDetector):
    def __init__(self, config: OutOfBoundsConfig | None = None) -> None:
        self.config = config or OutOfBoundsConfig()

    def detect(self, episode: Episode) -> list[FailureDetection]:
        detections: list[FailureDetection] = []
        lower = self.config.joint_limits_lower
        upper = self.config.joint_limits_upper
        if not lower and not upper:
            return detections
        for i, frame in enumerate(episode.frames):
            jp = frame.joint_positions
            for j, pos in enumerate(jp):
                if lower and j < len(lower) and pos < lower[j]:
                    detections.append(FailureDetection(
                        detector_name="OutOfBoundsDetector", confidence=self.config.confidence,
                        frame_idx=i, description=f"Joint {j} position {pos:.4f} below lower limit {lower[j]:.4f} at frame {i}",
                    ))
                    return detections
                if upper and j < len(upper) and pos > upper[j]:
                    detections.append(FailureDetection(
                        detector_name="OutOfBoundsDetector", confidence=self.config.confidence,
                        frame_idx=i, description=f"Joint {j} position {pos:.4f} above upper limit {upper[j]:.4f} at frame {i}",
                    ))
                    return detections
        return detections


@dataclass
class TimeoutConfig:
    max_duration_seconds: float = 60.0
    max_frames: int = 1000
    confidence: float = 0.70


class TimeoutDetector(BaseDetector):
    def __init__(self, config: TimeoutConfig | None = None) -> None:
        self.config = config or TimeoutConfig()

    def detect(self, episode: Episode) -> list[FailureDetection]:
        detections: list[FailureDetection] = []
        duration = episode.duration
        if duration is not None and duration > self.config.max_duration_seconds:
            detections.append(FailureDetection(
                detector_name="TimeoutDetector", confidence=self.config.confidence,
                frame_idx=None, description=f"Episode duration {duration:.1f}s exceeds limit of {self.config.max_duration_seconds:.1f}s",
            ))
        if episode.num_frames > self.config.max_frames:
            detections.append(FailureDetection(
                detector_name="TimeoutDetector", confidence=self.config.confidence,
                frame_idx=None, description=f"Episode has {episode.num_frames} frames, exceeding limit of {self.config.max_frames}",
            ))
        return detections


@dataclass
class RewardThresholdConfig:
    min_total_reward: float = 0.0
    min_avg_reward_per_frame: float | None = None
    confidence: float = 0.75


class RewardThresholdDetector(BaseDetector):
    def __init__(self, config: RewardThresholdConfig | None = None) -> None:
        self.config = config or RewardThresholdConfig()

    def detect(self, episode: Episode) -> list[FailureDetection]:
        detections: list[FailureDetection] = []
        total = episode.total_reward
        if total < self.config.min_total_reward:
            detections.append(FailureDetection(
                detector_name="RewardThresholdDetector", confidence=self.config.confidence,
                frame_idx=None, description=f"Total reward {total:.2f} below threshold {self.config.min_total_reward:.2f}",
            ))
        if self.config.min_avg_reward_per_frame is not None and episode.num_frames > 0:
            avg = total / episode.num_frames
            if avg < self.config.min_avg_reward_per_frame:
                detections.append(FailureDetection(
                    detector_name="RewardThresholdDetector", confidence=self.config.confidence,
                    frame_idx=None, description=f"Average reward per frame {avg:.4f} below threshold {self.config.min_avg_reward_per_frame:.4f}",
                ))
        return detections


@dataclass
class PipelineResult:
    episode_id: UUID
    detections: list[FailureDetection] = field(default_factory=list)
    failure_probability: float = 0.0
    detector_summaries: dict[str, str] = field(default_factory=dict)

    @property
    def is_failure(self) -> bool:
        return self.failure_probability > 0.0

    def to_legacy_result(self) -> DetectionResult:
        from orbit_modules.legacy import DetectionResult
        return DetectionResult(
            episode_id=hash(str(self.episode_id)) % (2**31),
            is_failure=self.is_failure,
            failure_reasons=[d.description for d in self.detections],
            confidence=self.failure_probability,
            failure_step=self.detections[0].frame_idx if self.detections else None,
        )


class DetectorPipeline:
    def __init__(self, detectors: list[BaseDetector] | None = None) -> None:
        if detectors is None:
            self.detectors: list[BaseDetector] = [
                GripperDropDetector(), StallDetector(), OutOfBoundsDetector(),
                TimeoutDetector(), RewardThresholdDetector(),
            ]
        else:
            self.detectors = detectors

    def add_detector(self, detector: BaseDetector) -> None:
        self.detectors.append(detector)

    def run(self, episode: Episode) -> PipelineResult:
        all_detections: list[FailureDetection] = []
        summaries: dict[str, str] = {}
        for detector in self.detectors:
            detections = detector.detect(episode)
            all_detections.extend(detections)
            summaries[detector.__class__.__name__] = detector.explain(detections)
        failure_prob = max(d.confidence for d in all_detections) if all_detections else 0.0
        return PipelineResult(
            episode_id=episode.episode_id, detections=all_detections,
            failure_probability=min(failure_prob, 1.0), detector_summaries=summaries,
        )

    def run_batch(self, episodes: list[Episode]) -> list[PipelineResult]:
        return [self.run(ep) for ep in episodes]


_DETECTOR_REGISTRY: dict[str, tuple[type[BaseDetector], type]] = {
    "gripper_drop": (GripperDropDetector, GripperDropConfig),
    "stall": (StallDetector, StallConfig),
    "out_of_bounds": (OutOfBoundsDetector, OutOfBoundsConfig),
    "timeout": (TimeoutDetector, TimeoutConfig),
    "reward_threshold": (RewardThresholdDetector, RewardThresholdConfig),
}


def load_pipeline_from_yaml(path: str | Path) -> DetectorPipeline:
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict) or "detectors" not in raw:
        raise ValueError(f"YAML config must have a top-level 'detectors' key: {path}")
    detectors: list[BaseDetector] = []
    for entry in raw["detectors"]:
        entry = dict(entry)
        name = entry.pop("name")
        if name not in _DETECTOR_REGISTRY:
            raise ValueError(f"Unknown detector '{name}'. Available: {list(_DETECTOR_REGISTRY)}")
        detector_cls, config_cls = _DETECTOR_REGISTRY[name]
        cfg = config_cls(**entry)
        detectors.append(detector_cls(cfg))
    return DetectorPipeline(detectors=detectors)
