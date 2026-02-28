"""Phase-2 heuristic failure detectors for the new Episode model.

Each detector is stateless: it receives an ``Episode`` and returns zero
or more ``FailureDetection`` instances.  The ``DetectorPipeline``
orchestrates all enabled detectors and produces an aggregate report.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np
import yaml  # type: ignore[import-untyped]

from orbit.logger.schemas import Episode

if TYPE_CHECKING:
    from orbit.detector.legacy import DetectionResult

logger = logging.getLogger(__name__)


# ── Result dataclass ───────────────────────────────────────────────


@dataclass
class FailureDetection:
    """A single failure signal emitted by one detector."""

    detector_name: str
    confidence: float  # 0.0 to 1.0
    frame_idx: int | None  # frame where the failure is localized, or None
    description: str


# ── Base class ─────────────────────────────────────────────────────


class BaseDetector(abc.ABC):
    """Abstract base for all heuristic detectors.

    Subclasses must implement ``detect`` and may override ``explain``.
    """

    @abc.abstractmethod
    def detect(self, episode: Episode) -> list[FailureDetection]:
        """Analyze an episode and return zero or more detections."""
        ...

    def explain(self, detections: list[FailureDetection]) -> str:
        """Return a human-readable explanation of the detections."""
        if not detections:
            return f"{self.__class__.__name__}: No failures detected."
        lines = [f"{self.__class__.__name__} detected {len(detections)} issue(s):"]
        for d in detections:
            frame_info = f" at frame {d.frame_idx}" if d.frame_idx is not None else ""
            lines.append(f"  - [{d.confidence:.0%}]{frame_info}: {d.description}")
        return "\n".join(lines)


# ── GripperDropDetector ────────────────────────────────────────────


@dataclass
class GripperDropConfig:
    """Config for GripperDropDetector."""

    closed_threshold: float = 0.8  # gripper_state > this means physically gripping
    open_threshold: float = 0.2  # gripper_state < this means physically released
    min_closed_frames: int = 10  # must be gripping for N+ frames before a drop counts
    release_zone_pct: float = 0.15  # last 15% of episode = intentional release
    reclose_window: int = 5  # frames to wait for re-close (sensor noise filter)
    confidence: float = 0.85


class GripperDropDetector(BaseDetector):
    """Detects when the gripper drops an object it was holding.

    Algorithm:
      1. Scan the gripper_state timeseries.
      2. Track runs of consecutive frames where gripper_state > closed_threshold
         (physically gripping).
      3. When a transition to open (gripper_state < open_threshold) occurs
         after a holding region of at least *min_closed_frames*:
           a. Skip if in the last *release_zone_pct* of the episode (intentional release).
           b. Skip if the gripper re-closes within *reclose_window* frames (sensor noise).
           c. Otherwise flag a drop.
    """

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
                    # Check 1: not in the release zone (last 15% of episode)
                    if i >= release_zone_start:
                        closed_run = 0
                        continue

                    # Check 2: gripper doesn't re-close within reclose_window
                    reclose = False
                    for j in range(
                        i + 1,
                        min(i + 1 + self.config.reclose_window, n_frames),
                    ):
                        if episode.frames[j].gripper_state > self.config.open_threshold:
                            reclose = True
                            break
                    if reclose:
                        closed_run = 0
                        continue

                    detections.append(
                        FailureDetection(
                            detector_name="GripperDropDetector",
                            confidence=self.config.confidence,
                            frame_idx=i,
                            description=(
                                f"Gripper opened at frame {i} after being closed "
                                f"for {closed_run} consecutive frames"
                            ),
                        )
                    )
                closed_run = 0
            else:
                # In the dead-zone between thresholds — don't reset or increment
                pass

        return detections


# ── StallDetector ──────────────────────────────────────────────────


@dataclass
class StallConfig:
    """Config for StallDetector."""

    velocity_threshold: float = 0.001  # L2-norm below this = stalled
    min_stall_frames: int = 10  # must be stalled for N+ consecutive frames
    confidence: float = 0.80


class StallDetector(BaseDetector):
    """Detects when the robot stalls (joint velocity near zero).

    Computes joint velocities as the finite difference of
    joint_positions between consecutive frames, divided by the
    timestamp delta.
    """

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
            dt = max(dt, 1e-6)  # guard against zero/negative dt

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
                    detections.append(
                        FailureDetection(
                            detector_name="StallDetector",
                            confidence=self.config.confidence,
                            frame_idx=stall_start,
                            description=(
                                f"Robot stalled for {stall_run} frames "
                                f"starting at frame {stall_start}"
                            ),
                        )
                    )
                stall_run = 0
                stall_start = None

        # Check if stall extends to end of episode
        if stall_run >= self.config.min_stall_frames and stall_start is not None:
            detections.append(
                FailureDetection(
                    detector_name="StallDetector",
                    confidence=self.config.confidence,
                    frame_idx=stall_start,
                    description=(
                        f"Robot stalled for {stall_run} frames starting at frame {stall_start}"
                    ),
                )
            )

        return detections


# ── OutOfBoundsDetector ────────────────────────────────────────────


@dataclass
class OutOfBoundsConfig:
    """Config for OutOfBoundsDetector."""

    joint_limits_lower: list[float] = field(default_factory=list)
    joint_limits_upper: list[float] = field(default_factory=list)
    confidence: float = 0.95


class OutOfBoundsDetector(BaseDetector):
    """Detects when joint positions exceed safe workspace limits.

    Skips detection if limits are not configured (empty lists).
    """

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
                    detections.append(
                        FailureDetection(
                            detector_name="OutOfBoundsDetector",
                            confidence=self.config.confidence,
                            frame_idx=i,
                            description=(
                                f"Joint {j} position {pos:.4f} below lower limit "
                                f"{lower[j]:.4f} at frame {i}"
                            ),
                        )
                    )
                    return detections  # first violation is enough
                if upper and j < len(upper) and pos > upper[j]:
                    detections.append(
                        FailureDetection(
                            detector_name="OutOfBoundsDetector",
                            confidence=self.config.confidence,
                            frame_idx=i,
                            description=(
                                f"Joint {j} position {pos:.4f} above upper limit "
                                f"{upper[j]:.4f} at frame {i}"
                            ),
                        )
                    )
                    return detections  # first violation is enough

        return detections


# ── TimeoutDetector ────────────────────────────────────────────────


@dataclass
class TimeoutConfig:
    """Config for TimeoutDetector."""

    max_duration_seconds: float = 60.0
    max_frames: int = 1000
    confidence: float = 0.70


class TimeoutDetector(BaseDetector):
    """Detects episodes exceeding a time or frame-count limit."""

    def __init__(self, config: TimeoutConfig | None = None) -> None:
        self.config = config or TimeoutConfig()

    def detect(self, episode: Episode) -> list[FailureDetection]:
        detections: list[FailureDetection] = []

        duration = episode.duration
        if duration is not None and duration > self.config.max_duration_seconds:
            detections.append(
                FailureDetection(
                    detector_name="TimeoutDetector",
                    confidence=self.config.confidence,
                    frame_idx=None,
                    description=(
                        f"Episode duration {duration:.1f}s exceeds limit of "
                        f"{self.config.max_duration_seconds:.1f}s"
                    ),
                )
            )

        if episode.num_frames > self.config.max_frames:
            detections.append(
                FailureDetection(
                    detector_name="TimeoutDetector",
                    confidence=self.config.confidence,
                    frame_idx=None,
                    description=(
                        f"Episode has {episode.num_frames} frames, exceeding limit of "
                        f"{self.config.max_frames}"
                    ),
                )
            )

        return detections


# ── RewardThresholdDetector ────────────────────────────────────────


@dataclass
class RewardThresholdConfig:
    """Config for RewardThresholdDetector."""

    min_total_reward: float = 0.0
    min_avg_reward_per_frame: float | None = None
    confidence: float = 0.75


class RewardThresholdDetector(BaseDetector):
    """Detects episodes with reward below configured threshold."""

    def __init__(self, config: RewardThresholdConfig | None = None) -> None:
        self.config = config or RewardThresholdConfig()

    def detect(self, episode: Episode) -> list[FailureDetection]:
        detections: list[FailureDetection] = []

        total = episode.total_reward
        if total < self.config.min_total_reward:
            detections.append(
                FailureDetection(
                    detector_name="RewardThresholdDetector",
                    confidence=self.config.confidence,
                    frame_idx=None,
                    description=(
                        f"Total reward {total:.2f} below threshold "
                        f"{self.config.min_total_reward:.2f}"
                    ),
                )
            )

        if self.config.min_avg_reward_per_frame is not None and episode.num_frames > 0:
            avg = total / episode.num_frames
            if avg < self.config.min_avg_reward_per_frame:
                detections.append(
                    FailureDetection(
                        detector_name="RewardThresholdDetector",
                        confidence=self.config.confidence,
                        frame_idx=None,
                        description=(
                            f"Average reward per frame {avg:.4f} below threshold "
                            f"{self.config.min_avg_reward_per_frame:.4f}"
                        ),
                    )
                )

        return detections


# ── Pipeline ───────────────────────────────────────────────────────


@dataclass
class PipelineResult:
    """Aggregate result from running all detectors on one episode."""

    episode_id: UUID
    detections: list[FailureDetection] = field(default_factory=list)
    failure_probability: float = 0.0  # 0.0 to 1.0
    detector_summaries: dict[str, str] = field(default_factory=dict)

    @property
    def is_failure(self) -> bool:
        return self.failure_probability > 0.0

    def to_legacy_result(self) -> DetectionResult:
        """Convert to a legacy ``DetectionResult`` for the Prescriber."""
        from orbit.detector.legacy import DetectionResult

        return DetectionResult(
            episode_id=hash(str(self.episode_id)) % (2**31),
            is_failure=self.is_failure,
            failure_reasons=[d.description for d in self.detections],
            confidence=self.failure_probability,
            failure_step=self.detections[0].frame_idx if self.detections else None,
        )


class DetectorPipeline:
    """Runs a set of detectors and aggregates results.

    ``failure_probability`` is the maximum individual detection confidence
    across all detections, clamped to [0, 1].  This is deliberately
    simple — a max-confidence model — because the detectors are
    heuristic and their confidences are not calibrated.

    When created with no arguments, includes all five detectors with
    sensible defaults.  Pass an explicit list (even ``[]``) to override.
    """

    def __init__(self, detectors: list[BaseDetector] | None = None) -> None:
        if detectors is None:
            self.detectors: list[BaseDetector] = [
                GripperDropDetector(),
                StallDetector(),
                OutOfBoundsDetector(),
                TimeoutDetector(),
                RewardThresholdDetector(),
            ]
        else:
            if len(detectors) == 0:
                import warnings

                warnings.warn(
                    "DetectorPipeline created with no detectors. Nothing will be detected.",
                    stacklevel=2,
                )
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
            episode_id=episode.episode_id,
            detections=all_detections,
            failure_probability=min(failure_prob, 1.0),
            detector_summaries=summaries,
        )

    def run_batch(self, episodes: list[Episode]) -> list[PipelineResult]:
        return [self.run(ep) for ep in episodes]


# ── YAML config loader ─────────────────────────────────────────────

_DETECTOR_REGISTRY: dict[str, tuple[type[BaseDetector], type]] = {
    "gripper_drop": (GripperDropDetector, GripperDropConfig),
    "stall": (StallDetector, StallConfig),
    "out_of_bounds": (OutOfBoundsDetector, OutOfBoundsConfig),
    "timeout": (TimeoutDetector, TimeoutConfig),
    "reward_threshold": (RewardThresholdDetector, RewardThresholdConfig),
}


def load_pipeline_from_yaml(path: str | Path) -> DetectorPipeline:
    """Build a DetectorPipeline from a YAML configuration file.

    Expected format::

        detectors:
          - name: gripper_drop
            closed_threshold: 0.3
            min_closed_frames: 5
          - name: stall
            velocity_threshold: 0.001
    """
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict) or "detectors" not in raw:
        raise ValueError(f"YAML config must have a top-level 'detectors' key: {path}")

    detectors: list[BaseDetector] = []
    for entry in raw["detectors"]:
        entry = dict(entry)  # defensive copy
        name = entry.pop("name")
        if name not in _DETECTOR_REGISTRY:
            raise ValueError(f"Unknown detector '{name}'. Available: {list(_DETECTOR_REGISTRY)}")
        detector_cls, config_cls = _DETECTOR_REGISTRY[name]
        cfg = config_cls(**entry)
        detectors.append(detector_cls(cfg))  # type: ignore[call-arg]

    return DetectorPipeline(detectors=detectors)
