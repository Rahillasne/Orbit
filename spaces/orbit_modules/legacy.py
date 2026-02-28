"""Legacy heuristic-based failure detection (deprecated).

Retained for backward compatibility with the prescriber module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from orbit_modules.schemas import EpisodeRecord


@dataclass
class DetectionResult:
    """Result of failure detection on a single episode."""

    episode_id: int
    is_failure: bool
    failure_reasons: list[str] = field(default_factory=list)
    confidence: float = 0.0
    failure_step: int | None = None
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class DetectorConfig:
    """Configuration for heuristic failure detection."""

    reward_threshold: float = -1.0
    min_reward_per_step: float = -0.5
    action_variance_threshold: float = 0.01
    consecutive_failure_steps: int = 10
    max_episode_length: int = 1000
    min_episode_length: int = 5


class HeuristicDetector:
    """Detects failure episodes using configurable heuristic rules."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()

    def detect(self, episode: EpisodeRecord) -> DetectionResult:
        reasons: list[str] = []
        metrics: dict[str, float] = {}
        earliest_failure_step: int | None = None

        reason = self._check_reward_threshold(episode, metrics)
        if reason:
            reasons.append(reason)

        reason = self._check_action_variance(episode, metrics)
        if reason:
            reasons.append(reason)

        reason, step_idx = self._check_consecutive_failures(episode, metrics)
        if reason:
            reasons.append(reason)
            if step_idx is not None:
                earliest_failure_step = step_idx

        reason = self._check_episode_length(episode, metrics)
        if reason:
            reasons.append(reason)

        reason = self._check_success_flag(episode)
        if reason:
            reasons.append(reason)

        total_checks = 5
        confidence = len(reasons) / total_checks if reasons else 0.0

        return DetectionResult(
            episode_id=episode.episode_id,
            is_failure=len(reasons) > 0,
            failure_reasons=reasons,
            confidence=confidence,
            failure_step=earliest_failure_step,
            metrics=metrics,
        )

    def detect_batch(self, episodes: list[EpisodeRecord]) -> list[DetectionResult]:
        return [self.detect(ep) for ep in episodes]

    def _check_reward_threshold(self, episode: EpisodeRecord, metrics: dict[str, float]) -> str | None:
        metrics["total_reward"] = episode.total_reward
        if episode.total_reward < self.config.reward_threshold:
            return f"Total reward {episode.total_reward:.2f} below threshold {self.config.reward_threshold:.2f}"
        return None

    def _check_action_variance(self, episode: EpisodeRecord, metrics: dict[str, float]) -> str | None:
        if not episode.steps:
            return None
        actions = np.array([s.action for s in episode.steps], dtype=np.float32)
        mean_variance = float(np.mean(np.var(actions, axis=0)))
        metrics["action_variance"] = mean_variance
        if mean_variance < self.config.action_variance_threshold:
            return f"Action variance {mean_variance:.6f} below threshold {self.config.action_variance_threshold:.6f} (robot may be stuck)"
        return None

    def _check_consecutive_failures(self, episode: EpisodeRecord, metrics: dict[str, float]) -> tuple[str | None, int | None]:
        if not episode.steps:
            return None, None
        rewards = [s.reward for s in episode.steps]
        max_run = 0
        current_run = 0
        run_start: int | None = None
        best_start: int | None = None
        for i, r in enumerate(rewards):
            if r < self.config.min_reward_per_step:
                if current_run == 0:
                    run_start = i
                current_run += 1
                if current_run > max_run:
                    max_run = current_run
                    best_start = run_start
            else:
                current_run = 0
        metrics["max_consecutive_low_reward"] = float(max_run)
        if max_run >= self.config.consecutive_failure_steps:
            return (f"{max_run} consecutive steps with reward below {self.config.min_reward_per_step:.2f}", best_start)
        return None, None

    def _check_episode_length(self, episode: EpisodeRecord, metrics: dict[str, float]) -> str | None:
        n = episode.num_steps or len(episode.steps)
        metrics["episode_length"] = float(n)
        if n < self.config.min_episode_length:
            return f"Episode too short ({n} steps, minimum {self.config.min_episode_length})"
        if n > self.config.max_episode_length:
            return f"Episode too long ({n} steps, maximum {self.config.max_episode_length})"
        return None

    def _check_success_flag(self, episode: EpisodeRecord) -> str | None:
        if episode.success is False:
            return "Episode explicitly marked as failure (success=False)"
        return None
