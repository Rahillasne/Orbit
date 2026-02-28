"""Corrective prescription generator based on failure analysis."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from orbit.detector.legacy import DetectionResult

if TYPE_CHECKING:
    from orbit.analyzer.embedding_gap import GapAnalysisResult
    from orbit.logger.schemas import EpisodeRecord


class PrescriptionType(str, Enum):
    """Categories of corrective prescriptions."""

    DATA_AUGMENTATION = "data_augmentation"
    REWARD_SHAPING = "reward_shaping"
    ACTION_SPACE = "action_space"
    EXPLORATION = "exploration"
    TASK_DECOMPOSITION = "task_decomposition"
    ENVIRONMENT = "environment"
    POLICY_ARCHITECTURE = "policy_architecture"


@dataclass
class Prescription:
    """A single corrective prescription."""

    prescription_type: PrescriptionType
    title: str
    description: str
    priority: int = 1
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    suggested_params: dict = field(default_factory=dict)


@dataclass
class PrescriptionReport:
    """Complete prescription report for a set of failure episodes."""

    prescriptions: list[Prescription]
    summary: str
    num_failures_analyzed: int
    num_success_reference: int = 0


class Prescriber:
    """Generates corrective prescriptions from detection and analysis results.

    Maps failure patterns to actionable recommendations:

    - Low action variance -> increase exploration noise
    - High embedding gap -> collect more diverse demonstrations
    - Consistent early failures -> simplify reward / decompose task
    - Reward threshold failures -> adjust reward shaping
    """

    def prescribe(
        self,
        detection_results: list[DetectionResult],
        gap_analysis: GapAnalysisResult | None = None,
        episodes: list[EpisodeRecord] | None = None,
    ) -> PrescriptionReport:
        """Generate a prescription report from detection and analysis results."""
        patterns = self._analyze_failure_patterns(detection_results)
        prescriptions = self._generate_prescriptions(patterns, gap_analysis)
        prescriptions = self._rank_prescriptions(prescriptions)

        summary = self._build_summary(detection_results, prescriptions)

        return PrescriptionReport(
            prescriptions=prescriptions,
            summary=summary,
            num_failures_analyzed=len(detection_results),
        )

    def _analyze_failure_patterns(self, results: list[DetectionResult]) -> dict[str, float]:
        """Count and normalize failure reason frequencies."""
        if not results:
            return {}

        reason_counter: Counter[str] = Counter()
        for result in results:
            for reason in result.failure_reasons:
                # Normalize to pattern category
                if "variance" in reason.lower():
                    reason_counter["low_action_variance"] += 1
                elif "reward" in reason.lower() and "consecutive" in reason.lower():
                    reason_counter["consecutive_low_reward"] += 1
                elif "reward" in reason.lower():
                    reason_counter["low_total_reward"] += 1
                elif "short" in reason.lower():
                    reason_counter["short_episode"] += 1
                elif "long" in reason.lower():
                    reason_counter["long_episode"] += 1
                elif "success=false" in reason.lower() or "failure" in reason.lower():
                    reason_counter["explicit_failure"] += 1

        total = len(results)
        return {k: v / total for k, v in reason_counter.items()}

    def _generate_prescriptions(
        self,
        patterns: dict[str, float],
        gap: GapAnalysisResult | None,
    ) -> list[Prescription]:
        """Map patterns to prescriptions via a rules engine."""
        prescriptions: list[Prescription] = []

        if patterns.get("low_action_variance", 0) > 0.3:
            prescriptions.append(
                Prescription(
                    prescription_type=PrescriptionType.EXPLORATION,
                    title="Increase Exploration Noise",
                    description=(
                        "A significant portion of failures show very low action variance, "
                        "suggesting the robot is getting stuck. Increase exploration noise "
                        "(epsilon for discrete actions, sigma for continuous) to encourage "
                        "the policy to explore more diverse actions."
                    ),
                    confidence=patterns["low_action_variance"],
                    evidence=[
                        f"{patterns['low_action_variance']:.0%} of failures had low action variance"
                    ],
                    suggested_params={
                        "exploration_noise_multiplier": 2.0,
                        "add_action_perturbation": True,
                    },
                )
            )

        if patterns.get("consecutive_low_reward", 0) > 0.5:
            prescriptions.append(
                Prescription(
                    prescription_type=PrescriptionType.TASK_DECOMPOSITION,
                    title="Decompose Task with Intermediate Rewards",
                    description=(
                        "Many failures show long stretches of negative reward, indicating "
                        "the agent is unable to make progress. Break the task into subtasks "
                        "with intermediate reward signals to provide a denser learning signal."
                    ),
                    confidence=patterns["consecutive_low_reward"],
                    evidence=[
                        f"{patterns['consecutive_low_reward']:.0%} of failures "
                        f"had consecutive low-reward steps"
                    ],
                    suggested_params={"add_intermediate_rewards": True},
                )
            )

        if patterns.get("low_total_reward", 0) > 0.3:
            prescriptions.append(
                Prescription(
                    prescription_type=PrescriptionType.REWARD_SHAPING,
                    title="Adjust Reward Shaping",
                    description=(
                        "Failures frequently have very low total reward. Consider adding "
                        "dense reward signals (e.g., distance-to-goal shaping) or increasing "
                        "the reward scale to make the learning signal stronger."
                    ),
                    confidence=patterns["low_total_reward"],
                    evidence=[
                        f"{patterns['low_total_reward']:.0%} of failures had low total reward"
                    ],
                    suggested_params={"reward_scale_multiplier": 5.0},
                )
            )

        if patterns.get("short_episode", 0) > 0.3:
            prescriptions.append(
                Prescription(
                    prescription_type=PrescriptionType.ENVIRONMENT,
                    title="Check Environment Reset and Initial States",
                    description=(
                        "Many episodes are terminating very early, which may indicate "
                        "issues with the initial state distribution or environment reset. "
                        "Verify that the environment starts in a reachable state."
                    ),
                    confidence=patterns["short_episode"],
                    evidence=[f"{patterns['short_episode']:.0%} of failures were abnormally short"],
                )
            )

        if patterns.get("long_episode", 0) > 0.3:
            prescriptions.append(
                Prescription(
                    prescription_type=PrescriptionType.POLICY_ARCHITECTURE,
                    title="Add Episode Length Penalty or Timeout",
                    description=(
                        "Many episodes ran for an excessive number of steps, suggesting "
                        "the agent is stuck in a loop. Add a step penalty to the reward "
                        "function or reduce the maximum episode length."
                    ),
                    confidence=patterns["long_episode"],
                    evidence=[f"{patterns['long_episode']:.0%} of failures were abnormally long"],
                )
            )

        # Gap-based prescriptions
        if gap is not None and gap.mean_gap_distance > 1.0:
            nearest = [f"{d:.2f}" for d in gap.nearest_success_distances[:5]]
            prescriptions.append(
                Prescription(
                    prescription_type=PrescriptionType.DATA_AUGMENTATION,
                    title="Collect More Diverse Demonstrations",
                    description=(
                        "The embedding gap between successful and failed "
                        "episodes is large "
                        f"(distance: {gap.mean_gap_distance:.2f}), indicating "
                        "failures occur in underrepresented regions of the "
                        "observation space. Collect more demonstrations "
                        "covering these failure regions, or apply image "
                        "augmentation to increase data diversity."
                    ),
                    confidence=min(gap.mean_gap_distance / 3.0, 1.0),
                    evidence=[
                        f"Mean embedding gap distance: {gap.mean_gap_distance:.2f}",
                        f"Nearest success distances: {nearest}",
                    ],
                )
            )

        # Default prescription if no patterns match
        if not prescriptions:
            prescriptions.append(
                Prescription(
                    prescription_type=PrescriptionType.DATA_AUGMENTATION,
                    title="General Training Improvement",
                    description=(
                        "No strong failure patterns were detected. Consider general "
                        "improvements: more training data, longer training, learning "
                        "rate tuning, or architecture changes."
                    ),
                    priority=3,
                    confidence=0.1,
                )
            )

        return prescriptions

    def _rank_prescriptions(self, prescriptions: list[Prescription]) -> list[Prescription]:
        """Rank prescriptions by confidence (highest first)."""
        for i, p in enumerate(sorted(prescriptions, key=lambda x: x.confidence, reverse=True)):
            p.priority = i + 1
        return sorted(prescriptions, key=lambda x: x.priority)

    def _build_summary(
        self,
        results: list[DetectionResult],
        prescriptions: list[Prescription],
    ) -> str:
        n_failures = sum(1 for r in results if r.is_failure)
        top = prescriptions[0] if prescriptions else None
        summary = f"Analyzed {len(results)} episodes, {n_failures} identified as failures."
        if top:
            summary += f" Top recommendation: {top.title}."
        return summary
