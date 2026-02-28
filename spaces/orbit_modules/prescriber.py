"""Corrective prescription generator based on failure analysis."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from orbit_modules.legacy import DetectionResult

if TYPE_CHECKING:
    from orbit_modules.schemas import EpisodeRecord


class PrescriptionType(str, Enum):
    DATA_AUGMENTATION = "data_augmentation"
    REWARD_SHAPING = "reward_shaping"
    ACTION_SPACE = "action_space"
    EXPLORATION = "exploration"
    TASK_DECOMPOSITION = "task_decomposition"
    ENVIRONMENT = "environment"
    POLICY_ARCHITECTURE = "policy_architecture"


@dataclass
class Prescription:
    prescription_type: PrescriptionType
    title: str
    description: str
    priority: int = 1
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    suggested_params: dict = field(default_factory=dict)


@dataclass
class PrescriptionReport:
    prescriptions: list[Prescription]
    summary: str
    num_failures_analyzed: int
    num_success_reference: int = 0


class Prescriber:
    def prescribe(
        self,
        detection_results: list[DetectionResult],
        gap_analysis=None,
        episodes=None,
    ) -> PrescriptionReport:
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
        if not results:
            return {}
        reason_counter: Counter[str] = Counter()
        for result in results:
            for reason in result.failure_reasons:
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

    def _generate_prescriptions(self, patterns: dict[str, float], gap=None) -> list[Prescription]:
        prescriptions: list[Prescription] = []

        if patterns.get("low_action_variance", 0) > 0.3:
            prescriptions.append(Prescription(
                prescription_type=PrescriptionType.EXPLORATION,
                title="Increase Exploration Noise",
                description="A significant portion of failures show very low action variance, suggesting the robot is getting stuck. Increase exploration noise to encourage the policy to explore more diverse actions.",
                confidence=patterns["low_action_variance"],
                evidence=[f"{patterns['low_action_variance']:.0%} of failures had low action variance"],
                suggested_params={"exploration_noise_multiplier": 2.0, "add_action_perturbation": True},
            ))

        if patterns.get("consecutive_low_reward", 0) > 0.5:
            prescriptions.append(Prescription(
                prescription_type=PrescriptionType.TASK_DECOMPOSITION,
                title="Decompose Task with Intermediate Rewards",
                description="Many failures show long stretches of negative reward. Break the task into subtasks with intermediate reward signals.",
                confidence=patterns["consecutive_low_reward"],
                evidence=[f"{patterns['consecutive_low_reward']:.0%} of failures had consecutive low-reward steps"],
                suggested_params={"add_intermediate_rewards": True},
            ))

        if patterns.get("low_total_reward", 0) > 0.3:
            prescriptions.append(Prescription(
                prescription_type=PrescriptionType.REWARD_SHAPING,
                title="Adjust Reward Shaping",
                description="Failures frequently have very low total reward. Consider adding dense reward signals or increasing the reward scale.",
                confidence=patterns["low_total_reward"],
                evidence=[f"{patterns['low_total_reward']:.0%} of failures had low total reward"],
                suggested_params={"reward_scale_multiplier": 5.0},
            ))

        if patterns.get("short_episode", 0) > 0.3:
            prescriptions.append(Prescription(
                prescription_type=PrescriptionType.ENVIRONMENT,
                title="Check Environment Reset and Initial States",
                description="Many episodes are terminating very early. Verify that the environment starts in a reachable state.",
                confidence=patterns["short_episode"],
                evidence=[f"{patterns['short_episode']:.0%} of failures were abnormally short"],
            ))

        if patterns.get("long_episode", 0) > 0.3:
            prescriptions.append(Prescription(
                prescription_type=PrescriptionType.POLICY_ARCHITECTURE,
                title="Add Episode Length Penalty or Timeout",
                description="Many episodes ran for an excessive number of steps, suggesting the agent is stuck in a loop.",
                confidence=patterns["long_episode"],
                evidence=[f"{patterns['long_episode']:.0%} of failures were abnormally long"],
            ))

        if gap is not None and hasattr(gap, "mean_gap_distance") and gap.mean_gap_distance > 1.0:
            nearest = [f"{d:.2f}" for d in gap.nearest_success_distances[:5]]
            prescriptions.append(Prescription(
                prescription_type=PrescriptionType.DATA_AUGMENTATION,
                title="Collect More Diverse Demonstrations",
                description=f"The embedding gap between successful and failed episodes is large (distance: {gap.mean_gap_distance:.2f}). Collect more demonstrations covering these failure regions.",
                confidence=min(gap.mean_gap_distance / 3.0, 1.0),
                evidence=[f"Mean embedding gap distance: {gap.mean_gap_distance:.2f}", f"Nearest success distances: {nearest}"],
            ))

        if not prescriptions:
            prescriptions.append(Prescription(
                prescription_type=PrescriptionType.DATA_AUGMENTATION,
                title="General Training Improvement",
                description="No strong failure patterns were detected. Consider general improvements: more training data, longer training, learning rate tuning.",
                priority=3, confidence=0.1,
            ))

        return prescriptions

    def _rank_prescriptions(self, prescriptions: list[Prescription]) -> list[Prescription]:
        for i, p in enumerate(sorted(prescriptions, key=lambda x: x.confidence, reverse=True)):
            p.priority = i + 1
        return sorted(prescriptions, key=lambda x: x.priority)

    def _build_summary(self, results: list[DetectionResult], prescriptions: list[Prescription]) -> str:
        n_failures = sum(1 for r in results if r.is_failure)
        top = prescriptions[0] if prescriptions else None
        summary = f"Analyzed {len(results)} episodes, {n_failures} identified as failures."
        if top:
            summary += f" Top recommendation: {top.title}."
        return summary
