"""Sim-to-real transfer readiness analysis.

Compares a simulation dataset against a real (or expected-real) dataset
and produces per-task transfer readiness scores based on embedding overlap,
visual domain gap, action distribution similarity, and episode diversity.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Sub-score weights for transfer readiness
_WEIGHTS = {
    "embedding_overlap": 0.35,
    "visual_gap": 0.25,
    "action_similarity": 0.25,
    "diversity": 0.15,
}


@dataclasses.dataclass
class Sim2RealReport:
    """Output of a sim-to-real readiness analysis."""

    overall_transfer_score: float
    per_task_scores: dict[str, dict]
    gap_analysis: dict
    prescription: list[dict]

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dictionary."""
        return {
            "overall_transfer_score": round(self.overall_transfer_score, 4),
            "per_task_scores": {
                task: {k: round(v, 4) if isinstance(v, float) else v for k, v in scores.items()}
                for task, scores in self.per_task_scores.items()
            },
            "gap_analysis": self.gap_analysis,
            "prescription": self.prescription,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


class Sim2RealProfiler:
    """Analyse transfer readiness between a sim and real dataset.

    Reuses existing ORBIT components: ``EmbeddingExtractor``,
    ``CoverageAnalyzer``, ``CapabilityScorer``, and ``DatasetLoader``.
    """

    def __init__(
        self,
        embedding_model: str = "google/siglip-base-patch16-224",
        device: str = "cpu",
    ) -> None:
        self.embedding_model = embedding_model
        self.device = device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        sim_dir: str | Path,
        real_dir: str | Path,
        task_descriptions: list[str] | None = None,
    ) -> Sim2RealReport:
        """Run the full sim-to-real readiness pipeline.

        Parameters
        ----------
        sim_dir:
            Path to the simulation dataset (HDF5 episodes + images).
        real_dir:
            Path to the real (or expected-real) dataset.
        task_descriptions:
            Optional list of task descriptions to score individually.

        Returns
        -------
        Sim2RealReport
        """
        from orbit.profile.coverage import CoverageAnalyzer
        from orbit.profile.embedding import EmbeddingExtractor
        from orbit.profile.loaders import DatasetLoader

        sim_dir = str(sim_dir)
        real_dir = str(real_dir)

        # 1. Extract embeddings
        logger.info("Extracting sim embeddings from %s", sim_dir)
        extractor = EmbeddingExtractor(
            model_name=self.embedding_model, device=self.device
        )
        sim_index = extractor.extract_from_directory(sim_dir)

        logger.info("Extracting real embeddings from %s", real_dir)
        real_index = extractor.extract_from_directory(real_dir)

        coverage_analyzer = CoverageAnalyzer()

        # 2. Embedding distribution overlap
        embedding_overlap = coverage_analyzer.compute_overlap(sim_index, real_index)
        logger.info("Embedding overlap: %.3f", embedding_overlap)

        # 3. Visual domain gap
        visual_gap_score = self._compute_visual_gap(
            coverage_analyzer, sim_index, real_index
        )
        logger.info("Visual domain gap: %.3f", visual_gap_score)

        # 4. Action distribution similarity
        sim_episodes = DatasetLoader.from_hdf5_directory(sim_dir)
        real_episodes = DatasetLoader.from_hdf5_directory(real_dir)
        action_similarity = compute_action_similarity(sim_episodes, real_episodes)
        logger.info("Action similarity: %.3f", action_similarity)

        # 5. Episode diversity / coverage of sim dataset
        sim_coverage = coverage_analyzer.analyze(sim_index)
        diversity = sim_coverage.overall_coverage_score
        logger.info("Sim diversity/coverage: %.3f", diversity)

        # 6. Per-task scoring
        sub_scores = {
            "embedding_overlap": embedding_overlap,
            "visual_gap": visual_gap_score,
            "action_similarity": action_similarity,
            "diversity": diversity,
        }

        per_task_scores = self._score_tasks(
            sim_index, sim_coverage, sim_episodes, task_descriptions, sub_scores
        )

        # 7. Overall transfer score (weighted mean of per-task scores)
        task_scores = [t["score"] for t in per_task_scores.values()]
        overall = float(np.mean(task_scores)) if task_scores else 0.0

        # 8. Gap analysis and prescriptions
        gap_analysis = self._build_gap_analysis(per_task_scores, sub_scores)
        prescription = self._build_prescription(per_task_scores)

        return Sim2RealReport(
            overall_transfer_score=overall,
            per_task_scores=per_task_scores,
            gap_analysis=gap_analysis,
            prescription=prescription,
        )

    # ------------------------------------------------------------------
    # Visual domain gap
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_visual_gap(
        coverage_analyzer,
        sim_index,
        real_index,
    ) -> float:
        """Summarize the visual domain gap as a single 0-1 score.

        Lower is better (less gap). Uses ``CoverageAnalyzer.find_gaps``
        to measure how poorly the sim embedding space covers the real one.
        """
        gaps = coverage_analyzer.find_gaps(sim_index, reference_index=real_index)
        if not gaps:
            return 0.0
        gap_scores = [g["gap_score"] for g in gaps]
        # Fraction of real frames with a large gap (> 0.3 threshold from find_gaps)
        # weighted by severity
        mean_gap = float(np.mean(gap_scores))
        return float(np.clip(mean_gap, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Per-task scoring
    # ------------------------------------------------------------------

    def _score_tasks(
        self,
        sim_index,
        sim_coverage,
        sim_episodes: list[dict],
        task_descriptions: list[str] | None,
        sub_scores: dict[str, float],
    ) -> dict[str, dict]:
        """Compute per-task transfer readiness scores."""
        # Compute the base transfer score from sub-scores
        base_transfer = _combine_sub_scores(sub_scores)

        if not task_descriptions:
            return {
                "general": {
                    "score": base_transfer,
                    **sub_scores,
                }
            }

        # Use CapabilityScorer to get per-task capability in sim dataset
        try:
            from orbit.profile.capability import CapabilityScorer
            from orbit.profile.quality import QualityEstimator

            quality = QualityEstimator().estimate_quality(sim_episodes)
            scorer = CapabilityScorer(
                model_name=self.embedding_model, device=self.device
            )
            capabilities = scorer.score_tasks(
                sim_index, sim_coverage, quality, task_descriptions
            )
        except Exception:
            logger.warning(
                "CapabilityScorer unavailable; using uniform task weights",
                exc_info=True,
            )
            capabilities = None

        per_task: dict[str, dict] = {}
        for i, task in enumerate(task_descriptions):
            if capabilities and i < len(capabilities):
                cap = capabilities[i]
                # Modulate base transfer by task-specific capability
                task_weight = cap.score
            else:
                task_weight = 1.0

            task_transfer = base_transfer * (0.5 + 0.5 * task_weight)
            task_transfer = float(np.clip(task_transfer, 0.0, 1.0))

            per_task[task] = {
                "score": task_transfer,
                **sub_scores,
                "sim_capability": task_weight,
            }

        return per_task

    # ------------------------------------------------------------------
    # Gap analysis & prescriptions
    # ------------------------------------------------------------------

    @staticmethod
    def _build_gap_analysis(
        per_task_scores: dict[str, dict],
        sub_scores: dict[str, float],
    ) -> dict:
        """Identify the biggest gaps and generate recommendations."""
        # Rank sub-scores to find weakest dimensions
        ranked = sorted(
            [
                ("embedding_overlap", sub_scores["embedding_overlap"]),
                ("visual_domain_similarity", 1.0 - sub_scores["visual_gap"]),
                ("action_similarity", sub_scores["action_similarity"]),
                ("diversity", sub_scores["diversity"]),
            ],
            key=lambda x: x[1],
        )

        biggest_gaps = [
            {"dimension": name, "score": round(score, 4)}
            for name, score in ranked
            if score < 0.7
        ]

        recommendations: list[str] = []
        if sub_scores["embedding_overlap"] < 0.5:
            recommendations.append(
                "Collect more real-world data in environments that match sim scenarios "
                "to increase embedding overlap."
            )
        if sub_scores["visual_gap"] > 0.5:
            recommendations.append(
                "Apply domain randomization (lighting, textures, backgrounds) in sim "
                "to reduce the visual domain gap."
            )
        if sub_scores["action_similarity"] < 0.5:
            recommendations.append(
                "Review action space calibration — sim and real action distributions "
                "differ significantly. Check actuator models and action scaling."
            )
        if sub_scores["diversity"] < 0.5:
            recommendations.append(
                "Increase sim episode diversity: vary initial conditions, object "
                "placements, and task parameters."
            )

        # Worst-performing tasks
        worst_tasks = sorted(per_task_scores.items(), key=lambda x: x[1]["score"])[:3]
        domain_shift = (
            f"Overall domain shift is {'high' if sub_scores['visual_gap'] > 0.5 else 'moderate' if sub_scores['visual_gap'] > 0.3 else 'low'}. "
            f"Embedding overlap is {sub_scores['embedding_overlap']:.0%}."
        )

        return {
            "biggest_gaps": biggest_gaps,
            "domain_shift_summary": domain_shift,
            "worst_tasks": [
                {"task": t, "score": round(s["score"], 4)} for t, s in worst_tasks
            ],
            "recommendations": recommendations,
        }

    @staticmethod
    def _build_prescription(per_task_scores: dict[str, dict]) -> list[dict]:
        """Generate ranked prescriptions for closing transfer gaps."""
        prescriptions: list[dict] = []

        for task, scores in per_task_scores.items():
            transfer = scores["score"]
            if transfer >= 0.7:
                continue

            gap = 1.0 - transfer
            estimated_demos = max(5, int(gap * 50))

            # Identify the weakest sub-dimension
            sub_dims = {
                "embedding_overlap": scores.get("embedding_overlap", 0),
                "visual_domain_similarity": 1.0 - scores.get("visual_gap", 0),
                "action_similarity": scores.get("action_similarity", 0),
                "diversity": scores.get("diversity", 0),
            }
            weakest_dim = min(sub_dims, key=sub_dims.get)  # type: ignore[arg-type]

            rationale_map = {
                "embedding_overlap": "Collect real demonstrations in visually similar setups to bridge the embedding gap.",
                "visual_domain_similarity": "Add domain randomization in sim or collect real data with diverse visual conditions.",
                "action_similarity": "Calibrate sim actuator models or collect real teleoperation data to align action distributions.",
                "diversity": "Increase sim coverage with more varied initial conditions and object configurations.",
            }

            prescriptions.append(
                {
                    "task": task,
                    "action": rationale_map.get(weakest_dim, "Collect more real data."),
                    "estimated_demos": estimated_demos,
                    "current_score": round(transfer, 4),
                    "target_score": round(min(transfer + 0.3, 1.0), 4),
                    "weakest_dimension": weakest_dim,
                    "rationale": f"Weakest dimension is {weakest_dim} "
                    f"({sub_dims[weakest_dim]:.2f}). "
                    f"Closing this gap would most improve transfer readiness.",
                }
            )

        # Sort by severity (lowest score first)
        prescriptions.sort(key=lambda p: p["current_score"])
        for i, p in enumerate(prescriptions):
            p["priority"] = i + 1

        return prescriptions


# ------------------------------------------------------------------
# Standalone utilities
# ------------------------------------------------------------------


def compute_action_similarity(
    sim_episodes: list[dict],
    real_episodes: list[dict],
) -> float:
    """Compute action distribution similarity between sim and real episodes.

    Uses per-dimension Wasserstein-1 distance, normalized to a 0-1
    similarity score.

    Returns 0.0 if either dataset has no episodes or no actions.
    """
    sim_actions = _collect_actions(sim_episodes)
    real_actions = _collect_actions(real_episodes)

    if sim_actions is None or real_actions is None:
        return 0.0

    # Align action dimensions (use minimum)
    dim = min(sim_actions.shape[1], real_actions.shape[1])
    sim_actions = sim_actions[:, :dim]
    real_actions = real_actions[:, :dim]

    from scipy.stats import wasserstein_distance

    distances = []
    for d in range(dim):
        dist = wasserstein_distance(sim_actions[:, d], real_actions[:, d])
        distances.append(dist)

    mean_dist = float(np.mean(distances))
    similarity = 1.0 / (1.0 + mean_dist)
    return float(np.clip(similarity, 0.0, 1.0))


def _collect_actions(episodes: list[dict]) -> np.ndarray | None:
    """Concatenate all actions from a list of episode dicts."""
    arrays = []
    for ep in episodes:
        actions = ep.get("actions")
        if actions is not None and len(actions) > 0:
            arrays.append(np.asarray(actions, dtype=np.float32))
    if not arrays:
        return None
    return np.concatenate(arrays, axis=0)


def _combine_sub_scores(sub_scores: dict[str, float]) -> float:
    """Compute weighted transfer readiness score from sub-scores."""
    score = (
        _WEIGHTS["embedding_overlap"] * sub_scores["embedding_overlap"]
        + _WEIGHTS["visual_gap"] * (1.0 - sub_scores["visual_gap"])
        + _WEIGHTS["action_similarity"] * sub_scores["action_similarity"]
        + _WEIGHTS["diversity"] * sub_scores["diversity"]
    )
    return float(np.clip(score, 0.0, 1.0))
