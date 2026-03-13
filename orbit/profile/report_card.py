"""Report card generation — graded assessment of robot datasets.

Converts raw profiler scores into letter grades (A–F), detects strengths
and weaknesses via a rule engine, and renders a rich CLI report card.
"""

from __future__ import annotations

import logging
import warnings
from datetime import datetime

import numpy as np

from orbit.profile.types import (
    DatasetGap,
    DatasetProfile,
    DatasetReportCard,
    Prescription,
    TaskAssessment,
)

logger = logging.getLogger(__name__)

# Grade thresholds
_GRADE_THRESHOLDS = [
    ("A", 0.85),
    ("B", 0.70),
    ("C", 0.50),
    ("D", 0.30),
]

# Volume thresholds (episode count → score)
_VOLUME_THRESHOLDS = [
    (100, 0.90),
    (50, 0.75),
    (25, 0.55),
    (10, 0.35),
]

# Overall weights
_OVERALL_WEIGHTS = {
    "coverage": 0.30,
    "quality": 0.30,
    "diversity": 0.25,
    "volume": 0.15,
}


def score_to_grade(score: float) -> str:
    """Convert a 0–1 score to a letter grade."""
    for grade, threshold in _GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return "F"


class ReportCardGenerator:
    """Generates a graded report card from a DatasetProfile."""

    def generate(self, profile: DatasetProfile) -> DatasetReportCard:
        """Generate a complete report card from a profile."""
        coverage_score = self._compute_coverage_score(profile)
        quality_score = self._compute_quality_score(profile)
        diversity_score = self._compute_diversity_score(profile)
        volume_score = self._compute_volume_score(profile)

        overall_score = (
            _OVERALL_WEIGHTS["coverage"] * coverage_score
            + _OVERALL_WEIGHTS["quality"] * quality_score
            + _OVERALL_WEIGHTS["diversity"] * diversity_score
            + _OVERALL_WEIGHTS["volume"] * volume_score
        )

        strengths = self._detect_strengths(
            profile, coverage_score, quality_score, diversity_score, volume_score
        )
        weaknesses = self._detect_weaknesses(
            profile, coverage_score, quality_score, diversity_score, volume_score
        )
        gaps = self._detect_gaps(profile)
        prescriptions = self._generate_prescriptions(profile, gaps)
        task_assessments = [self._assess_task(cap) for cap in profile.capabilities]

        return DatasetReportCard(
            dataset_name=profile.dataset_name,
            overall_grade=score_to_grade(overall_score),
            overall_score=overall_score,
            coverage_grade=score_to_grade(coverage_score),
            coverage_score=coverage_score,
            quality_grade=score_to_grade(quality_score),
            quality_score=quality_score,
            diversity_grade=score_to_grade(diversity_score),
            diversity_score=diversity_score,
            volume_grade=score_to_grade(volume_score),
            volume_score=volume_score,
            strengths=strengths,
            weaknesses=weaknesses,
            gaps=gaps,
            prescriptions=prescriptions,
            task_assessments=task_assessments,
            timestamp=datetime.now().isoformat(),
        )

    # ------------------------------------------------------------------
    # Score computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_coverage_score(profile: DatasetProfile) -> float:
        """Coverage score from the existing coverage analysis."""
        return float(np.clip(profile.coverage.overall_coverage_score, 0.0, 1.0))

    @staticmethod
    def _compute_quality_score(profile: DatasetProfile) -> float:
        """Quality score from the existing quality estimator."""
        return float(np.clip(profile.quality.aggregate_score, 0.0, 1.0))

    @staticmethod
    def _compute_diversity_score(profile: DatasetProfile) -> float:
        """Diversity score based on embedding cluster count and entropy.

        Uses HDBSCAN on reconstructed embeddings from the FAISS index.
        Falls back to KMeans if HDBSCAN fails.
        """
        idx = profile.embedding_index
        if idx.num_embeddings < 5:
            return 0.1  # Too few embeddings for meaningful diversity

        # Reconstruct embeddings from FAISS index
        try:
            embeddings = np.vstack(
                [idx.index.reconstruct(i) for i in range(idx.num_embeddings)]
            ).astype(np.float32)
        except Exception:
            # FAISS index may not support reconstruct
            return 0.5  # Neutral fallback

        # Reduce dimensionality for clustering
        n_components = min(10, embeddings.shape[1], idx.num_embeddings - 1)
        if n_components < 2:
            return 0.3

        try:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=n_components)
            reduced = pca.fit_transform(embeddings)
        except Exception:
            return 0.5

        # Try HDBSCAN first, fall back to KMeans
        n_clusters = _count_clusters(reduced, idx.num_embeddings)

        # Compute entropy of cluster assignments for diversity measure
        # More clusters + even distribution = higher diversity
        if n_clusters <= 1:
            return 0.15

        # Normalize: more clusters relative to data size = more diverse
        # Cap at ~20 clusters for normalization
        cluster_ratio = min(n_clusters / 20.0, 1.0)

        # Also consider embedding spread (std of pairwise distances)
        if len(reduced) > 1:
            sample_size = min(200, len(reduced))
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(len(reduced), size=sample_size, replace=False)
            sample = reduced[sample_idx]
            pairwise_dists = np.linalg.norm(sample[:, None] - sample[None, :], axis=2)
            spread = float(np.mean(pairwise_dists))
            # Normalize spread (heuristic: spread of ~2-5 in PCA space is good)
            spread_score = float(np.clip(spread / 5.0, 0.0, 1.0))
        else:
            spread_score = 0.0

        diversity = 0.6 * cluster_ratio + 0.4 * spread_score
        return float(np.clip(diversity, 0.0, 1.0))

    @staticmethod
    def _compute_volume_score(profile: DatasetProfile) -> float:
        """Volume score from episode count thresholds."""
        n = profile.num_episodes
        for threshold, score in _VOLUME_THRESHOLDS:
            if n >= threshold:
                return score
        return 0.15  # < 10 episodes

    # ------------------------------------------------------------------
    # Strength / weakness detection (rule engine)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_strengths(
        profile: DatasetProfile,
        coverage: float,
        quality: float,
        diversity: float,
        volume: float,
    ) -> list[str]:
        """Detect dataset strengths via rule engine."""
        strengths: list[str] = []

        # Coverage rules
        if coverage >= 0.80:
            strengths.append("Excellent scene variety across embedding space")
        elif coverage >= 0.60:
            strengths.append("Good scene coverage in embedding space")

        # Quality rules
        if quality >= 0.80:
            strengths.append("High-quality, consistent demonstrations")
        elif quality >= 0.65:
            strengths.append("Above-average demonstration quality")

        # Quality signal breakdown
        sb = profile.quality.signal_breakdown
        if sb:
            if sb.mutual_information >= 0.70:
                strengths.append("Strong state-action coupling — intentional demonstrations")
            if sb.action_smoothness >= 0.80:
                strengths.append("Smooth expert-quality trajectories (low jerk)")
            if sb.episode_completion >= 0.75:
                strengths.append("Good episode completion rates (goal convergence)")
            if sb.observation_consistency >= 0.90:
                strengths.append("Clean sensor data with no anomalies")
            if sb.demonstration_quality >= 0.75:
                strengths.append("Expert-level action variance patterns")

        # Diversity rules
        if diversity >= 0.70:
            strengths.append("Multiple distinct manipulation strategies detected")
        elif diversity >= 0.50:
            strengths.append("Moderate scenario diversity")

        # Volume rules
        if volume >= 0.85:
            strengths.append("Large dataset — sufficient for robust training")
        elif volume >= 0.70:
            strengths.append("Sufficient data volume for fine-tuning")

        # Capability rules
        high_cap_count = sum(1 for c in profile.capabilities if c.score >= 0.70)
        if high_cap_count >= 3:
            strengths.append(f"Strong capability across {high_cap_count} tasks")
        elif high_cap_count >= 1:
            top = max(profile.capabilities, key=lambda c: c.score)
            strengths.append(f"Strong coverage for '{top.task_description}'")

        # Ensure at least one strength
        if not strengths:
            strengths.append("Dataset is available for analysis")

        return strengths

    @staticmethod
    def _detect_weaknesses(
        profile: DatasetProfile,
        coverage: float,
        quality: float,
        diversity: float,
        volume: float,
    ) -> list[str]:
        """Detect dataset weaknesses via rule engine."""
        weaknesses: list[str] = []

        # Coverage rules
        if coverage < 0.30:
            weaknesses.append("Very sparse embedding space coverage")
        elif coverage < 0.50:
            weaknesses.append("Below-average scene coverage")

        # Quality rules
        if quality < 0.30:
            weaknesses.append("Poor demonstration quality — noisy or inconsistent")
        elif quality < 0.50:
            weaknesses.append("Below-average demonstration quality")

        # Quality signal breakdown
        sb = profile.quality.signal_breakdown
        if sb:
            if sb.mutual_information < 0.30:
                weaknesses.append(
                    "Low state-action mutual information — demonstrations may be random"
                )
            if sb.action_smoothness < 0.30:
                weaknesses.append("Jerky actions — demonstrations lack smoothness")
            if sb.episode_completion < 0.30:
                weaknesses.append("Many episodes fail to converge to goal states")
            if sb.observation_consistency < 0.50:
                weaknesses.append("Sensor inconsistencies detected (NaN, jumps, or dropouts)")
            if sb.demonstration_quality < 0.30:
                weaknesses.append("Action variance suggests non-expert demonstrations")

        # Diversity rules
        if diversity < 0.20:
            weaknesses.append("Very low diversity — nearly identical episodes")
        elif diversity < 0.40:
            weaknesses.append("Limited environmental/scenario variation")

        # Volume rules
        if volume < 0.20:
            weaknesses.append("Critically insufficient episode count (< 10)")
        elif volume < 0.40:
            weaknesses.append("Dataset may be too small for reliable training")

        # Cross-dimension rules
        if coverage < 0.40 and profile.num_episodes >= 100:
            weaknesses.append("Large dataset but redundant — many similar episodes")
        if quality >= 0.80 and diversity < 0.30:
            weaknesses.append("High-quality demos but limited environmental variation")
        if coverage >= 0.70 and quality < 0.40:
            weaknesses.append("Good scene variety but inconsistent demonstrations")

        # Low quality episodes
        n_low = len(profile.quality.low_quality_episodes)
        if n_low > 0:
            pct = n_low / max(len(profile.quality.episode_scores), 1) * 100
            if pct >= 30:
                weaknesses.append(f"{n_low} low-quality episodes ({pct:.0f}% of dataset)")

        # Capability rules
        low_cap_count = sum(1 for c in profile.capabilities if c.score < 0.30)
        if low_cap_count >= 2:
            weaknesses.append(f"Poor coverage for {low_cap_count} target tasks")

        return weaknesses

    # ------------------------------------------------------------------
    # Gap detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_gaps(profile: DatasetProfile) -> list[DatasetGap]:
        """Detect gaps in the dataset using sparse regions and capability gaps."""
        gaps: list[DatasetGap] = []

        # Sparse region gaps
        for i, region in enumerate(profile.coverage.sparse_regions):
            density = region.get("density", 0.0)
            if density < 0.3:
                severity = "Critical" if density < 0.1 else "Major"
                desc = region.get("description", f"Sparse region {i} in embedding space")
                gaps.append(
                    DatasetGap(
                        dimension=f"coverage_region_{i}",
                        severity=severity,
                        description=f"Sparse coverage: {desc}",
                        recommendation="Collect demonstrations covering this region",
                    )
                )

        # Capability-based gaps
        for cap in profile.capabilities:
            if cap.score < 0.50 and cap.gap_description:
                if cap.score < 0.20:
                    severity = "Critical"
                elif cap.score < 0.35:
                    severity = "Major"
                else:
                    severity = "Minor"
                gaps.append(
                    DatasetGap(
                        dimension=f"task:{cap.task_description}",
                        severity=severity,
                        description=cap.gap_description,
                        recommendation=f"Collect demonstrations for '{cap.task_description}'",
                    )
                )

        # Quality-based gaps
        sb = profile.quality.signal_breakdown
        if sb:
            if sb.action_smoothness < 0.30:
                gaps.append(
                    DatasetGap(
                        dimension="action_quality",
                        severity="Major",
                        description=(
                            "Demonstrations have excessive jerk — likely teleoperation noise"
                        ),
                        recommendation="Re-collect with smoother control or apply action smoothing",
                    )
                )
            if sb.observation_consistency < 0.50:
                gaps.append(
                    DatasetGap(
                        dimension="sensor_quality",
                        severity="Major",
                        description="Sensor data has anomalies (NaN values, sudden jumps)",
                        recommendation="Check sensor calibration and clean data",
                    )
                )

        # Sort by severity
        severity_order = {"Critical": 0, "Major": 1, "Minor": 2}
        gaps.sort(key=lambda g: severity_order.get(g.severity, 3))

        return gaps

    # ------------------------------------------------------------------
    # Prescriptions
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_prescriptions(
        profile: DatasetProfile, gaps: list[DatasetGap]
    ) -> list[Prescription]:
        """Generate ranked data collection prescriptions from gaps."""
        prescriptions: list[Prescription] = []

        for i, gap in enumerate(gaps):
            if gap.severity == "Critical":
                demos = 20
                expected = "Would significantly improve the weakest dimension"
            elif gap.severity == "Major":
                demos = 15
                expected = "Would improve grade by approximately one letter"
            else:
                demos = 10
                expected = "Would provide incremental improvement"

            prescriptions.append(
                Prescription(
                    priority=i + 1,
                    action=gap.recommendation,
                    estimated_demos=demos,
                    expected_improvement=expected,
                    dimension=gap.dimension,
                )
            )

        return prescriptions[:10]  # Cap at 10 prescriptions

    # ------------------------------------------------------------------
    # Task assessment
    # ------------------------------------------------------------------

    @staticmethod
    def _assess_task(cap) -> TaskAssessment:
        """Convert a CapabilityScore to a graded TaskAssessment."""
        grade = score_to_grade(cap.score)

        # Relevance label
        if cap.score >= 0.60:
            relevance = "High"
        elif cap.score >= 0.30:
            relevance = "Medium"
        elif cap.score >= 0.10:
            relevance = "Low"
        else:
            relevance = "Not Found"

        # Coverage label
        if cap.score >= 0.70:
            coverage_label = "Well Covered"
        elif cap.score >= 0.40:
            coverage_label = "Partially Covered"
        else:
            coverage_label = "Poorly Covered"

        # Confidence label
        if cap.confidence >= 0.70:
            confidence = "High"
        elif cap.confidence >= 0.40:
            confidence = "Medium"
        else:
            confidence = "Low"

        # Finding
        if cap.score >= 0.70:
            finding = f"Dataset has strong coverage for '{cap.task_description}'"
        elif cap.score >= 0.40:
            finding = (
                f"Partial coverage for '{cap.task_description}' — "
                f"additional demonstrations would help"
            )
        elif cap.gap_description:
            finding = cap.gap_description
        else:
            finding = f"Insufficient data for '{cap.task_description}'"

        return TaskAssessment(
            task=cap.task_description,
            grade=grade,
            score=cap.score,
            relevance=relevance,
            coverage=coverage_label,
            confidence=confidence,
            finding=finding,
        )

    # ------------------------------------------------------------------
    # CLI rendering
    # ------------------------------------------------------------------

    def render_cli(
        self,
        card: DatasetReportCard,
        predicted_success_rate: float | None = None,
        prediction_ci: tuple[float, float] | None = None,
        prediction_confidence: str | None = None,
    ) -> str:
        """Render the report card as a formatted string for terminal display.

        Parameters
        ----------
        card:
            The report card to render.
        predicted_success_rate:
            Optional predicted success rate from the learned predictor.
        prediction_ci:
            Optional (low, high) confidence interval.
        prediction_confidence:
            Optional confidence level string ("high", "medium", "low").
        """
        lines: list[str] = []
        w = 52  # inner width

        def bar(char: str = "=") -> str:
            return char * (w + 2)

        def pad(text: str) -> str:
            return f"  {text}"

        # Header
        lines.append(bar("="))
        lines.append(pad("ORBIT Dataset Report Card"))
        lines.append(pad(f"Dataset: {card.dataset_name}"))
        lines.append(bar("="))
        lines.append("")

        # Predicted success rate (primary output)
        if predicted_success_rate is not None:
            lines.append(pad(f"Predicted Success Rate:  {predicted_success_rate:.1%}"))
            if prediction_ci:
                lines.append(
                    pad(
                        f"  90% CI: [{prediction_ci[0]:.1%}, {prediction_ci[1]:.1%}]"
                        f"  (confidence: {prediction_confidence or 'unknown'})"
                    )
                )
            lines.append("")

        # Overall grade (human-readable summary)
        lines.append(pad(f"Overall Grade:  {card.overall_grade}  ({card.overall_score:.2f})"))
        if predicted_success_rate is not None:
            lines.append(
                pad(
                    "  Note: The grade is a human-readable summary. "
                    "The predicted success rate above is the calibrated estimate."
                )
            )
        lines.append("")

        # Sub-grades
        lines.append(bar("-"))
        lines.append(
            pad(
                f"Coverage:   {card.coverage_grade} ({card.coverage_score:.2f})"
                f"   |   Quality:    {card.quality_grade} ({card.quality_score:.2f})"
            )
        )
        lines.append(
            pad(
                f"Diversity:  {card.diversity_grade} ({card.diversity_score:.2f})"
                f"   |   Volume:     {card.volume_grade} ({card.volume_score:.2f})"
            )
        )
        lines.append(bar("-"))
        lines.append("")

        # Strengths
        if card.strengths:
            lines.append(pad("Strengths:"))
            for s in card.strengths:
                lines.append(pad(f"  + {s}"))
            lines.append("")

        # Weaknesses
        if card.weaknesses:
            lines.append(pad("Weaknesses:"))
            for w_str in card.weaknesses:
                lines.append(pad(f"  - {w_str}"))
            lines.append("")

        # Task assessments
        if card.task_assessments:
            lines.append(bar("-"))
            lines.append(pad("Task Grades:"))
            for ta in card.task_assessments:
                status = f"{ta.grade} ({ta.score:.2f})"
                lines.append(pad(f"  {ta.task:<30s}  {status:<12s}  [{ta.coverage}]"))
            lines.append("")

        # Gaps
        if card.gaps:
            lines.append(bar("-"))
            lines.append(pad("Detected Gaps:"))
            for gap in card.gaps[:5]:
                lines.append(pad(f"  [{gap.severity}] {gap.description}"))
            lines.append("")

        # Prescriptions
        if card.prescriptions:
            lines.append(bar("-"))
            lines.append(pad("Prescriptions:"))
            for p in card.prescriptions[:5]:
                lines.append(pad(f"  {p.priority}. {p.action} (~{p.estimated_demos} demos)"))
            lines.append("")

        lines.append(bar("="))
        return "\n".join(lines)


def _count_clusters(reduced: np.ndarray, n_embeddings: int) -> int:
    """Count distinct clusters using HDBSCAN with KMeans fallback."""
    # Try HDBSCAN
    try:
        import hdbscan

        min_cluster_size = max(3, n_embeddings // 20)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=2)
            labels = clusterer.fit_predict(reduced)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters >= 1:
            return n_clusters
    except ImportError:
        pass
    except Exception:
        pass

    # Fall back to KMeans with silhouette score
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        best_k = 1
        best_score = -1.0
        max_k = min(10, n_embeddings // 3)
        if max_k < 2:
            return 1

        for k in range(2, max_k + 1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                km = KMeans(n_clusters=k, n_init="auto", random_state=42)
                labels = km.fit_predict(reduced)
            score = silhouette_score(reduced, labels)
            if score > best_score:
                best_score = score
                best_k = k
        return best_k
    except Exception:
        return 1
