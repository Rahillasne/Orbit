"""Dataclasses for the orbit.profile module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class QualitySignalBreakdown:
    """Per-signal quality scores (all 0-1)."""

    mutual_information: float
    action_smoothness: float
    episode_completion: float
    observation_consistency: float
    demonstration_quality: float


@dataclass
class EmbeddingStats:
    """Distribution statistics computed from visual embeddings."""

    mean_norm: float = 0.0
    std_norm: float = 0.0
    mean_pairwise_cosine: float = 0.0
    std_pairwise_cosine: float = 0.0
    min_pairwise_cosine: float = 0.0
    max_pairwise_cosine: float = 0.0
    num_clusters: int = 0
    noise_ratio: float = 0.0
    silhouette_score: float = 0.0
    calinski_harabasz: float = 0.0
    convex_hull_volume: float = 0.0
    effective_dimensionality: float = 0.0
    mean_sequential_distance: float = 0.0
    std_sequential_distance: float = 0.0
    mean_episode_trajectory_length: float = 0.0
    std_episode_trajectory_length: float = 0.0
    skewness_pc1: float = 0.0
    kurtosis_pc1: float = 0.0
    entropy_estimate: float = 0.0
    uniformity_score: float = 0.0


@dataclass
class ActionStats:
    """Distribution statistics computed from episode action data."""

    dimensionality: float = 0.0
    mean_magnitude: float = 0.0
    std_magnitude: float = 0.0
    smoothness: float = 0.0
    action_range_utilization: float = 0.0
    num_action_modes: float = 0.0
    mean_episode_action_entropy: float = 0.0
    cross_episode_action_consistency: float = 0.0
    action_dimensionality_effective: float = 0.0
    mean_action_autocorrelation: float = 0.0
    zero_action_fraction: float = 0.0
    action_boundary_fraction: float = 0.0


@dataclass
class ScoreBreakdown:
    """Component breakdown of the capability score."""

    visual_relevance: float
    data_quality: float
    coverage_diversity: float
    volume: float


@dataclass
class ScoringWeights:
    """Configurable weights for the capability scoring formula."""

    visual_relevance: float = 0.35
    data_quality: float = 0.35
    coverage_diversity: float = 0.20
    volume: float = 0.10


@dataclass
class EmbeddingIndex:
    """FAISS index + metadata for a dataset's visual embeddings."""

    index: Any  # faiss.Index
    episode_ids: list[int]
    frame_indices: list[int]
    dimension: int
    num_embeddings: int


@dataclass
class CoverageMap:
    """Density map of embedding space coverage."""

    dense_regions: list[dict]  # {center: np.array, density: float, description: str}
    sparse_regions: list[dict]  # same structure
    overall_coverage_score: float  # 0-1
    umap_projection: np.ndarray | None  # 2D projection for visualization


@dataclass
class CapabilityScore:
    """How capable the dataset is for a specific task."""

    task_description: str
    score: float  # 0-1
    confidence: float  # 0-1
    supporting_episodes: int
    action_diversity: float  # entropy of actions in relevant episodes
    environment_diversity: float  # diversity of visual conditions
    gap_description: str | None  # what's missing if score < threshold
    score_breakdown: ScoreBreakdown | None = None


@dataclass
class QualityMetrics:
    """Per-episode and aggregate quality scores."""

    episode_scores: dict[int, float]  # episode_id -> quality score
    aggregate_score: float  # mean quality
    low_quality_episodes: list[int]  # episodes below threshold
    mutual_information_estimate: float  # MI between states and actions
    signal_breakdown: QualitySignalBreakdown | None = None


@dataclass
class DatasetProfile:
    """Complete profile of a dataset's capabilities."""

    dataset_name: str
    num_episodes: int
    num_frames: int
    embedding_index: EmbeddingIndex
    coverage: CoverageMap
    capabilities: list[CapabilityScore]
    quality: QualityMetrics
    prescriptions: list[dict] = field(default_factory=list)  # what data to collect next
    timestamp: str = ""
    embedding_stats: EmbeddingStats | None = None
    action_stats: ActionStats | None = None
    predicted_success_rate: float | None = None
    prediction_ci_low: float | None = None
    prediction_ci_high: float | None = None
    prediction_confidence: str | None = None


# ---------------------------------------------------------------------------
# Report card types
# ---------------------------------------------------------------------------


@dataclass
class DatasetGap:
    """A detected gap in dataset coverage."""

    dimension: str  # e.g. "rotation", "lighting", "speed_variation"
    severity: str  # "Critical", "Major", "Minor"
    description: str  # Human-readable description
    recommendation: str  # What to collect


@dataclass
class TaskAssessment:
    """Graded assessment for a single task."""

    task: str
    grade: str  # A, B, C, D, F
    score: float
    relevance: str  # "High", "Medium", "Low", "Not Found"
    coverage: str  # "Well Covered", "Partially Covered", "Poorly Covered"
    confidence: str  # "High", "Medium", "Low"
    finding: str  # One-sentence summary


@dataclass
class Prescription:
    """A specific data collection recommendation."""

    priority: int
    action: str  # e.g. "Collect 20 demos of cup grasping with varied lighting"
    estimated_demos: int
    expected_improvement: str  # e.g. "Would improve coverage grade from C to B"
    dimension: str  # Which gap this addresses


@dataclass
class DatasetReportCard:
    """Human-readable graded assessment of a robot dataset."""

    dataset_name: str
    overall_grade: str  # A, B, C, D, F
    overall_score: float
    coverage_grade: str
    coverage_score: float
    quality_grade: str
    quality_score: float
    diversity_grade: str
    diversity_score: float
    volume_grade: str
    volume_score: float
    strengths: list[str]
    weaknesses: list[str]
    gaps: list[DatasetGap]
    prescriptions: list[Prescription]
    task_assessments: list[TaskAssessment]
    timestamp: str = ""

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "overall_grade": self.overall_grade,
            "overall_score": round(self.overall_score, 3),
            "grades": {
                "coverage": {"grade": self.coverage_grade, "score": round(self.coverage_score, 3)},
                "quality": {"grade": self.quality_grade, "score": round(self.quality_score, 3)},
                "diversity": {
                    "grade": self.diversity_grade,
                    "score": round(self.diversity_score, 3),
                },
                "volume": {"grade": self.volume_grade, "score": round(self.volume_score, 3)},
            },
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "gaps": [
                {
                    "dimension": g.dimension,
                    "severity": g.severity,
                    "description": g.description,
                    "recommendation": g.recommendation,
                }
                for g in self.gaps
            ],
            "prescriptions": [
                {
                    "priority": p.priority,
                    "action": p.action,
                    "estimated_demos": p.estimated_demos,
                    "expected_improvement": p.expected_improvement,
                    "dimension": p.dimension,
                }
                for p in self.prescriptions
            ],
            "task_assessments": [
                {
                    "task": t.task,
                    "grade": t.grade,
                    "score": round(t.score, 3),
                    "relevance": t.relevance,
                    "coverage": t.coverage,
                    "confidence": t.confidence,
                    "finding": t.finding,
                }
                for t in self.task_assessments
            ],
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        """Convert to a JSON string."""
        import json

        return json.dumps(self.to_dict(), indent=2)
