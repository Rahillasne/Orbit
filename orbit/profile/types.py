"""Dataclasses for the orbit.profile module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


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


@dataclass
class QualityMetrics:
    """Per-episode and aggregate quality scores."""

    episode_scores: dict[int, float]  # episode_id -> quality score
    aggregate_score: float  # mean quality
    low_quality_episodes: list[int]  # episodes below threshold
    mutual_information_estimate: float  # MI between states and actions


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
