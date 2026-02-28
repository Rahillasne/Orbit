"""Data models for the frame-level embedding analyzer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingAnalyzerConfig:
    """Configuration for the frame-level embedding analyzer."""

    model_name: str = "google/siglip-base-patch16-224"
    device: str = "cpu"
    batch_size: int = 32
    num_neighbors: int = 5
    cache_dir: str = ".orbit_cache/embeddings"
    use_gpu_faiss: bool = False
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "cosine"
    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: int = 3


# ---------------------------------------------------------------------------
# Frame-level result
# ---------------------------------------------------------------------------


class FrameGapResult(BaseModel):
    """Gap analysis result for a single deployment frame."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    frame_idx: int
    episode_id: UUID
    gap_score: float
    nearest_distances: list[float]
    nearest_indices: list[int]


# ---------------------------------------------------------------------------
# Episode-level aggregation
# ---------------------------------------------------------------------------


class EpisodeGapSummary(BaseModel):
    """Aggregated gap metrics for a single deployment episode."""

    episode_id: UUID
    outcome: str
    mean_gap: float
    max_gap: float
    gap_percentile_95: float
    gap_trajectory: list[float]
    num_frames: int


# ---------------------------------------------------------------------------
# Failure clustering
# ---------------------------------------------------------------------------


class FailureCluster(BaseModel):
    """A single cluster of failure frames."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cluster_id: int
    size: int
    avg_gap_score: float
    representative_frame_indices: list[int]
    representative_episode_ids: list[str]
    temporal_distribution: dict[str, Any] = Field(default_factory=dict)
    centroid: list[float] = Field(default_factory=list)


class FailureClusterReport(BaseModel):
    """Report from clustering all failure frames."""

    clusters: list[FailureCluster] = Field(default_factory=list)
    num_failure_frames: int = 0
    num_noise_frames: int = 0
    num_clusters: int = 0


# ---------------------------------------------------------------------------
# Full pipeline result
# ---------------------------------------------------------------------------


class AnalysisReport(BaseModel):
    """Complete result from running the full analysis pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    episode_summaries: list[EpisodeGapSummary] = Field(default_factory=list)
    cluster_report: FailureClusterReport | None = None
    training_embedding_count: int = 0
    deployment_embedding_count: int = 0
    visualization_paths: dict[str, str] = Field(default_factory=dict)
