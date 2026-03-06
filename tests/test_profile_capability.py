"""Tests for orbit.profile.capability (CapabilityScorer)."""

from __future__ import annotations

import faiss
import numpy as np
import pytest

from orbit.profile.capability import CapabilityScorer
from orbit.profile.types import (
    CapabilityScore,
    CoverageMap,
    DatasetProfile,
    EmbeddingIndex,
    QualityMetrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 64


def _make_cluster(
    n: int, dim: int, center: np.ndarray, std: float = 0.05, seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts = center + rng.standard_normal((n, dim)).astype(np.float32) * std
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return (pts / np.maximum(norms, 1e-8)).astype(np.float32)


def _build_index(
    embeddings: np.ndarray, episode_ids: list[int] | None = None
) -> EmbeddingIndex:
    embs = embeddings.astype(np.float32).copy()
    faiss.normalize_L2(embs)
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    n = len(embs)
    return EmbeddingIndex(
        index=idx,
        episode_ids=episode_ids or list(range(n)),
        frame_indices=list(range(n)),
        dimension=embs.shape[1],
        num_embeddings=n,
    )


def _make_coverage(dense_centers: list[np.ndarray]) -> CoverageMap:
    dense = [{"center": c, "density": 100.0, "size": 30} for c in dense_centers]
    return CoverageMap(
        dense_regions=dense,
        sparse_regions=[],
        overall_coverage_score=0.8,
        umap_projection=None,
    )


def _make_quality(episode_ids: list[int], score: float = 0.8) -> QualityMetrics:
    scores = {eid: score for eid in episode_ids}
    return QualityMetrics(
        episode_scores=scores,
        aggregate_score=score,
        low_quality_episodes=[],
        mutual_information_estimate=1.0,
    )


def _scorer_with_mock(center_map: dict[str, np.ndarray]) -> CapabilityScorer:
    """Create a scorer that returns controlled text embeddings."""
    scorer = CapabilityScorer(top_k=50)
    scorer._text_encoder_mode = "mock"

    rng = np.random.default_rng(99)

    def mock_encode(texts):
        out = []
        for t in texts:
            matched = False
            for keyword, vec in center_map.items():
                if keyword in t.lower():
                    out.append(vec.copy())
                    matched = True
                    break
            if not matched:
                # Random orthogonal vector
                v = rng.standard_normal(len(list(center_map.values())[0])).astype(np.float32)
                for vec in center_map.values():
                    v -= v.dot(vec) * vec
                norm = np.linalg.norm(v)
                v /= max(norm, 1e-8)
                out.append(v)
        result = np.array(out, dtype=np.float32)
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        return (result / np.maximum(norms, 1e-8)).astype(np.float32)

    scorer._encode_texts = mock_encode
    return scorer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def centers():
    rng = np.random.default_rng(42)
    c = rng.standard_normal(DIM).astype(np.float32)
    c /= np.linalg.norm(c)
    return {"positive": c, "negative": -c}


@pytest.fixture
def dataset_index(centers):
    """50 embeddings clustered around positive center."""
    embs = _make_cluster(50, DIM, centers["positive"], std=0.03, seed=1)
    ep_ids = [i // 5 for i in range(50)]  # 10 episodes of 5 frames each
    return _build_index(embs, episode_ids=ep_ids)


@pytest.fixture
def dataset_coverage(centers):
    return _make_coverage([centers["positive"]])


@pytest.fixture
def dataset_quality():
    return _make_quality(list(range(10)), score=0.8)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCapabilityScorer:
    def test_relevant_task_scores_high(
        self, centers, dataset_index, dataset_coverage, dataset_quality
    ):
        """A task matching the dataset's content should score > 0.6."""
        scorer = _scorer_with_mock({"pick": centers["positive"]})
        caps = scorer.score_tasks(
            dataset_index, dataset_coverage, dataset_quality,
            ["pick up red cube"],
        )
        assert len(caps) == 1
        print(f"Relevant task score: {caps[0].score:.3f}")
        assert caps[0].score > 0.6
        assert caps[0].gap_description is None  # score > 0.5

    def test_irrelevant_task_scores_low(
        self, centers, dataset_index, dataset_coverage, dataset_quality
    ):
        """A task unrelated to the dataset should score < 0.3."""
        scorer = _scorer_with_mock({"navigate": centers["negative"]})
        caps = scorer.score_tasks(
            dataset_index, dataset_coverage, dataset_quality,
            ["navigate outdoor terrain"],
        )
        assert len(caps) == 1
        print(f"Irrelevant task score: {caps[0].score:.3f}")
        assert caps[0].score < 0.3
        assert caps[0].gap_description is not None  # score < 0.5

    def test_diverse_dataset_broad_capability(self, centers):
        """Dataset with two clusters should score moderately for varied tasks."""
        embs = np.vstack([
            _make_cluster(30, DIM, centers["positive"], std=0.03, seed=1),
            _make_cluster(30, DIM, centers["negative"], std=0.03, seed=2),
        ])
        ep_ids = [i // 5 for i in range(60)]
        index = _build_index(embs, episode_ids=ep_ids)
        coverage = _make_coverage([centers["positive"], centers["negative"]])
        quality = _make_quality(list(range(12)), score=0.7)

        scorer = _scorer_with_mock({
            "pick": centers["positive"],
            "navigate": centers["negative"],
        })

        caps = scorer.score_tasks(
            index, coverage, quality,
            ["pick up objects", "navigate terrain", "unrelated task"],
        )
        print(f"Pick score: {caps[0].score:.3f}, Navigate score: {caps[1].score:.3f}, Unrelated: {caps[2].score:.3f}")
        # Both matching tasks should score well (above threshold)
        assert caps[0].score > 0.3
        assert caps[1].score > 0.3
        # Unrelated should be significantly lower
        assert caps[2].score < min(caps[0].score, caps[1].score)

    def test_failure_zone_prediction(self, centers, dataset_index, dataset_coverage, dataset_quality):
        """Deployment embeddings far from training should be predicted failures."""
        profile = DatasetProfile(
            dataset_name="test",
            num_episodes=10,
            num_frames=50,
            embedding_index=dataset_index,
            coverage=dataset_coverage,
            capabilities=[],
            quality=dataset_quality,
        )

        # Target embeddings in the opposite region
        target_embs = _make_cluster(20, DIM, centers["negative"], std=0.03, seed=5)
        scorer = CapabilityScorer(top_k=50)
        zones = scorer.predict_failure_zones(profile, target_embs)

        print(f"Failure zones found: {len(zones)}")
        assert len(zones) > 0
        total_failures = sum(z["num_frames"] for z in zones)
        print(f"Total predicted failures: {total_failures} / 20")
        assert total_failures > 10  # most should be failures

    def test_compare_profiles_identical(self, centers, dataset_index, dataset_coverage, dataset_quality):
        """Comparing a profile to itself should show high overlap."""
        cap = CapabilityScore(
            task_description="pick up cube",
            score=0.8, confidence=0.9,
            supporting_episodes=10,
            action_diversity=0.5,
            environment_diversity=0.1,
            gap_description=None,
        )
        profile = DatasetProfile(
            dataset_name="test",
            num_episodes=10,
            num_frames=50,
            embedding_index=dataset_index,
            coverage=dataset_coverage,
            capabilities=[cap],
            quality=dataset_quality,
        )

        scorer = CapabilityScorer(top_k=50)
        result = scorer.compare_profiles(profile, profile)

        print(f"Self-overlap: {result['overlap']:.3f}")
        assert result["overlap"] > 0.9
        assert len(result["unique_to_a"]) == 0
        assert len(result["unique_to_b"]) == 0
