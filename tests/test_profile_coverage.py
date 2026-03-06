"""Tests for orbit.profile.coverage (CoverageAnalyzer)."""

from __future__ import annotations

import faiss
import numpy as np
import pytest

from orbit.profile.coverage import CoverageAnalyzer
from orbit.profile.types import EmbeddingIndex


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cluster(
    n: int, dim: int, center: np.ndarray, std: float = 0.05, seed: int = 0
) -> np.ndarray:
    """Generate L2-normalized embeddings clustered around *center*."""
    rng = np.random.default_rng(seed)
    pts = center + rng.standard_normal((n, dim)).astype(np.float32) * std
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return (pts / np.maximum(norms, 1e-8)).astype(np.float32)


def _build_index(embeddings: np.ndarray) -> EmbeddingIndex:
    """Build an EmbeddingIndex from raw embeddings."""
    embs = embeddings.astype(np.float32).copy()
    faiss.normalize_L2(embs)
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    return EmbeddingIndex(
        index=idx,
        episode_ids=list(range(len(embs))),
        frame_indices=list(range(len(embs))),
        dimension=embs.shape[1],
        num_embeddings=len(embs),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCoverageAnalyzer:
    def test_two_clusters(self):
        """Two well-separated clusters should produce ≥ 2 dense regions."""
        dim = 64
        rng = np.random.default_rng(0)
        c1 = rng.standard_normal(dim).astype(np.float32)
        c2 = -c1  # opposite direction

        embs = np.vstack([
            _make_cluster(30, dim, c1, std=0.02, seed=1),
            _make_cluster(30, dim, c2, std=0.02, seed=2),
        ])
        ei = _build_index(embs)

        analyzer = CoverageAnalyzer(n_clusters=10, min_cluster_size=5)
        cmap = analyzer.analyze(ei)

        total_regions = len(cmap.dense_regions) + len(cmap.sparse_regions)
        assert total_regions >= 2
        assert 0.0 <= cmap.overall_coverage_score <= 1.0

    def test_coverage_score_range(self):
        """Dense uniform data should score higher than sparse scattered data."""
        dim = 32
        rng = np.random.default_rng(42)

        # Dense: tight cluster
        dense_center = rng.standard_normal(dim).astype(np.float32)
        dense_embs = _make_cluster(50, dim, dense_center, std=0.01, seed=10)
        dense_ei = _build_index(dense_embs)

        # Sparse: widely scattered
        sparse_embs = rng.standard_normal((50, dim)).astype(np.float32)
        norms = np.linalg.norm(sparse_embs, axis=1, keepdims=True)
        sparse_embs = (sparse_embs / np.maximum(norms, 1e-8)).astype(np.float32)
        sparse_ei = _build_index(sparse_embs)

        analyzer = CoverageAnalyzer(min_cluster_size=3)
        score_dense = analyzer.analyze(dense_ei).overall_coverage_score
        score_sparse = analyzer.analyze(sparse_ei).overall_coverage_score

        # Dense should score higher (points are closer together → higher similarity)
        assert score_dense > score_sparse

    def test_find_gaps(self):
        """Dataset A covers region X; dataset B covers X+Y. Gaps in Y."""
        dim = 64
        rng = np.random.default_rng(0)
        center_x = rng.standard_normal(dim).astype(np.float32)
        center_y = -center_x

        embs_a = _make_cluster(30, dim, center_x, std=0.02, seed=1)
        embs_b = np.vstack([
            _make_cluster(30, dim, center_x, std=0.02, seed=1),
            _make_cluster(30, dim, center_y, std=0.02, seed=2),
        ])

        idx_a = _build_index(embs_a)
        idx_b = _build_index(embs_b)

        analyzer = CoverageAnalyzer(min_cluster_size=3)
        gaps = analyzer.find_gaps(idx_a, reference_index=idx_b)

        assert len(gaps) > 0
        # Gaps should come from region Y (high gap score)
        assert all("gap_score" in g for g in gaps)
        assert gaps[0]["gap_score"] > 0.3

    def test_overlap_identical(self):
        """Overlap of a dataset with itself should be ~1.0."""
        dim = 32
        embs = _make_cluster(30, dim, np.ones(dim, dtype=np.float32), std=0.05)
        ei = _build_index(embs)

        analyzer = CoverageAnalyzer()
        overlap = analyzer.compute_overlap(ei, ei)
        assert overlap > 0.9

    def test_overlap_disjoint(self):
        """Non-overlapping datasets should have low overlap."""
        dim = 64
        rng = np.random.default_rng(0)
        c1 = rng.standard_normal(dim).astype(np.float32)
        c2 = -c1

        ei_a = _build_index(_make_cluster(30, dim, c1, std=0.01, seed=1))
        ei_b = _build_index(_make_cluster(30, dim, c2, std=0.01, seed=2))

        analyzer = CoverageAnalyzer()
        overlap = analyzer.compute_overlap(ei_a, ei_b)
        assert overlap < 0.3

    def test_small_dataset(self):
        """3 embeddings should work without crashing."""
        dim = 32
        rng = np.random.default_rng(99)
        embs = rng.standard_normal((3, dim)).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = (embs / np.maximum(norms, 1e-8)).astype(np.float32)
        ei = _build_index(embs)

        analyzer = CoverageAnalyzer(min_cluster_size=2)
        cmap = analyzer.analyze(ei)

        assert 0.0 <= cmap.overall_coverage_score <= 1.0
