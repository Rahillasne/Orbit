"""Tests for the new EmbeddingAnalyzer (orbit.analyzer.embedding_analyzer)."""

from __future__ import annotations

import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest
from PIL import Image

from orbit.analyzer.embedding_analyzer import EmbeddingAnalyzer
from orbit.analyzer.models import (
    AnalysisReport,
    EmbeddingAnalyzerConfig,
    EpisodeGapSummary,
    FailureClusterReport,
    FrameGapResult,
)
from orbit.logger.schemas import Episode, EpisodeFrame, Outcome

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_images(n: int, seed: int = 42) -> list[Image.Image]:
    """Generate synthetic test images."""
    rng = np.random.default_rng(seed)
    return [Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)) for _ in range(n)]


def _make_synthetic_embeddings(
    n: int,
    dim: int = 768,
    center: np.ndarray | None = None,
    std: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic L2-normalized embeddings around a center."""
    rng = np.random.default_rng(seed)
    if center is None:
        center = rng.standard_normal(dim).astype(np.float32)
    embs = center + rng.standard_normal((n, dim)).astype(np.float32) * std
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    normalized = (embs / np.maximum(norms, 1e-8)).astype(np.float32)
    return np.ascontiguousarray(normalized)


def _make_episode_with_images(
    tmp_path: Path,
    n_frames: int,
    outcome: Outcome,
    prefix: str = "ep",
) -> Episode:
    """Create an Episode with actual image files on disk."""
    frames = []
    img_dir = tmp_path / "images" / prefix
    img_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img_path = img_dir / f"frame_{i:04d}.png"
        img.save(img_path)
        frames.append(
            EpisodeFrame(
                timestamp=1000.0 + i * 0.033,
                joint_positions=[0.1] * 6,
                gripper_state=0.5,
                image_path=str(img_path),
                action=[0.01] * 6,
                reward=0.1,
            )
        )
    return Episode(
        task_name="test_task",
        frames=frames,
        outcome=outcome,
        start_time=datetime.datetime(2025, 1, 1),
        end_time=datetime.datetime(2025, 1, 1, 0, 1, 0),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_siglip():
    """Patch transformers to return a fake SigLIP model (768-dim)."""
    import torch

    mock_model = MagicMock()
    mock_processor = MagicMock()

    def fake_process(images, return_tensors="pt"):
        n = len(images) if isinstance(images, list) else 1
        return {"pixel_values": torch.randn(n, 3, 224, 224)}

    mock_processor.side_effect = fake_process

    def fake_get_image_features(**kwargs):
        n = kwargs["pixel_values"].shape[0]
        return torch.randn(n, 768)

    mock_model.get_image_features = MagicMock(side_effect=fake_get_image_features)
    mock_model.eval.return_value = mock_model
    mock_model.to.return_value = mock_model

    with (
        patch("transformers.AutoModel") as mock_auto_model,
        patch("transformers.AutoProcessor") as mock_auto_proc,
    ):
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_auto_proc.from_pretrained.return_value = mock_processor
        yield mock_model, mock_processor


@pytest.fixture
def analyzer(mock_siglip):
    """Create an EmbeddingAnalyzer with mocked SigLIP."""
    return EmbeddingAnalyzer(EmbeddingAnalyzerConfig(batch_size=8))


# ---------------------------------------------------------------------------
# Embedding tests
# ---------------------------------------------------------------------------


class TestEmbedding:
    """Tests for embedding computation."""

    def test_embed_images_shape(self, analyzer):
        images = _make_test_images(10)
        embeddings = analyzer.embed_images(images)
        assert embeddings.shape == (10, 768)

    def test_embeddings_are_normalized(self, analyzer):
        images = _make_test_images(5)
        embeddings = analyzer.embed_images(images)
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_batch_processing_count(self, analyzer):
        """Batching produces correct result count regardless of batch size."""
        analyzer.config.batch_size = 3
        images = _make_test_images(10)
        embeddings = analyzer.embed_images(images)
        assert embeddings.shape[0] == 10

    def test_single_image(self, analyzer):
        images = _make_test_images(1)
        embeddings = analyzer.embed_images(images)
        assert embeddings.shape == (1, 768)


# ---------------------------------------------------------------------------
# FAISS index tests
# ---------------------------------------------------------------------------


class TestFAISSIndex:
    """Tests for FAISS index building and querying."""

    def test_build_index_flat_ip(self, analyzer):
        embs = _make_synthetic_embeddings(100, dim=768, seed=0)
        analyzer._build_faiss_index(embs)
        assert analyzer._faiss_index is not None
        assert analyzer._faiss_index.ntotal == 100

    def test_search_returns_correct_shape(self, analyzer):
        training = _make_synthetic_embeddings(100, dim=768, seed=0)
        analyzer._build_faiss_index(training)
        query = _make_synthetic_embeddings(10, dim=768, seed=1)
        dists, indices = analyzer._faiss_index.search(query, 5)
        assert dists.shape == (10, 5)
        assert indices.shape == (10, 5)

    def test_cosine_similarity_self_query(self, analyzer):
        """Querying identical vectors should yield cosine similarity ~1.0."""
        training = _make_synthetic_embeddings(50, dim=768, seed=0)
        analyzer._build_faiss_index(training)
        D, _ = analyzer._faiss_index.search(training[:5], 1)
        assert np.all(D > 0.95)

    def test_index_is_inner_product(self, analyzer):
        """Verify IndexFlatIP is used (not L2)."""
        embs = _make_synthetic_embeddings(10, dim=768)
        analyzer._build_faiss_index(embs)
        D, _ = analyzer._faiss_index.search(embs[:1], 1)
        assert D[0, 0] > 0.0


# ---------------------------------------------------------------------------
# Gap score tests
# ---------------------------------------------------------------------------


class TestGapScores:
    """Tests for gap score computation accuracy."""

    def test_near_training_has_low_gap(self, analyzer):
        """Points near training distribution should have low gap scores."""
        center = np.zeros(768, dtype=np.float32)
        center[0] = 1.0
        # Use small std so noise doesn't overwhelm the center direction
        training = _make_synthetic_embeddings(100, dim=768, center=center, std=0.01, seed=0)
        analyzer._training_embeddings = training
        analyzer._build_faiss_index(training)

        near_embs = _make_synthetic_embeddings(10, dim=768, center=center, std=0.01, seed=99)
        D, _ = analyzer._faiss_index.search(near_embs, analyzer.config.num_neighbors)
        gap_scores = 1.0 - D.mean(axis=1)
        assert np.all(gap_scores < 0.15)

    def test_far_from_training_has_high_gap(self, analyzer):
        """Points orthogonal to training should have high gap scores."""
        center1 = np.zeros(768, dtype=np.float32)
        center1[0] = 1.0
        training = _make_synthetic_embeddings(100, dim=768, center=center1, std=0.01, seed=0)
        analyzer._training_embeddings = training
        analyzer._build_faiss_index(training)

        center2 = np.zeros(768, dtype=np.float32)
        center2[1] = 1.0
        far_embs = _make_synthetic_embeddings(10, dim=768, center=center2, std=0.01, seed=99)
        D, _ = analyzer._faiss_index.search(far_embs, analyzer.config.num_neighbors)
        gap_scores = 1.0 - D.mean(axis=1)
        assert np.all(gap_scores > 0.5)

    def test_episode_aggregation_statistics(self, analyzer):
        """Verify mean, max, p95 are computed correctly."""
        gap_trajectory = [0.1, 0.2, 0.3, 0.8, 0.05]
        eid = uuid4()
        frame_results = [
            FrameGapResult(
                frame_idx=i,
                episode_id=eid,
                gap_score=g,
                nearest_distances=[g],
                nearest_indices=[0],
            )
            for i, g in enumerate(gap_trajectory)
        ]
        summary = analyzer._aggregate_episode(eid, Outcome.FAILURE, frame_results)
        assert abs(summary.mean_gap - float(np.mean(gap_trajectory))) < 1e-6
        assert abs(summary.max_gap - 0.8) < 1e-6
        assert summary.gap_percentile_95 >= 0.3
        assert summary.num_frames == 5
        assert summary.gap_trajectory == gap_trajectory

    def test_compute_gap_scores_with_episodes(self, analyzer, tmp_path):
        """End-to-end gap score computation with image files."""
        training = _make_synthetic_embeddings(50, dim=768, seed=0)
        analyzer._training_embeddings = training
        analyzer._build_faiss_index(training)

        ep = _make_episode_with_images(tmp_path, 5, Outcome.SUCCESS, prefix="test")
        frame_results, episode_summaries = analyzer.compute_gap_scores([ep])
        assert len(frame_results) == 5
        assert len(episode_summaries) == 1
        assert episode_summaries[0].num_frames == 5
        assert all(0.0 <= fr.gap_score <= 2.0 for fr in frame_results)

    def test_gap_scores_without_index_raises(self, analyzer):
        """Calling compute_gap_scores before indexing should raise."""
        with pytest.raises(RuntimeError, match="Training data not indexed"):
            analyzer.compute_gap_scores([])


# ---------------------------------------------------------------------------
# Caching tests
# ---------------------------------------------------------------------------


class TestCaching:
    """Tests for embedding caching to disk."""

    def test_cache_save_and_load(self, analyzer, tmp_path):
        analyzer.config.cache_dir = str(tmp_path / "cache")
        embs = _make_synthetic_embeddings(50, dim=768)
        cache_key = analyzer._cache_key("/some/dataset/path")
        analyzer._save_cached_embeddings(cache_key, embs)
        loaded = analyzer._load_cached_embeddings(cache_key)
        assert loaded is not None
        np.testing.assert_array_equal(embs, loaded)

    def test_cache_miss_returns_none(self, analyzer, tmp_path):
        analyzer.config.cache_dir = str(tmp_path / "cache")
        result = analyzer._load_cached_embeddings("nonexistent_key")
        assert result is None

    def test_cache_key_deterministic(self, analyzer):
        key1 = analyzer._cache_key("/path/to/dataset")
        key2 = analyzer._cache_key("/path/to/dataset")
        assert key1 == key2

    def test_different_model_different_cache(self, analyzer):
        key1 = analyzer._cache_key("/path/to/dataset")
        analyzer.config.model_name = "other-model"
        key2 = analyzer._cache_key("/path/to/dataset")
        assert key1 != key2

    def test_index_training_uses_cache(self, analyzer, tmp_path):
        """Second call to index_training_data loads from cache."""
        analyzer.config.cache_dir = str(tmp_path / "cache")
        img_dir = tmp_path / "train_imgs"
        img_dir.mkdir()
        for i in range(5):
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            img.save(img_dir / f"img_{i}.png")

        n1 = analyzer.index_training_data(img_dir)
        assert n1 == 5

        # Second call should hit cache
        analyzer._faiss_index = None
        analyzer._training_embeddings = None
        n2 = analyzer.index_training_data(img_dir)
        assert n2 == 5
        assert analyzer._faiss_index is not None


# ---------------------------------------------------------------------------
# Failure clustering tests
# ---------------------------------------------------------------------------


class TestFailureClustering:
    """Tests for HDBSCAN failure clustering."""

    def test_clusters_separable_data(self):
        """Three well-separated clusters should produce multiple clusters."""
        import hdbscan

        dim = 768
        centers = [np.zeros(dim, dtype=np.float32) for _ in range(3)]
        centers[0][0] = 1.0
        centers[1][1] = 1.0
        centers[2][2] = 1.0

        all_embs = []
        for i, c in enumerate(centers):
            cluster_embs = _make_synthetic_embeddings(30, dim=dim, center=c, std=0.01, seed=i * 100)
            all_embs.append(cluster_embs)
        failure_embs = np.vstack(all_embs)

        clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
        labels = clusterer.fit_predict(failure_embs)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        assert n_clusters >= 2

    def test_cluster_report_structure(self):
        """Verify FailureClusterReport has correct structure."""
        report = FailureClusterReport(
            clusters=[],
            num_failure_frames=0,
            num_noise_frames=0,
            num_clusters=0,
        )
        assert report.num_clusters == 0
        assert isinstance(report.clusters, list)

    def test_empty_failures_returns_empty_report(self, analyzer):
        """No failure episodes -> empty cluster report."""
        report = analyzer.cluster_failures([])
        assert report.num_clusters == 0
        assert report.num_failure_frames == 0

    def test_cluster_failures_with_episodes(self, analyzer, tmp_path):
        """Cluster failures with actual episode data."""
        training = _make_synthetic_embeddings(50, dim=768, seed=0)
        analyzer._training_embeddings = training
        analyzer._build_faiss_index(training)

        failure_eps = []
        for i in range(3):
            ep = _make_episode_with_images(tmp_path, 10, Outcome.FAILURE, prefix=f"fail_{i}")
            failure_eps.append(ep)

        report = analyzer.cluster_failures(failure_eps)
        assert isinstance(report, FailureClusterReport)
        assert report.num_failure_frames == 30


# ---------------------------------------------------------------------------
# Visualization tests
# ---------------------------------------------------------------------------


class TestVisualization:
    """Tests for UMAP + Plotly visualization generation."""

    def test_umap_fit_transform_separate(self, analyzer):
        """UMAP fitted on training should transform deployment data."""
        training = _make_synthetic_embeddings(100, dim=768, seed=0)
        deployment = _make_synthetic_embeddings(20, dim=768, seed=1)

        analyzer._fit_umap(training)
        assert analyzer._umap_reducer is not None

        train_2d = analyzer._umap_reducer.transform(training)
        deploy_2d = analyzer._umap_reducer.transform(deployment)
        assert train_2d.shape == (100, 2)
        assert deploy_2d.shape == (20, 2)

    def test_gap_heatmap_generates_html(self, analyzer, tmp_path):
        """Gap heatmap should produce an HTML file with plotly content."""
        summaries = [
            EpisodeGapSummary(
                episode_id=uuid4(),
                outcome="failure",
                mean_gap=0.5 + i * 0.1,
                max_gap=0.8,
                gap_percentile_95=0.7,
                gap_trajectory=[0.3, 0.5, 0.8, 0.4],
                num_frames=4,
            )
            for i in range(5)
        ]
        heatmap_path = tmp_path / "gap_heatmap.html"
        analyzer._generate_gap_heatmap(summaries, heatmap_path)
        assert heatmap_path.exists()
        content = heatmap_path.read_text()
        assert "plotly" in content.lower()
        assert "Gap Score" in content

    def test_umap_scatter_generates_html(self, analyzer, tmp_path):
        """UMAP scatter should produce an HTML file."""
        train_2d = np.random.randn(50, 2).astype(np.float32)
        deploy_2d = np.random.randn(10, 2).astype(np.float32)

        scatter_path = tmp_path / "scatter.html"
        analyzer._generate_umap_scatter(
            train_2d,
            deploy_2d,
            episode_ids=["ep1"] * 5 + ["ep2"] * 5,
            frame_indices=list(range(10)),
            gap_scores=[0.1 * i for i in range(10)],
            outcomes=["success"] * 5 + ["failure"] * 5,
            output_path=scatter_path,
        )
        assert scatter_path.exists()
        content = scatter_path.read_text()
        assert "plotly" in content.lower()
        assert "UMAP" in content


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline_synthetic(self, analyzer, tmp_path):
        """Full pipeline: index -> gap scores -> cluster -> viz."""
        training_dir = tmp_path / "training"
        training_dir.mkdir()
        for i in range(20):
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            img.save(training_dir / f"train_{i:04d}.png")

        success_ep = _make_episode_with_images(tmp_path, 10, Outcome.SUCCESS, prefix="success")
        failure_ep = _make_episode_with_images(tmp_path, 10, Outcome.FAILURE, prefix="failure")

        output_dir = tmp_path / "viz_output"
        report = analyzer.analyze(
            training_source=training_dir,
            deployment_episodes=[success_ep, failure_ep],
            output_dir=output_dir,
            generate_viz=True,
        )

        assert isinstance(report, AnalysisReport)
        assert len(report.episode_summaries) == 2
        assert report.training_embedding_count == 20
        assert report.deployment_embedding_count == 20
        assert report.cluster_report is not None

        for name, path in report.visualization_paths.items():
            assert Path(path).exists()
            assert Path(path).read_text()

    def test_analyze_with_image_list(self, analyzer, tmp_path):
        """Training from direct image list instead of directory."""
        training_images = _make_test_images(15)
        ep = _make_episode_with_images(tmp_path, 5, Outcome.SUCCESS, prefix="ep")

        report = analyzer.analyze(
            training_source=training_images,
            deployment_episodes=[ep],
            generate_viz=False,
        )
        assert report.training_embedding_count == 15
        assert len(report.episode_summaries) == 1

    def test_analyze_no_failures_skips_clustering(self, analyzer, tmp_path):
        """When all episodes succeed, cluster_report should be None."""
        training_dir = tmp_path / "training"
        training_dir.mkdir()
        for i in range(10):
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            img.save(training_dir / f"train_{i:04d}.png")

        success_ep = _make_episode_with_images(tmp_path, 5, Outcome.SUCCESS, prefix="s")
        report = analyzer.analyze(
            training_source=training_dir,
            deployment_episodes=[success_ep],
            generate_viz=False,
        )
        assert report.cluster_report is None


# ---------------------------------------------------------------------------
# Performance test
# ---------------------------------------------------------------------------


class TestPerformance:
    """Performance profiling tests."""

    def test_embedding_throughput(self, analyzer):
        """100 images should embed quickly with mocked model."""
        import time

        images = _make_test_images(100)
        start = time.perf_counter()
        embeddings = analyzer.embed_images(images)
        elapsed = time.perf_counter() - start
        assert embeddings.shape[0] == 100
        assert elapsed < 10.0
