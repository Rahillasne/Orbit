"""Tests for orbit.profile.feature_extractor."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from orbit.profile.types import (
    ActionStats,
    CapabilityScore,
    CoverageMap,
    DatasetProfile,
    DatasetReportCard,
    EmbeddingIndex,
    EmbeddingStats,
    QualityMetrics,
    QualitySignalBreakdown,
    ScoreBreakdown,
    TaskAssessment,
)

# ======================================================================
# Helpers
# ======================================================================


def _make_faiss_index(n: int = 100, dim: int = 16, seed: int = 42):
    """Build a FAISS IndexFlatIP with L2-normalized random embeddings."""
    import faiss

    rng = np.random.default_rng(seed)
    embeddings = rng.standard_normal((n, dim)).astype(np.float32)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, embeddings


def _make_test_profile(
    n_episodes: int = 5,
    n_frames: int = 100,
    dim: int = 16,
    seed: int = 42,
    with_capabilities: bool = True,
) -> DatasetProfile:
    """Create a minimal DatasetProfile with realistic synthetic data."""
    index, _ = _make_faiss_index(n_frames, dim, seed)

    rng = np.random.default_rng(seed)

    # Distribute frames across episodes
    ep_ids = sorted(rng.integers(0, n_episodes, size=n_frames).tolist())
    frame_ids = []
    counters: dict[int, int] = {}
    for eid in ep_ids:
        counters[eid] = counters.get(eid, 0) + 1
        frame_ids.append(counters[eid] - 1)

    embedding_index = EmbeddingIndex(
        index=index,
        episode_ids=ep_ids,
        frame_indices=frame_ids,
        dimension=dim,
        num_embeddings=n_frames,
    )

    coverage = CoverageMap(
        dense_regions=[
            {
                "center": rng.standard_normal(dim).astype(np.float32),
                "density": 0.8,
                "size": 30,
                "description": "cluster A",
            },
        ],
        sparse_regions=[
            {
                "center": rng.standard_normal(dim).astype(np.float32),
                "density": 0.2,
                "size": 10,
                "description": "sparse B",
            },
        ],
        overall_coverage_score=0.65,
        umap_projection=None,
    )

    quality = QualityMetrics(
        episode_scores={i: 0.5 + 0.1 * i for i in range(n_episodes)},
        aggregate_score=0.7,
        low_quality_episodes=[0],
        mutual_information_estimate=0.3,
        signal_breakdown=QualitySignalBreakdown(
            mutual_information=0.6,
            action_smoothness=0.75,
            episode_completion=0.5,
            observation_consistency=0.9,
            demonstration_quality=0.65,
        ),
    )

    capabilities = []
    if with_capabilities:
        capabilities = [
            CapabilityScore(
                task_description="pick up cube",
                score=0.72,
                confidence=0.8,
                supporting_episodes=40,
                action_diversity=0.6,
                environment_diversity=0.3,
                gap_description=None,
                score_breakdown=ScoreBreakdown(
                    visual_relevance=0.8,
                    data_quality=0.7,
                    coverage_diversity=0.6,
                    volume=0.5,
                ),
            ),
        ]

    return DatasetProfile(
        dataset_name="test_dataset",
        num_episodes=n_episodes,
        num_frames=n_frames,
        embedding_index=embedding_index,
        coverage=coverage,
        capabilities=capabilities,
        quality=quality,
    )


def _make_test_report_card() -> DatasetReportCard:
    return DatasetReportCard(
        dataset_name="test_dataset",
        overall_grade="B",
        overall_score=0.72,
        coverage_grade="B",
        coverage_score=0.71,
        quality_grade="B",
        quality_score=0.75,
        diversity_grade="C",
        diversity_score=0.55,
        volume_grade="B",
        volume_score=0.70,
        strengths=["Good quality"],
        weaknesses=["Low diversity"],
        gaps=[],
        prescriptions=[],
        task_assessments=[
            TaskAssessment(
                task="pick up cube",
                grade="B",
                score=0.72,
                relevance="High",
                coverage="Partially Covered",
                confidence="High",
                finding="Decent coverage for cube picking.",
            ),
        ],
    )


def _make_test_episodes(n_episodes: int = 5, T: int = 20, action_dim: int = 6, seed: int = 42):
    """Create synthetic episode dicts with states and actions."""
    rng = np.random.default_rng(seed)
    episodes = []
    for i in range(n_episodes):
        states = np.cumsum(rng.standard_normal((T, action_dim)) * 0.1, axis=0)
        actions = np.diff(states, axis=0, prepend=states[:1]) + rng.standard_normal(
            (T, action_dim)
        ) * 0.01
        episodes.append({"episode_id": i, "states": states, "actions": actions})
    return episodes


# ======================================================================
# Tests — DatasetFeatureExtractor
# ======================================================================


class TestDatasetFeatureExtractor:
    def test_extract_returns_64_dims(self):
        from orbit.profile.feature_extractor import DatasetFeatureExtractor

        ext = DatasetFeatureExtractor()
        profile = _make_test_profile()
        vec = ext.extract(profile)
        assert vec.shape == (64,)
        assert vec.dtype == np.float32

    def test_extract_with_report_card(self):
        from orbit.profile.feature_extractor import DatasetFeatureExtractor

        ext = DatasetFeatureExtractor()
        profile = _make_test_profile()
        rc = _make_test_report_card()
        vec = ext.extract(profile, report_card=rc)
        assert vec.shape == (64,)
        # Task features (indices 52-55) should be non-zero with report card
        assert vec[52] > 0  # primary task score

    def test_extract_with_episodes(self):
        from orbit.profile.feature_extractor import DatasetFeatureExtractor

        ext = DatasetFeatureExtractor()
        profile = _make_test_profile()
        episodes = _make_test_episodes()
        vec = ext.extract(profile, episodes=episodes)
        assert vec.shape == (64,)
        # Action dimensionality (index 20) should be non-zero
        assert vec[20] > 0

    def test_extract_with_metadata(self):
        from orbit.profile.feature_extractor import DatasetFeatureExtractor

        ext = DatasetFeatureExtractor()
        profile = _make_test_profile()
        metadata = {
            "fps": 30.0,
            "image_resolution_pixels": 640 * 480,
            "observation_dims": 24,
            "dataset_size_mb": 500.0,
            "frame_brightness_mean": 0.5,
            "reward_signal_present": 1,
        }
        vec = ext.extract(profile, metadata=metadata)
        assert vec.shape == (64,)
        # Scale features should reflect metadata
        assert vec[48] == pytest.approx(30.0, abs=1e-5)  # fps

    def test_extract_missing_everything(self):
        """No report_card, no episodes, no metadata — still 64 dims."""
        from orbit.profile.feature_extractor import DatasetFeatureExtractor

        ext = DatasetFeatureExtractor()
        profile = _make_test_profile(with_capabilities=False)
        vec = ext.extract(profile)
        assert vec.shape == (64,)

    def test_all_features_finite(self):
        from orbit.profile.feature_extractor import DatasetFeatureExtractor

        ext = DatasetFeatureExtractor()
        profile = _make_test_profile()
        episodes = _make_test_episodes()
        rc = _make_test_report_card()
        vec = ext.extract(profile, report_card=rc, episodes=episodes)
        assert np.all(np.isfinite(vec))

    def test_deterministic(self):
        from orbit.profile.feature_extractor import DatasetFeatureExtractor

        ext = DatasetFeatureExtractor()
        profile = _make_test_profile()
        v1 = ext.extract(profile)
        # Clear cached stats to force recomputation
        profile.embedding_stats = None
        profile.action_stats = None
        v2 = ext.extract(profile)
        np.testing.assert_array_equal(v1, v2)

    def test_feature_names(self):
        from orbit.profile.feature_extractor import DatasetFeatureExtractor

        assert len(DatasetFeatureExtractor.FEATURE_NAMES) == 64

    def test_batch_extraction(self):
        from orbit.profile.feature_extractor import DatasetFeatureExtractor

        ext = DatasetFeatureExtractor()
        profiles = [_make_test_profile(seed=i) for i in range(3)]
        batch = ext.extract_batch(profiles)
        assert batch.shape == (3, 64)
        assert batch.dtype == np.float32

    def test_embedding_stats_cached(self):
        from orbit.profile.feature_extractor import DatasetFeatureExtractor

        ext = DatasetFeatureExtractor()
        profile = _make_test_profile()
        assert profile.embedding_stats is None
        ext.extract(profile)
        assert profile.embedding_stats is not None
        # Second call should reuse cached
        cached = profile.embedding_stats
        ext.extract(profile)
        assert profile.embedding_stats is cached

    def test_observation_only_dataset(self):
        """No episodes provided — action features should all be 0 except smoothness fallback."""
        from orbit.profile.feature_extractor import DatasetFeatureExtractor

        ext = DatasetFeatureExtractor()
        profile = _make_test_profile()
        vec = ext.extract(profile)
        # Action features are indices 20-31
        # dimensionality (20) should be 0 when no episodes
        assert vec[20] == 0.0
        # smoothness (23) falls back to signal_breakdown.action_smoothness = 0.75
        assert vec[23] == pytest.approx(0.75, abs=1e-5)

    def test_small_profile(self):
        """Profile with very few embeddings doesn't crash."""
        from orbit.profile.feature_extractor import DatasetFeatureExtractor

        ext = DatasetFeatureExtractor()
        profile = _make_test_profile(n_episodes=1, n_frames=3, dim=4)
        vec = ext.extract(profile)
        assert vec.shape == (64,)
        assert np.all(np.isfinite(vec))


# ======================================================================
# Tests — FeatureScaler
# ======================================================================


class TestFeatureScaler:
    def test_fit_transform(self):
        from orbit.profile.feature_extractor import FeatureScaler

        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 64)).astype(np.float32) * 10 + 5
        scaler = FeatureScaler()
        X_t = scaler.fit_transform(X)
        # After standardisation, mean ≈ 0, std ≈ 1
        assert np.abs(X_t.mean(axis=0)).max() < 0.5
        assert X_t.dtype == np.float32

    def test_save_load_roundtrip(self):
        from orbit.profile.feature_extractor import FeatureScaler

        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 64)).astype(np.float32)
        scaler = FeatureScaler()
        scaler.fit(X)
        X_t1 = scaler.transform(X)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        scaler.save(path)
        loaded = FeatureScaler.load(path)
        X_t2 = loaded.transform(X)
        np.testing.assert_allclose(X_t1, X_t2, atol=1e-5)

    def test_unfitted_raises(self):
        from orbit.profile.feature_extractor import FeatureScaler

        scaler = FeatureScaler()
        X = np.zeros((5, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="not been fitted"):
            scaler.transform(X)

    def test_save_unfitted_raises(self):
        from orbit.profile.feature_extractor import FeatureScaler

        scaler = FeatureScaler()
        with pytest.raises(ValueError, match="unfitted"):
            scaler.save("/tmp/bad.json")

    def test_scaler_integration_with_extractor(self):
        """FeatureScaler works end-to-end with DatasetFeatureExtractor."""
        from orbit.profile.feature_extractor import DatasetFeatureExtractor, FeatureScaler

        profiles = [_make_test_profile(seed=i) for i in range(10)]
        ext_raw = DatasetFeatureExtractor()
        raw = ext_raw.extract_batch(profiles)

        scaler = FeatureScaler()
        scaler.fit(raw)

        ext_scaled = DatasetFeatureExtractor(scaler=scaler)
        scaled = ext_scaled.extract(profiles[0])
        assert scaled.shape == (64,)
        assert np.all(np.isfinite(scaled))


# ======================================================================
# Tests — Stat computation edge cases
# ======================================================================


class TestEmbeddingStatsComputation:
    def test_compute_from_faiss(self):
        from orbit.profile.feature_extractor import _compute_embedding_stats

        profile = _make_test_profile(n_frames=50, dim=16)
        stats = _compute_embedding_stats(profile)
        assert isinstance(stats, EmbeddingStats)
        assert stats.mean_norm > 0
        assert 0 <= stats.noise_ratio <= 1

    def test_empty_index(self):
        import faiss

        from orbit.profile.feature_extractor import _compute_embedding_stats

        index = faiss.IndexFlatIP(8)
        profile = DatasetProfile(
            dataset_name="empty",
            num_episodes=0,
            num_frames=0,
            embedding_index=EmbeddingIndex(
                index=index, episode_ids=[], frame_indices=[], dimension=8, num_embeddings=0
            ),
            coverage=CoverageMap([], [], 0.0, None),
            capabilities=[],
            quality=QualityMetrics({}, 0.0, [], 0.0),
        )
        stats = _compute_embedding_stats(profile)
        assert stats.mean_norm == 0.0


class TestActionStatsComputation:
    def test_compute_from_episodes(self):
        from orbit.profile.feature_extractor import _compute_action_stats

        episodes = _make_test_episodes()
        stats = _compute_action_stats(episodes)
        assert isinstance(stats, ActionStats)
        assert stats.dimensionality == 6.0
        assert stats.mean_magnitude > 0

    def test_empty_episodes(self):
        from orbit.profile.feature_extractor import _compute_action_stats

        stats = _compute_action_stats([])
        assert stats.dimensionality == 0.0
        assert stats.mean_magnitude == 0.0

    def test_single_short_episode(self):
        from orbit.profile.feature_extractor import _compute_action_stats

        ep = [{"episode_id": 0, "states": np.zeros((1, 4)), "actions": np.zeros((1, 4))}]
        stats = _compute_action_stats(ep)
        # Single timestep → skipped, returns defaults
        assert stats.dimensionality == 0.0
