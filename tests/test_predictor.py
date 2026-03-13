"""Tests for orbit.profile.predictor (DatasetQualityModel)."""

from __future__ import annotations

import numpy as np

from orbit.profile.predictor import DatasetQualityModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FEATURE_DIM = 64


def _make_synthetic_data(n: int = 30, seed: int = 42):
    """Generate synthetic (features, success_rates) for testing."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, FEATURE_DIM)).astype(np.float32)
    # Success rate is a noisy linear combination of first few features
    y = np.clip(X[:, 0] * 0.4 + X[:, 1] * 0.3 + rng.normal(0, 0.1, n), 0, 1)
    return X, y.astype(np.float32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDatasetQualityModel:
    def test_fit_predict_synthetic(self):
        """Model trains on synthetic data and produces predictions."""
        X, y = _make_synthetic_data()
        model = DatasetQualityModel()
        model.fit(X, y)
        pred = model.predict(X[0])
        assert isinstance(pred, float)

    def test_predictions_bounded(self):
        """Predictions should always be in [0, 1] even with extreme inputs."""
        rng = np.random.default_rng(42)
        X = rng.random((20, FEATURE_DIM)).astype(np.float32) * 100
        y = np.array([0.0] * 10 + [1.0] * 10, dtype=np.float32)
        model = DatasetQualityModel()
        model.fit(X, y)
        for i in range(len(X)):
            pred = model.predict(X[i])
            assert 0.0 <= pred <= 1.0, f"Prediction {pred} out of bounds"

    def test_single_sample_predict(self):
        """Should handle 1D input via reshape."""
        X, y = _make_synthetic_data(20)
        model = DatasetQualityModel()
        model.fit(X, y)
        # Pass a 1D array
        pred = model.predict(X[0])
        assert isinstance(pred, float)
        assert 0.0 <= pred <= 1.0

    def test_uncertainty_estimation(self):
        """Uncertainty should be non-negative."""
        X, y = _make_synthetic_data(20)
        model = DatasetQualityModel()
        model.fit(X, y)
        pred, uncert = model.predict_with_uncertainty(X[0])
        assert isinstance(pred, float)
        assert isinstance(uncert, float)
        assert uncert >= 0.0
        assert 0.0 <= pred <= 1.0

    def test_save_load_roundtrip(self, tmp_path):
        """Saved model produces identical predictions after loading."""
        X, y = _make_synthetic_data(20)
        model = DatasetQualityModel()
        model.fit(X, y)
        preds_before = [model.predict(X[i]) for i in range(len(X))]

        model_path = tmp_path / "model.pkl"
        model.save(model_path)

        loaded = DatasetQualityModel(model_path=model_path)
        preds_after = [loaded.predict(X[i]) for i in range(len(X))]

        np.testing.assert_array_almost_equal(preds_before, preds_after)

    def test_loocv_results_stored(self):
        """After fit, loocv_results should have expected keys."""
        X, y = _make_synthetic_data(15)
        model = DatasetQualityModel()
        model.fit(X, y)

        assert "spearman_rho" in model.loocv_results
        assert "pearson_r" in model.loocv_results
        assert "mae" in model.loocv_results
        assert "predictions" in model.loocv_results
        assert "actuals" in model.loocv_results
        assert "n_samples" in model.loocv_results
        assert model.loocv_results["n_samples"] == 15
        assert model.loocv_results["mae"] >= 0

    def test_model_size_under_5mb(self, tmp_path):
        """Saved model should be under 5MB."""
        X, y = _make_synthetic_data(50)
        model = DatasetQualityModel()
        model.fit(X, y)
        model_path = tmp_path / "model.pkl"
        model.save(model_path)
        size_mb = model_path.stat().st_size / (1024 * 1024)
        assert size_mb < 5.0, f"Model size {size_mb:.2f}MB exceeds 5MB limit"


class TestExtractFromMetadata:
    def test_returns_64_dim(self):
        """extract_from_metadata should return a 64-dim vector."""
        from orbit.profile.feature_extractor import DatasetFeatureExtractor

        ext = DatasetFeatureExtractor()
        features = ext.extract_from_metadata(
            {
                "num_episodes": 1000,
                "avg_episode_length": 40,
                "action_dims": 7,
                "image_resolution": [320, 240],
                "diversity_estimate": "high",
                "quality_estimate": "high",
            }
        )
        assert features.shape == (64,)
        assert features.dtype == np.float32

    def test_quality_diversity_mapping(self):
        """High quality/diversity should produce different values than low."""
        from orbit.profile.feature_extractor import DatasetFeatureExtractor

        ext = DatasetFeatureExtractor()
        high = ext.extract_from_metadata(
            {
                "num_episodes": 1000,
                "avg_episode_length": 40,
                "action_dims": 7,
                "diversity_estimate": "high",
                "quality_estimate": "high",
            }
        )
        low = ext.extract_from_metadata(
            {
                "num_episodes": 1000,
                "avg_episode_length": 40,
                "action_dims": 7,
                "diversity_estimate": "low",
                "quality_estimate": "low",
            }
        )
        assert not np.array_equal(high, low)

    def test_handles_missing_fields(self):
        """Should handle missing optional fields gracefully."""
        from orbit.profile.feature_extractor import DatasetFeatureExtractor

        ext = DatasetFeatureExtractor()
        features = ext.extract_from_metadata({"num_episodes": 100})
        assert features.shape == (64,)
        assert np.all(np.isfinite(features))
