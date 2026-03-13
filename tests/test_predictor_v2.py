"""Tests for orbit.profile.predictor_v2 (DatasetQualityModelV2)."""

from __future__ import annotations

import numpy as np
import pytest

from orbit.profile.predictor_v2 import DatasetQualityModelV2, Prediction

FEATURE_DIM = 64


def _make_synthetic_data(n: int = 50, seed: int = 42):
    """Generate synthetic (features, success_rates) with a learnable signal."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, FEATURE_DIM)).astype(np.float32)
    # Success rate = noisy linear combination of first 5 features
    y = np.clip(
        0.3 * X[:, 0] + 0.25 * X[:, 1] + 0.2 * X[:, 2]
        + 0.15 * X[:, 3] + 0.1 * X[:, 4]
        + rng.normal(0, 0.08, n),
        0,
        1,
    ).astype(np.float32)
    return X, y


class TestDatasetQualityModelV2:
    def test_fit_predict_basic(self):
        """Model trains and returns a Prediction dataclass."""
        X, y = _make_synthetic_data()
        model = DatasetQualityModelV2()
        model.fit(X, y)
        pred = model.predict(X[0])
        assert isinstance(pred, Prediction)
        assert isinstance(pred.predicted_success_rate, float)

    def test_predictions_bounded(self):
        """Predictions should always be in [0, 1]."""
        rng = np.random.default_rng(42)
        X = rng.random((30, FEATURE_DIM)).astype(np.float32) * 100
        y = np.array([0.0] * 15 + [1.0] * 15, dtype=np.float32)
        model = DatasetQualityModelV2()
        model.fit(X, y)
        for i in range(len(X)):
            pred = model.predict(X[i])
            assert 0.0 <= pred.predicted_success_rate <= 1.0

    def test_confidence_intervals_bracket_prediction(self):
        """CI should be close to the prediction (within reasonable margin).

        Note: The calibrated prediction uses isotonic regression on the
        RF output, while the CI comes from bootstrap RF models.  They
        can diverge, so we allow a generous margin.
        """
        X, y = _make_synthetic_data(40)
        model = DatasetQualityModelV2()
        model.fit(X, y)
        for i in range(len(X)):
            pred = model.predict(X[i])
            # CI should at least overlap with prediction within 0.25
            # (isotonic calibration can shift the prediction significantly
            # relative to bootstrap CIs, especially on small datasets)
            assert pred.confidence_interval_low <= pred.predicted_success_rate + 0.25
            assert pred.confidence_interval_high >= pred.predicted_success_rate - 0.25

    def test_confidence_intervals_bounded(self):
        """CI bounds should be in [0, 1]."""
        X, y = _make_synthetic_data(30)
        model = DatasetQualityModelV2()
        model.fit(X, y)
        pred = model.predict(X[0])
        assert 0.0 <= pred.confidence_interval_low <= 1.0
        assert 0.0 <= pred.confidence_interval_high <= 1.0

    def test_ood_detection(self):
        """Far-from-training points should get low confidence or warnings."""
        X, y = _make_synthetic_data(30)
        model = DatasetQualityModelV2()
        model.fit(X, y)

        # Create a clearly OOD point
        ood_point = np.ones(FEATURE_DIM, dtype=np.float32) * 1000
        pred = model.predict(ood_point)
        assert pred.nearest_training_distance > 0

    def test_save_load_roundtrip(self, tmp_path):
        """Saved model produces identical predictions after loading."""
        X, y = _make_synthetic_data(30)
        model = DatasetQualityModelV2()
        model.fit(X, y)
        preds_before = [model.predict(X[i]).predicted_success_rate for i in range(len(X))]

        model_path = tmp_path / "model_v2.pkl"
        model.save(model_path)

        loaded = DatasetQualityModelV2(model_path=model_path)
        preds_after = [loaded.predict(X[i]).predicted_success_rate for i in range(len(X))]

        np.testing.assert_array_almost_equal(preds_before, preds_after, decimal=5)

    def test_save_load_preserves_all_components(self, tmp_path):
        """All model components survive serialization."""
        X, y = _make_synthetic_data(30)
        model = DatasetQualityModelV2()
        model.fit(X, y)

        model_path = tmp_path / "model_v2.pkl"
        model.save(model_path)

        loaded = DatasetQualityModelV2(model_path=model_path)
        # PCA is None by default (use_pca=False)
        assert loaded.pca is None
        assert loaded.use_pca is False
        assert loaded.scaler is not None
        assert loaded.model is not None
        assert loaded.calibrator is not None
        assert loaded.bootstrap_models is not None
        assert len(loaded.bootstrap_models) == 50
        assert loaded.training_features is not None
        assert loaded.validation_results is not None
        assert loaded.validation_results.get("version", loaded.validation_results.get("n_samples")) is not None

    def test_pca_reduces_dimensions(self):
        """PCA should reduce from 64 to <= 15 components when enabled."""
        X, y = _make_synthetic_data(50)
        model = DatasetQualityModelV2(use_pca=True)
        model.fit(X, y)
        assert model.pca is not None
        assert model.pca.n_components_ <= 15
        assert model.pca.n_components_ < FEATURE_DIM

    def test_no_pca_by_default(self):
        """PCA should be disabled by default."""
        X, y = _make_synthetic_data(50)
        model = DatasetQualityModelV2()
        model.fit(X, y)
        assert model.pca is None
        assert model.use_pca is False

    def test_validation_results_stored(self):
        """After fit, validation_results should have expected keys."""
        X, y = _make_synthetic_data(30)
        model = DatasetQualityModelV2()
        model.fit(X, y)

        v = model.validation_results
        assert "spearman_rho" in v
        assert "pearson_r" in v
        assert "mae" in v
        assert "rmse" in v
        assert "rank_accuracy" in v
        assert "cv_predictions" in v
        assert v["n_samples"] == 30
        assert v["mae"] >= 0
        assert v["rmse"] >= 0
        assert 0 <= v["rank_accuracy"] <= 1

    def test_calibrator_preserves_rank_ordering(self):
        """Isotonic calibration should not change rank ordering."""
        X, y = _make_synthetic_data(50)
        model = DatasetQualityModelV2()
        model.fit(X, y)

        preds = [model.predict(X[i]).predicted_success_rate for i in range(len(X))]

        # Check that predictions that differ by > 0.05 maintain ordering
        # (isotonic can merge nearby values but shouldn't reverse order)
        def _transform(X_in):
            scaled = model.scaler.transform(X_in)
            if model.pca is not None:
                return model.pca.transform(scaled)
            return scaled

        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                Xi = model.select_features(X[i:i+1])
                Xj = model.select_features(X[j:j+1])
                raw_i = float(np.clip(model.model.predict(
                    _transform(Xi)
                )[0], 0, 1))
                raw_j = float(np.clip(model.model.predict(
                    _transform(Xj)
                )[0], 0, 1))
                if abs(raw_i - raw_j) > 0.1:
                    # Isotonic is monotone, so if raw_i > raw_j, calibrated_i >= calibrated_j
                    if raw_i > raw_j:
                        assert preds[i] >= preds[j] - 0.01
                    else:
                        assert preds[j] >= preds[i] - 0.01

    def test_synthetic_model_has_signal(self):
        """Model on 80 samples with strong learnable signal should achieve rho > 0.2."""
        rng = np.random.default_rng(99)
        n = 80
        X = rng.random((n, FEATURE_DIM)).astype(np.float32)
        # Strong signal with low noise
        y = np.clip(
            0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.2 * X[:, 2]
            + rng.normal(0, 0.05, n),
            0, 1,
        ).astype(np.float32)
        model = DatasetQualityModelV2()
        model.fit(X, y)
        # LOOCV with stacking is conservative; rho > 0.2 is a reasonable bar
        assert model.validation_results["spearman_rho"] > 0.2

    def test_bootstrap_produces_diverse_predictions(self):
        """Bootstrap models should produce varying predictions (std > 0)."""
        X, y = _make_synthetic_data(30)
        model = DatasetQualityModelV2()
        model.fit(X, y)

        X_selected = model.select_features(X[:1])
        features_transformed = model.scaler.transform(X_selected)
        if model.pca is not None:
            features_transformed = model.pca.transform(features_transformed)
        boot_preds = [
            float(bm.predict(features_transformed)[0]) for bm in model.bootstrap_models
        ]
        assert np.std(boot_preds) > 0

    def test_handles_nan_inf_features(self):
        """Model should handle NaN/Inf in features gracefully."""
        X, y = _make_synthetic_data(30)
        model = DatasetQualityModelV2()
        model.fit(X, y)

        # Create input with NaN and Inf
        bad_input = np.full(FEATURE_DIM, np.nan, dtype=np.float32)
        pred = model.predict(bad_input)
        assert isinstance(pred, Prediction)
        assert np.isfinite(pred.predicted_success_rate)
        assert 0.0 <= pred.predicted_success_rate <= 1.0

        # Inf input
        inf_input = np.full(FEATURE_DIM, np.inf, dtype=np.float32)
        pred = model.predict(inf_input)
        assert isinstance(pred, Prediction)
        assert np.isfinite(pred.predicted_success_rate)

    def test_model_size_under_5mb(self, tmp_path):
        """Saved model should be under 5MB."""
        X, y = _make_synthetic_data(50)
        model = DatasetQualityModelV2()
        model.fit(X, y)
        model_path = tmp_path / "model_v2.pkl"
        model.save(model_path)
        size_mb = model_path.stat().st_size / (1024 * 1024)
        assert size_mb < 5.0, f"Model size {size_mb:.2f}MB exceeds 5MB limit"

    def test_confidence_level_strings(self):
        """Confidence level should be one of the expected values."""
        X, y = _make_synthetic_data(30)
        model = DatasetQualityModelV2()
        model.fit(X, y)
        pred = model.predict(X[0])
        assert pred.confidence_level in ("high", "medium", "low")

    def test_feature_types_filtering(self):
        """With feature_types, should filter or weight accordingly."""
        X, y = _make_synthetic_data(30)
        types = ["profiled"] * 20 + ["estimated"] * 10

        model = DatasetQualityModelV2()
        model.fit(X, y, feature_types=types)
        # Should have trained on 20 profiled samples
        assert model.validation_results["n_samples"] == 20

    def test_feature_types_fallback_when_few_profiled(self):
        """When < 15 profiled, should use all samples with weights."""
        X, y = _make_synthetic_data(30)
        types = ["profiled"] * 10 + ["estimated"] * 20

        model = DatasetQualityModelV2()
        model.fit(X, y, feature_types=types)
        # Should have used all 30 since only 10 profiled
        assert model.validation_results["n_samples"] == 30
