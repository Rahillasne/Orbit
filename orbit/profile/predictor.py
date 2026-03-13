"""Learned predictor for downstream task success rate.

Takes the 64-dim dataset feature vector produced by
:class:`DatasetFeatureExtractor` and predicts the expected success rate
of a policy trained on that dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class DatasetQualityModel:
    """Predicts downstream task success rate from dataset features."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        if model_path:
            self.load(model_path)
        else:
            self._build_default_model()

    def _build_default_model(self) -> None:
        """Build the prediction model — an ensemble for robustness."""
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.linear_model import Ridge
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        # Ensemble of 4 models — average their predictions
        self.models = {
            "gbr": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        GradientBoostingRegressor(
                            n_estimators=200,
                            max_depth=4,
                            learning_rate=0.05,
                            subsample=0.8,
                            min_samples_leaf=3,
                        ),
                    ),
                ]
            ),
            "rf": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=200,
                            max_depth=6,
                            min_samples_leaf=2,
                        ),
                    ),
                ]
            ),
            "mlp": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        MLPRegressor(
                            hidden_layer_sizes=(128, 64, 32),
                            activation="relu",
                            max_iter=1000,
                            early_stopping=True,
                            validation_fraction=0.15,
                        ),
                    ),
                ]
            ),
            "ridge": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", Ridge(alpha=1.0)),
                ]
            ),
        }
        self.weights = {"gbr": 0.35, "rf": 0.25, "mlp": 0.25, "ridge": 0.15}
        self.loocv_results: dict = {}

    def fit(
        self,
        features: np.ndarray,
        success_rates: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        """Train all models on (features, success_rates) pairs.

        Parameters
        ----------
        sample_weights:
            Per-sample importance weights.  Models that support
            ``sample_weight`` in ``.fit()`` will use them; others ignore.
        """
        success_rates = np.clip(success_rates, 0.0, 1.0)

        for name, model in self.models.items():
            if sample_weights is not None:
                try:
                    model.fit(
                        features,
                        success_rates,
                        **{
                            f"{model.steps[-1][0]}__sample_weight": sample_weights,
                        },
                    )
                except TypeError:
                    model.fit(features, success_rates)
            else:
                model.fit(features, success_rates)

        # Compute leave-one-out cross-validation
        self.loocv_results = self._compute_loocv(
            features,
            success_rates,
            sample_weights,
        )

    def predict(self, features: np.ndarray) -> float:
        """Predict success rate for a single dataset feature vector."""
        if features.ndim == 1:
            features = features.reshape(1, -1)

        predictions = {}
        for name, model in self.models.items():
            pred = model.predict(features)[0]
            predictions[name] = np.clip(pred, 0.0, 1.0)

        # Weighted ensemble average
        final = sum(predictions[name] * self.weights[name] for name in self.models)
        return float(np.clip(final, 0.0, 1.0))

    def predict_with_uncertainty(self, features: np.ndarray) -> tuple[float, float]:
        """Return (prediction, uncertainty) where uncertainty is std across ensemble."""
        if features.ndim == 1:
            features = features.reshape(1, -1)

        preds = [
            float(np.clip(model.predict(features)[0], 0.0, 1.0)) for model in self.models.values()
        ]
        return float(np.mean(preds)), float(np.std(preds))

    def _compute_loocv(
        self,
        features: np.ndarray,
        success_rates: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> dict:
        """Leave-one-out cross-validation to assess model quality."""
        from scipy import stats
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import LeaveOneOut
        from sklearn.preprocessing import StandardScaler

        loo = LeaveOneOut()
        predictions = []
        actuals = []

        for train_idx, test_idx in loo.split(features):
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = success_rates[train_idx], success_rates[test_idx]
            w_train = sample_weights[train_idx] if sample_weights is not None else None

            # Train a quick GBR for LOO (faster than full ensemble)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
            )
            model.fit(X_train_s, y_train, sample_weight=w_train)
            pred = model.predict(X_test_s)[0]

            predictions.append(float(np.clip(pred, 0, 1)))
            actuals.append(float(y_test[0]))

        predictions_arr = np.array(predictions)
        actuals_arr = np.array(actuals)

        rho, rho_p = stats.spearmanr(predictions_arr, actuals_arr)
        r, r_p = stats.pearsonr(predictions_arr, actuals_arr)
        mae = float(np.mean(np.abs(predictions_arr - actuals_arr)))

        return {
            "spearman_rho": float(rho),
            "spearman_p": float(rho_p),
            "pearson_r": float(r),
            "pearson_p": float(r_p),
            "mae": mae,
            "predictions": predictions_arr.tolist(),
            "actuals": actuals_arr.tolist(),
            "n_samples": len(actuals_arr),
        }

    def save(self, path: str | Path) -> None:
        """Persist the trained model to disk."""
        import joblib

        joblib.dump(
            {"models": self.models, "weights": self.weights, "loocv": self.loocv_results},
            path,
            compress=3,
        )

    def load(self, path: str | Path) -> None:
        """Load a previously saved model."""
        import joblib

        data = joblib.load(path)
        self.models = data["models"]
        self.weights = data["weights"]
        self.loocv_results = data.get("loocv", {})
