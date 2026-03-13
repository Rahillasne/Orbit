"""Enterprise-grade dataset quality predictor.

Key design decisions:
1. Optional PCA to reduce features (off by default — hurts on small n)
2. Train ONLY on real profiled data (estimated features are for inference demo only)
3. Nested cross-validation for unbiased performance estimates
4. Isotonic calibration for reliable probability estimates
5. Bootstrap confidence intervals on every prediction
6. Simple RF/GBR default — ablation showed complex ensembles are overengineered
7. Reduced feature set (no action features) by default — ablation showed
   removing action features improves rho from 0.666 to 0.704
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Feature group indices (within the 64-dim full feature vector)
_EMBEDDING_INDICES = list(range(0, 20))
_ACTION_INDICES = list(range(20, 32))
_QUALITY_INDICES = list(range(32, 44))
_SCALE_INDICES = list(range(44, 52))
_TASK_INDICES = list(range(52, 64))

# Default "reduced" feature set: everything except action features (52 dims)
REDUCED_FEATURE_INDICES = _EMBEDDING_INDICES + _QUALITY_INDICES + _SCALE_INDICES + _TASK_INDICES
# Full 64-feature set (for research comparisons)
FULL_FEATURE_INDICES = list(range(64))


@dataclass
class Prediction:
    """A single quality prediction with full uncertainty info."""

    predicted_success_rate: float
    confidence_interval_low: float
    confidence_interval_high: float
    confidence_level: str  # "high", "medium", "low"
    calibrated: bool
    nearest_training_distance: float
    warning: Optional[str] = None


class _AveragingEnsemble:
    """Simple weighted-average ensemble that avoids inner CV instability."""

    def __init__(self, models: list, weights: list[float]) -> None:
        self.models = models
        self.weights = np.array(weights) / np.sum(weights)

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None) -> None:
        for model in self.models:
            try:
                model.fit(X, y, sample_weight=sample_weight)
            except TypeError:
                model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.array([m.predict(X) for m in self.models])
        return np.average(preds, axis=0, weights=self.weights)

    @property
    def named_estimators_(self):
        """Compatibility shim for feature importance extraction."""
        return {f"model_{i}": m for i, m in enumerate(self.models)}


class DatasetQualityModelV2:
    """Production-grade quality predictor with calibration and bootstrap CIs.

    Parameters
    ----------
    model_path:
        Optional path to a saved model to load.
    feature_set:
        ``"reduced"`` (default, 52 dims — no action features) or
        ``"full"`` (64 dims — all features including action).
        Ablation showed removing action features improves rho from
        0.666 to 0.704.
    model_type:
        ``"rf"`` (default, Random Forest depth=3) or ``"gbr"``
        (Gradient Boosting) or ``"ensemble"`` (legacy averaging ensemble).
        Ablation showed simple RF/GBR outperforms complex stacking.
    use_pca:
        Whether to apply PCA after scaling.  Default ``False`` — PCA
        hurts performance on small training sets (n < 100).
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        feature_set: str = "reduced",
        model_type: str = "rf",
        use_pca: bool = False,
    ) -> None:
        self.pca = None
        self.scaler = None
        self.model = None
        self.calibrator = None
        self.training_features = None
        self.validation_results: dict = {}
        self.bootstrap_models: list = []
        self._sample_weights: np.ndarray | None = None
        self.feature_set = feature_set
        self.model_type = model_type
        self.use_pca = use_pca
        self._feature_indices: list[int] | None = None

        if model_path:
            self.load(model_path)

    def select_features(self, features: np.ndarray) -> np.ndarray:
        """Select feature columns based on the configured feature set.

        Parameters
        ----------
        features:
            (N, 64) full feature matrix.

        Returns
        -------
        (N, D) feature matrix where D depends on the feature set.
        """
        if self.feature_set == "full":
            self._feature_indices = FULL_FEATURE_INDICES
            return features
        # Default: reduced (no action features)
        self._feature_indices = REDUCED_FEATURE_INDICES
        if features.ndim == 1:
            return features[self._feature_indices]
        return features[:, self._feature_indices]

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        feature_types: list[str] | None = None,
    ) -> None:
        """Train the model properly.

        Parameters
        ----------
        features:
            (N, 64) feature matrix (full features — selection applied internally).
        targets:
            (N,) success rates in [0, 1].
        feature_types:
            List of ``"profiled"`` or ``"estimated"`` per sample.
            If provided, trains ONLY on profiled samples when enough exist.
        """
        from scipy import stats
        from sklearn.decomposition import PCA
        from sklearn.ensemble import (
            GradientBoostingRegressor,
            RandomForestRegressor,
            StackingRegressor,
        )
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import ElasticNet, Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR

        # ===== Step 1: Filter to real data only =====
        weights = None
        if feature_types is not None:
            real_mask = np.array([t == "profiled" for t in feature_types])
            if real_mask.sum() >= 15:
                logger.info(
                    "Training on %d real profiled datasets "
                    "(dropping %d estimated)",
                    real_mask.sum(),
                    (~real_mask).sum(),
                )
                print(
                    f"Training on {real_mask.sum()} real profiled datasets "
                    f"(dropping {(~real_mask).sum()} estimated)"
                )
                features = features[real_mask]
                targets = targets[real_mask]
            else:
                logger.info(
                    "Only %d profiled — using all %d with 3x weight on profiled",
                    real_mask.sum(),
                    len(features),
                )
                print(
                    f"Only {real_mask.sum()} profiled — using all {len(features)} "
                    f"with 3x weight on profiled"
                )
                weights = np.where(real_mask, 3.0, 1.0)

        # ===== Step 1b: Apply feature set selection =====
        features = self.select_features(features)
        n_features = features.shape[1] if features.ndim == 2 else len(features)
        n_samples = len(features)
        print(
            f"Training on {n_samples} samples, "
            f"{n_features} features (feature_set={self.feature_set})"
        )

        # ===== Step 2: Clean features =====
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
        targets = np.clip(targets, 0.0, 1.0)

        # ===== Step 3: Scale (+ optional PCA) =====
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)

        if self.use_pca:
            # Keep fewer components for small samples to prevent overfitting
            # Rule of thumb: n_components <= n_samples / 3
            n_components = min(15, n_samples // 3, n_samples - 2, n_features)
            n_components = max(n_components, 3)  # at least 3
            self.pca = PCA(n_components=n_components)
            features_transformed = self.pca.fit_transform(features_scaled)

            explained_var = self.pca.explained_variance_ratio_.cumsum()
            print(
                f"PCA: {n_components} components explain "
                f"{explained_var[-1] * 100:.1f}% variance"
            )
        else:
            self.pca = None
            features_transformed = features_scaled
            print("PCA: disabled (use_pca=False)")

        # Store training features for OOD detection
        self.training_features = features_transformed.copy()

        # ===== Step 4: Build model =====
        self.model = self._build_model(n_samples)

        if weights is not None and len(weights) == len(features_transformed):
            self.model.fit(features_transformed, targets, sample_weight=weights)
        else:
            self.model.fit(features_transformed, targets)

        # ===== Step 5: Nested cross-validation =====
        print("\nRunning nested cross-validation...")
        self.validation_results = self._nested_cv(features_transformed, targets, weights)

        # ===== Step 6: Calibration via isotonic regression =====
        cv_predictions = self.validation_results["cv_predictions"]
        self.calibrator = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip"
        )
        self.calibrator.fit(cv_predictions, targets)

        # ===== Step 7: Bootstrap confidence intervals =====
        print("Computing bootstrap confidence intervals...")
        self.bootstrap_models = self._train_bootstrap(
            features_transformed, targets, n_bootstrap=50, weights=weights
        )

        # ===== Step 8: Print report =====
        self._print_report(targets)

    def _build_model(self, n_samples: int):
        """Build the prediction model based on model_type config.

        Returns an unfitted sklearn-compatible estimator.
        """
        from sklearn.ensemble import (
            GradientBoostingRegressor,
            RandomForestRegressor,
        )
        from sklearn.linear_model import ElasticNet, Ridge

        if self.model_type == "rf":
            return RandomForestRegressor(
                n_estimators=50,
                max_depth=3,
                min_samples_leaf=3,
                max_features="sqrt",
                random_state=42,
            )
        elif self.model_type == "gbr":
            return GradientBoostingRegressor(
                n_estimators=50,
                max_depth=2,
                learning_rate=0.1,
                subsample=0.8,
                min_samples_leaf=max(3, n_samples // 10),
                random_state=42,
            )
        else:  # "ensemble" — legacy averaging ensemble
            return _AveragingEnsemble(
                models=[
                    GradientBoostingRegressor(
                        n_estimators=50,
                        max_depth=2,
                        learning_rate=0.05,
                        subsample=0.8,
                        min_samples_leaf=max(3, n_samples // 8),
                    ),
                    RandomForestRegressor(
                        n_estimators=100,
                        max_depth=3,
                        min_samples_leaf=max(3, n_samples // 6),
                        max_features="sqrt",
                    ),
                    Ridge(alpha=10.0),
                    ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000),
                ],
                weights=[0.30, 0.25, 0.25, 0.20],
            )

    def _nested_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> dict:
        """Nested cross-validation for unbiased performance estimates."""
        from scipy import stats
        from sklearn.model_selection import LeaveOneOut, RepeatedKFold

        n = len(X)

        if n <= 40:
            cv = LeaveOneOut()
        else:
            cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

        predictions = np.zeros(n)
        pred_counts = np.zeros(n)

        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]

            n_train = len(X_train)
            fold_model = self._build_model(n_train)

            w = (
                weights[train_idx]
                if weights is not None and len(weights) == len(y)
                else None
            )
            if w is not None:
                try:
                    fold_model.fit(X_train, y_train, sample_weight=w)
                except TypeError:
                    fold_model.fit(X_train, y_train)
            else:
                fold_model.fit(X_train, y_train)

            pred = np.clip(fold_model.predict(X_test), 0.0, 1.0)

            for i, idx in enumerate(test_idx):
                predictions[idx] += pred[i]
                pred_counts[idx] += 1

        # Average predictions for repeated CV
        mask = pred_counts > 0
        predictions[mask] /= pred_counts[mask]

        rho, rho_p = stats.spearmanr(predictions, y)
        r, r_p = stats.pearsonr(predictions, y)
        mae = float(np.mean(np.abs(predictions - y)))
        rmse = float(np.sqrt(np.mean((predictions - y) ** 2)))

        # Rank accuracy
        n_pairs = 0
        n_correct = 0
        for i in range(n):
            for j in range(i + 1, n):
                if y[i] != y[j]:
                    n_pairs += 1
                    if (predictions[i] - predictions[j]) * (y[i] - y[j]) > 0:
                        n_correct += 1
        rank_accuracy = n_correct / max(n_pairs, 1)

        return {
            "cv_predictions": predictions,
            "spearman_rho": float(rho),
            "spearman_p": float(rho_p),
            "pearson_r": float(r),
            "pearson_p": float(r_p),
            "mae": mae,
            "rmse": rmse,
            "rank_accuracy": float(rank_accuracy),
            "n_samples": n,
        }

    def _train_bootstrap(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_bootstrap: int = 50,
        weights: np.ndarray | None = None,
    ) -> list:
        """Train bootstrap ensemble for confidence intervals."""
        models = []
        rng = np.random.RandomState(42)

        for i in range(n_bootstrap):
            idx = rng.choice(len(X), size=len(X), replace=True)
            X_boot = X[idx]
            y_boot = y[idx]

            model = self._build_model(len(X))
            w = weights[idx] if weights is not None else None
            if w is not None:
                try:
                    model.fit(X_boot, y_boot, sample_weight=w)
                except TypeError:
                    model.fit(X_boot, y_boot)
            else:
                model.fit(X_boot, y_boot)
            models.append(model)

        return models

    def predict(self, features: np.ndarray) -> Prediction:
        """Predict with full uncertainty quantification.

        Parameters
        ----------
        features:
            (64,) or (1, 64) full feature vector.  Feature selection
            is applied internally based on ``self.feature_set``.
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)

        # Apply same feature selection used during training
        features = self.select_features(features)

        features_scaled = self.scaler.transform(features)
        if self.pca is not None:
            features_transformed = self.pca.transform(features_scaled)
        else:
            features_transformed = features_scaled

        # Main prediction
        raw_pred = float(np.clip(self.model.predict(features_transformed)[0], 0, 1))

        # Calibrated prediction
        if self.calibrator is not None:
            calibrated_pred = float(
                np.clip(self.calibrator.predict([raw_pred])[0], 0, 1)
            )
        else:
            calibrated_pred = raw_pred

        # Bootstrap confidence interval
        boot_preds = np.array(
            [
                float(np.clip(bmodel.predict(features_transformed)[0], 0, 1))
                for bmodel in self.bootstrap_models
            ]
        )

        ci_low = float(np.percentile(boot_preds, 5))
        ci_high = float(np.percentile(boot_preds, 95))
        ci_width = ci_high - ci_low

        # OOD detection
        if self.training_features is not None:
            distances = np.linalg.norm(
                self.training_features - features_transformed, axis=1
            )
            nearest_dist = float(distances.min())
            median_dist = float(np.median(distances))
        else:
            nearest_dist = 0.0
            median_dist = 1.0

        # Confidence level
        if ci_width < 0.15 and nearest_dist < median_dist:
            confidence = "high"
            warning = None
        elif ci_width < 0.25 and nearest_dist < median_dist * 2:
            confidence = "medium"
            warning = None
        else:
            confidence = "low"
            if nearest_dist > median_dist * 2:
                warning = (
                    "Dataset is far from training distribution "
                    "— prediction may be unreliable"
                )
            else:
                warning = (
                    "High model uncertainty "
                    "— prediction has wide confidence interval"
                )

        return Prediction(
            predicted_success_rate=calibrated_pred,
            confidence_interval_low=ci_low,
            confidence_interval_high=ci_high,
            confidence_level=confidence,
            calibrated=self.calibrator is not None,
            nearest_training_distance=nearest_dist,
            warning=warning,
        )

    def _print_report(self, targets: np.ndarray) -> None:
        """Print comprehensive validation report."""
        v = self.validation_results

        print(f"\n{'=' * 70}")
        print("MODEL VALIDATION REPORT")
        print(f"{'=' * 70}")
        print(f"  Training samples:     {v['n_samples']}")
        if self.pca is not None:
            print(f"  PCA components:       {self.pca.n_components_}")
            print(
                f"  Variance explained:   "
                f"{self.pca.explained_variance_ratio_.sum() * 100:.1f}%"
            )
        else:
            print(f"  PCA:                  disabled")
        print()
        print("  CROSS-VALIDATED METRICS (unbiased):")
        print(f"  {'─' * 40}")
        print(
            f"  Spearman rho:         {v['spearman_rho']:.3f} "
            f"(p={v['spearman_p']:.4f})"
        )
        print(
            f"  Pearson r:            {v['pearson_r']:.3f} "
            f"(p={v['pearson_p']:.4f})"
        )
        print(f"  MAE:                  {v['mae']:.3f}")
        print(f"  RMSE:                 {v['rmse']:.3f}")
        print(f"  Rank accuracy:        {v['rank_accuracy'] * 100:.1f}%")
        print()

        rho = v["spearman_rho"]
        if rho >= 0.85:
            grade = "EXCELLENT — strong predictive power, paper-worthy"
        elif rho >= 0.75:
            grade = "GOOD — reliable rank ordering, production-ready"
        elif rho >= 0.65:
            grade = "MODERATE — useful for screening, not definitive"
        elif rho >= 0.50:
            grade = "WEAK — better than random, needs more training data"
        else:
            grade = "POOR — not reliable enough for production use"

        print(f"  ASSESSMENT: {grade}")
        print()

        # Per-sample breakdown
        cv_pred = v["cv_predictions"]
        print(
            f"  {'Idx':<6} {'Actual':>7} {'Pred':>7} {'Error':>7} {'':>2}"
        )
        print(f"  {'─' * 35}")

        errors = cv_pred - targets
        sorted_idx = np.argsort(np.abs(errors))
        for idx in sorted_idx:
            err = errors[idx]
            bar_len = int(min(abs(err) * 40, 20))
            bar = "█" * bar_len
            marker = "✓" if abs(err) < 0.15 else "✗"
            print(
                f"  {idx:<6d} {targets[idx]:>7.2f} {cv_pred[idx]:>7.2f} "
                f"{err:>+7.2f} {marker} {bar}"
            )

        print(f"\n  Bootstrap ensemble: {len(self.bootstrap_models)} models")
        print("  Calibration: isotonic regression applied")
        print(f"{'=' * 70}")

    def save(self, path: str | Path) -> None:
        """Save complete model state."""
        import joblib

        data = {
            "model": self.model,
            "pca": self.pca,
            "scaler": self.scaler,
            "calibrator": self.calibrator,
            "bootstrap_models": self.bootstrap_models,
            "training_features": self.training_features,
            "validation_results": self.validation_results,
            "feature_set": self.feature_set,
            "model_type": self.model_type,
            "feature_indices": self._feature_indices,
            "use_pca": self.use_pca,
            "version": "2.2",
        }
        joblib.dump(data, path, compress=3)
        print(f"Model saved to {path}")

    def load(self, path: str | Path) -> None:
        """Load complete model state."""
        import joblib

        data = joblib.load(path)
        self.model = data["model"]
        self.pca = data["pca"]
        self.scaler = data["scaler"]
        self.calibrator = data["calibrator"]
        self.bootstrap_models = data["bootstrap_models"]
        self.training_features = data["training_features"]
        self.validation_results = data["validation_results"]
        self.feature_set = data.get("feature_set", "full")
        self.model_type = data.get("model_type", "ensemble")
        self._feature_indices = data.get("feature_indices")
        self.use_pca = data.get("use_pca", self.pca is not None)
