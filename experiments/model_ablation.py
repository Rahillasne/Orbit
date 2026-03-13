#!/usr/bin/env python3
"""Phase 1: Systematic model ablation — baselines, simplified models, LOOCV evaluation.

This is the most important experiment in the ORBIT project. It answers:
"Can we predict policy success from dataset properties alone?"

We compare dumb baselines against progressively more complex models,
all evaluated with leave-one-out cross-validation (LOOCV) on n=37 real
profiled datasets. The goal is to find the simplest model that achieves
Spearman rho >= 0.70.

Usage:
    python3 experiments/model_ablation.py
    python3 experiments/model_ablation.py --use-all  # include estimated features too
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────────────────────────────
# Feature names for interpretability
# ──────────────────────────────────────────────────────────────────

FEATURE_GROUPS = {
    "embedding": list(range(0, 20)),
    "action": list(range(20, 32)),
    "quality": list(range(32, 44)),
    "scale": list(range(44, 52)),
    "task": list(range(52, 64)),
}

FEATURE_NAMES = [
    # Embedding (0-19)
    "emb_mean_norm", "emb_std_norm", "emb_mean_pw_cos", "emb_std_pw_cos",
    "emb_min_pw_cos", "emb_max_pw_cos", "emb_num_clusters", "emb_noise_ratio",
    "emb_silhouette", "emb_calinski_harabasz", "emb_hull_volume", "emb_eff_dim",
    "emb_mean_seq_dist", "emb_std_seq_dist", "emb_mean_traj_len", "emb_std_traj_len",
    "emb_skewness_pc1", "emb_kurtosis_pc1", "emb_entropy", "emb_uniformity",
    # Action (20-31)
    "act_dims", "act_mean_mag", "act_std_mag", "act_smoothness",
    "act_range_util", "act_num_modes", "act_ep_entropy", "act_cross_consist",
    "act_eff_dim", "act_autocorr", "act_zero_frac", "act_boundary_frac",
    # Quality (32-43)
    "qual_aggregate", "qual_smoothness", "qual_completion", "qual_obs_consist",
    "qual_demo_quality", "qual_brightness_mean", "qual_brightness_std",
    "qual_blur", "qual_temporal_consist", "qual_reward", "qual_language", "qual_multi_cam",
    # Scale (44-51)
    "scale_log_eps", "scale_log_frames", "scale_avg_ep_len", "scale_std_ep_len",
    "scale_fps", "scale_img_res", "scale_obs_dims", "scale_size_mb",
    # Task (52-63)
    "task_primary_score", "task_visual_rel", "task_coverage_div", "task_data_qual",
    "task_coverage", "task_quality", "task_diversity", "task_volume",
    "task_emb_x_act", "task_clusters_per_ep", "task_hull_x_qual", "task_uniform_x_div",
]


# ──────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────


def load_data(
    cache_dir: str = "benchmarks/cached_features",
    gt_file: str = "orbit/benchmarks/ground_truth_comprehensive.json",
    profiled_only: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Load features and ground truth success rates.

    Returns: (features, targets, labels, types)
    """
    with open(gt_file) as f:
        gt = json.load(f)
    gt_lookup = {e["id"]: e for e in gt["benchmarks"]}

    features_list, success_rates, labels, types = [], [], [], []

    for npy_file in sorted(Path(cache_dir).glob("*.npy")):
        entry_id = npy_file.stem
        if entry_id not in gt_lookup:
            continue
        entry = gt_lookup[entry_id]

        meta_file = npy_file.with_suffix(".json")
        meta = {}
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)

        feat_type = meta.get("type", "unknown")
        if profiled_only and feat_type != "profiled":
            continue

        feat = np.load(npy_file)
        features_list.append(feat)

        sr = entry.get("reported_success_rate", meta.get("success_rate", 0.0))
        if entry.get("metric_type") == "normalized_score":
            sr = min(sr / 100.0, 1.0)

        success_rates.append(sr)
        labels.append(entry_id)
        types.append(feat_type)

    features = np.array(features_list, dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
    targets = np.array(success_rates, dtype=np.float64)

    return features, targets, labels, types


# ──────────────────────────────────────────────────────────────────
# LOOCV evaluation
# ──────────────────────────────────────────────────────────────────


def loocv_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    model_factory,
    preprocess=None,
) -> dict:
    """Evaluate a model with leave-one-out cross-validation.

    Parameters
    ----------
    X : features (n, d)
    y : targets (n,)
    model_factory : callable that returns a fresh sklearn estimator
    preprocess : optional callable(X_train, X_test) -> (X_train_t, X_test_t)
    """
    loo = LeaveOneOut()
    predictions = np.zeros_like(y)

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        if preprocess is not None:
            X_train, X_test = preprocess(X_train, X_test)

        model = model_factory()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions[test_idx] = np.clip(pred, 0.0, 1.0)

    rho, rho_p = stats.spearmanr(predictions, y)
    r, r_p = stats.pearsonr(predictions, y)
    mae = float(np.mean(np.abs(predictions - y)))
    rmse = float(np.sqrt(np.mean((predictions - y) ** 2)))

    # Rank accuracy (concordance)
    n = len(y)
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
        "predictions": predictions,
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "pearson_r": float(r),
        "pearson_p": float(r_p),
        "mae": mae,
        "rmse": rmse,
        "rank_accuracy": rank_accuracy,
    }


# ──────────────────────────────────────────────────────────────────
# Preprocessing helpers
# ──────────────────────────────────────────────────────────────────


def make_pca_preprocess(n_components: int):
    """Return a preprocess function that scales + PCA transforms."""
    def preprocess(X_train, X_test):
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        n_comp = min(n_components, X_train_s.shape[0] - 1, X_train_s.shape[1])
        pca = PCA(n_components=n_comp)
        X_train_p = pca.fit_transform(X_train_s)
        X_test_p = pca.transform(X_test_s)
        return X_train_p, X_test_p
    return preprocess


def scale_preprocess(X_train, X_test):
    """Just scale, no PCA."""
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)


# ──────────────────────────────────────────────────────────────────
# Models to evaluate
# ──────────────────────────────────────────────────────────────────


def get_models() -> list[tuple[str, callable, callable | None]]:
    """Return (name, model_factory, preprocess) tuples."""
    return [
        # ===== BASELINES =====
        (
            "Baseline A: Mean predictor",
            lambda: _MeanPredictor(),
            None,
        ),
        (
            "Baseline B: Log(episodes) only",
            lambda: Ridge(alpha=1.0),
            lambda Xtr, Xte: (Xtr[:, 44:45], Xte[:, 44:45]),  # scale_log_episodes
        ),
        (
            "Baseline C: Top-1 capability only",
            lambda: Ridge(alpha=1.0),
            lambda Xtr, Xte: (Xtr[:, 52:53], Xte[:, 52:53]),  # task_primary_score
        ),
        (
            "Baseline D: Linear regression (all 64)",
            lambda: LinearRegression(),
            scale_preprocess,
        ),
        (
            "Baseline E: Ridge (alpha=1.0, all 64)",
            lambda: Ridge(alpha=1.0),
            scale_preprocess,
        ),
        # ===== SIMPLIFIED MODELS =====
        (
            "Model S1: Ridge on PCA(5)",
            lambda: Ridge(alpha=1.0),
            make_pca_preprocess(5),
        ),
        (
            "Model S2: Ridge on PCA(3)",
            lambda: Ridge(alpha=1.0),
            make_pca_preprocess(3),
        ),
        (
            "Model S3: RF (depth=3, n=50)",
            lambda: RandomForestRegressor(
                n_estimators=50, max_depth=3,
                min_samples_leaf=3, random_state=42,
            ),
            scale_preprocess,
        ),
        (
            "Model S4: RF on PCA(5)",
            lambda: RandomForestRegressor(
                n_estimators=50, max_depth=3,
                min_samples_leaf=3, random_state=42,
            ),
            make_pca_preprocess(5),
        ),
        (
            "Model S5: GBR (depth=2, n=50, lr=0.1)",
            lambda: GradientBoostingRegressor(
                n_estimators=50, max_depth=2,
                learning_rate=0.1, subsample=0.8,
                min_samples_leaf=3, random_state=42,
            ),
            scale_preprocess,
        ),
        (
            "Model S6: GBR on PCA(5)",
            lambda: GradientBoostingRegressor(
                n_estimators=50, max_depth=2,
                learning_rate=0.1, subsample=0.8,
                min_samples_leaf=3, random_state=42,
            ),
            make_pca_preprocess(5),
        ),
        (
            "Model S7: ElasticNet (a=0.1, l1=0.5)",
            lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
            scale_preprocess,
        ),
        (
            "Model S8: Lasso (a=0.01)",
            lambda: Lasso(alpha=0.01, max_iter=5000),
            scale_preprocess,
        ),
        (
            "Model S9: Ridge (alpha=10.0, all 64)",
            lambda: Ridge(alpha=10.0),
            scale_preprocess,
        ),
        (
            "Model S10: Ridge (alpha=100.0, all 64)",
            lambda: Ridge(alpha=100.0),
            scale_preprocess,
        ),
        # ===== CURRENT PRODUCTION MODEL (for comparison) =====
        (
            "Current: AveragingEnsemble on PCA(12)",
            lambda: _CurrentEnsemble(),
            make_pca_preprocess(12),
        ),
    ]


class _MeanPredictor:
    """Always predict the training mean."""
    def fit(self, X, y):
        self.mean_ = np.mean(y)
    def predict(self, X):
        return np.full(len(X), self.mean_)


class _CurrentEnsemble:
    """Reproduce the current production model for fair comparison."""
    def __init__(self):
        from orbit.profile.predictor_v2 import _AveragingEnsemble
        self.model = _AveragingEnsemble(
            models=[
                GradientBoostingRegressor(
                    n_estimators=50, max_depth=2,
                    learning_rate=0.05, subsample=0.8,
                    min_samples_leaf=3,
                ),
                RandomForestRegressor(
                    n_estimators=100, max_depth=3,
                    min_samples_leaf=3, max_features="sqrt",
                ),
                Ridge(alpha=10.0),
                ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000),
            ],
            weights=[0.30, 0.25, 0.25, 0.20],
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


# ──────────────────────────────────────────────────────────────────
# Main experiment
# ──────────────────────────────────────────────────────────────────


def run_ablation(features, targets, labels, types):
    """Run all models and return results."""
    models = get_models()
    results = []

    print(f"\n{'=' * 90}")
    print(f"MODEL ABLATION STUDY — {len(targets)} datasets ({sum(1 for t in types if t == 'profiled')} profiled)")
    print(f"{'=' * 90}")
    print(f"Target distribution: mean={targets.mean():.3f}, std={targets.std():.3f}, "
          f"min={targets.min():.3f}, max={targets.max():.3f}")
    print()

    header = f"{'Model':<45} {'ρ':>6} {'r':>6} {'MAE':>6} {'RMSE':>6} {'Rank%':>6}"
    print(header)
    print("─" * len(header))

    for name, factory, preprocess in models:
        res = loocv_evaluate(features, targets, factory, preprocess)
        res["name"] = name
        results.append(res)

        rho_str = f"{res['spearman_rho']:.3f}"
        r_str = f"{res['pearson_r']:.3f}"
        mae_str = f"{res['mae']:.3f}"
        rmse_str = f"{res['rmse']:.3f}"
        rank_str = f"{res['rank_accuracy'] * 100:.1f}"

        # Highlight best-so-far
        print(f"  {name:<43} {rho_str:>6} {r_str:>6} {mae_str:>6} {rmse_str:>6} {rank_str:>6}")

    # Sort by Spearman rho
    results.sort(key=lambda r: r["spearman_rho"], reverse=True)

    print(f"\n{'=' * 90}")
    print("RANKING BY SPEARMAN RHO")
    print(f"{'=' * 90}")
    for i, res in enumerate(results):
        marker = " ★" if i == 0 else ""
        print(f"  {i + 1:>2}. {res['name']:<43} ρ={res['spearman_rho']:.3f}  "
              f"MAE={res['mae']:.3f}{marker}")

    return results


def print_detailed_predictions(best_result, targets, labels):
    """Print per-dataset predictions for the best model."""
    print(f"\n{'=' * 90}")
    print(f"DETAILED PREDICTIONS — {best_result['name']}")
    print(f"{'=' * 90}")

    predictions = best_result["predictions"]
    errors = predictions - targets
    sorted_idx = np.argsort(np.abs(errors))

    print(f"\n{'Dataset':<42} {'Actual':>7} {'Pred':>7} {'Error':>7}")
    print("─" * 65)

    for idx in sorted_idx:
        err = errors[idx]
        marker = "✓" if abs(err) < 0.15 else "✗"
        print(f"  {labels[idx]:<40} {targets[idx]:>7.3f} {predictions[idx]:>7.3f} "
              f"{err:>+7.3f} {marker}")

    # 5 worst predictions
    worst_idx = np.argsort(np.abs(errors))[-5:][::-1]
    print(f"\n5 WORST PREDICTIONS:")
    for idx in worst_idx:
        print(f"  {labels[idx]:<40} actual={targets[idx]:.3f}  "
              f"pred={predictions[idx]:.3f}  error={errors[idx]:+.3f}")

    # Systematic bias check
    mean_residual = np.mean(errors)
    print(f"\nMean residual: {mean_residual:+.4f} {'(systematic low bias)' if mean_residual < -0.05 else '(systematic high bias)' if mean_residual > 0.05 else '(no systematic bias)'}")
    print(f"Median residual: {np.median(errors):+.4f}")


def run_feature_importance(features, targets, labels):
    """Permutation importance for the best model on full data."""
    from sklearn.inspection import permutation_importance

    print(f"\n{'=' * 90}")
    print("FEATURE IMPORTANCE (Permutation, 100 repeats)")
    print(f"{'=' * 90}")

    # Train Ridge on scaled features (our likely best simple model)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    model = Ridge(alpha=10.0)
    model.fit(X_scaled, targets)

    result = permutation_importance(
        model, X_scaled, targets,
        n_repeats=100, random_state=42,
        scoring="neg_mean_absolute_error",
    )

    importances = result.importances_mean
    sorted_idx = np.argsort(importances)[::-1]

    print(f"\n{'Rank':>4} {'Feature':<30} {'Importance':>12} {'Std':>10}")
    print("─" * 60)
    for rank, idx in enumerate(sorted_idx[:20]):
        name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"feat_{idx}"
        print(f"  {rank + 1:>3}. {name:<28} {importances[idx]:>12.4f} "
              f"±{result.importances_std[idx]:>8.4f}")

    # Also show Ridge coefficients
    print(f"\nTOP 10 RIDGE COEFFICIENTS (|coef|):")
    coefs = np.abs(model.coef_)
    sorted_coef_idx = np.argsort(coefs)[::-1]
    for rank, idx in enumerate(sorted_coef_idx[:10]):
        name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"feat_{idx}"
        print(f"  {rank + 1:>3}. {name:<28} coef={model.coef_[idx]:>+10.4f}")

    return importances, sorted_idx


def make_scatter_plot(best_result, targets, labels, output_path="docs/predicted_vs_actual.png"):
    """Save predicted vs actual scatter plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 8))

        predictions = best_result["predictions"]
        ax.scatter(targets, predictions, c="#2563eb", s=60, alpha=0.7,
                   edgecolors="black", linewidth=0.5, zorder=5)

        # Label outliers
        errors = np.abs(predictions - targets)
        threshold = np.percentile(errors, 80)
        for i in range(len(targets)):
            if errors[i] >= threshold:
                ax.annotate(
                    labels[i].replace("_", "\n"),
                    (targets[i], predictions[i]),
                    fontsize=6, alpha=0.7,
                    xytext=(5, 5), textcoords="offset points",
                )

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect", linewidth=1)
        ax.set_xlabel("Actual Success Rate", fontsize=12)
        ax.set_ylabel("Predicted Success Rate (LOOCV)", fontsize=12)
        ax.set_title(
            f"ORBIT Quality Predictor — {best_result['name']}\n"
            f"ρ={best_result['spearman_rho']:.3f}, r={best_result['pearson_r']:.3f}, "
            f"MAE={best_result['mae']:.3f}",
            fontsize=11,
        )
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\nScatter plot saved to {output_path}")
        plt.close()
    except ImportError:
        print("matplotlib not installed — skipping plot")


def save_results_markdown(results, best_result, targets, labels, output_path="docs/model_ablation_results.md"):
    """Save results as a clean markdown table."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write("# Model Ablation Results\n\n")
        f.write(f"**Dataset**: {len(targets)} datasets  \n")
        f.write(f"**Evaluation**: Leave-One-Out Cross-Validation (LOOCV)  \n")
        f.write(f"**Target**: Success rate in [0, 1]  \n\n")

        f.write("## Model Comparison\n\n")
        f.write("| Rank | Model | Spearman ρ | Pearson r | MAE | RMSE | Rank Acc. |\n")
        f.write("|------|-------|-----------|-----------|-----|------|----------|\n")

        for i, res in enumerate(results):
            f.write(f"| {i + 1} | {res['name']} | {res['spearman_rho']:.3f} | "
                    f"{res['pearson_r']:.3f} | {res['mae']:.3f} | {res['rmse']:.3f} | "
                    f"{res['rank_accuracy'] * 100:.1f}% |\n")

        f.write(f"\n## Best Model: {best_result['name']}\n\n")
        f.write(f"- Spearman ρ = {best_result['spearman_rho']:.3f} (p = {best_result['spearman_p']:.4f})\n")
        f.write(f"- Pearson r = {best_result['pearson_r']:.3f} (p = {best_result['pearson_p']:.4f})\n")
        f.write(f"- MAE = {best_result['mae']:.3f}\n")
        f.write(f"- RMSE = {best_result['rmse']:.3f}\n")
        f.write(f"- Rank Accuracy = {best_result['rank_accuracy'] * 100:.1f}%\n\n")

        f.write("## Per-Dataset Predictions (Best Model)\n\n")
        f.write("| Dataset | Actual | Predicted | Error |\n")
        f.write("|---------|--------|-----------|-------|\n")

        predictions = best_result["predictions"]
        errors = predictions - targets
        sorted_idx = np.argsort(np.abs(errors))

        for idx in sorted_idx:
            marker = "✓" if abs(errors[idx]) < 0.15 else "✗"
            f.write(f"| {labels[idx]} | {targets[idx]:.3f} | {predictions[idx]:.3f} | "
                    f"{errors[idx]:+.3f} {marker} |\n")

        # Diagnosis
        f.write("\n## Diagnosis\n\n")
        mean_res = np.mean(errors)
        f.write(f"- Mean residual: {mean_res:+.4f}\n")
        f.write(f"- Median residual: {np.median(errors):+.4f}\n")

        if mean_res < -0.05:
            f.write("- **Systematic low bias detected** — predictions consistently underestimate\n")
        elif mean_res > 0.05:
            f.write("- **Systematic high bias detected** — predictions consistently overestimate\n")
        else:
            f.write("- No significant systematic bias\n")

        # 5 worst
        f.write("\n### 5 Worst Predictions\n\n")
        worst_idx = np.argsort(np.abs(errors))[-5:][::-1]
        for idx in worst_idx:
            f.write(f"- **{labels[idx]}**: actual={targets[idx]:.3f}, "
                    f"pred={predictions[idx]:.3f}, error={errors[idx]:+.3f}\n")

    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Model ablation study")
    parser.add_argument("--use-all", action="store_true",
                        help="Include estimated features (not just profiled)")
    parser.add_argument("--cache-dir", default="benchmarks/cached_features")
    parser.add_argument("--gt-file", default="orbit/benchmarks/ground_truth_comprehensive.json")
    args = parser.parse_args()

    # Load data
    features, targets, labels, types = load_data(
        cache_dir=args.cache_dir,
        gt_file=args.gt_file,
        profiled_only=not args.use_all,
    )

    print(f"Loaded {len(features)} datasets "
          f"({sum(1 for t in types if t == 'profiled')} profiled, "
          f"{sum(1 for t in types if t != 'profiled')} estimated)")

    if len(features) < 5:
        print("ERROR: Not enough data. Run profile_training_data.py first.")
        sys.exit(1)

    # Run ablation
    results = run_ablation(features, targets, labels, types)

    # Best model
    best = results[0]

    # Detailed predictions
    print_detailed_predictions(best, targets, labels)

    # Feature importance
    run_feature_importance(features, targets, labels)

    # Scatter plot
    make_scatter_plot(best, targets, labels)

    # Save markdown
    save_results_markdown(results, best, targets, labels)

    # Final assessment
    rho = best["spearman_rho"]
    print(f"\n{'=' * 90}")
    print("FINAL ASSESSMENT")
    print(f"{'=' * 90}")
    if rho >= 0.85:
        print(f"  ρ = {rho:.3f} — EXCELLENT. Paper-worthy predictive power.")
    elif rho >= 0.70:
        print(f"  ρ = {rho:.3f} — GOOD. Reliable for screening. Publishable.")
    elif rho >= 0.50:
        print(f"  ρ = {rho:.3f} — MODERATE. Better than random, needs more data.")
    else:
        print(f"  ρ = {rho:.3f} — WEAK. Features need fundamental rework.")
    print(f"  Best model: {best['name']}")
    print(f"  vs mean baseline MAE: {results[-1]['mae'] if results[-1]['name'].startswith('Baseline A') else 'N/A'}")
    print()


if __name__ == "__main__":
    main()
