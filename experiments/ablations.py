#!/usr/bin/env python3
"""Phase 3: Ablation studies a reviewer will demand.

Experiments:
1. Feature group ablation (each group alone / each group removed)
2. Scoring weight sensitivity analysis
3. Learning curve (sample size vs performance)
4. Feature importance (permutation + SHAP if available)
5. Error analysis

Usage:
    python3 experiments/ablations.py
    python3 experiments/ablations.py --use-all
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

# Reuse data loading from model_ablation
sys.path.insert(0, str(Path(__file__).parent))
from model_ablation import FEATURE_GROUPS, FEATURE_NAMES, load_data, loocv_evaluate, scale_preprocess


# ──────────────────────────────────────────────────────────────────
# Best model from Phase 1 (RF on scaled features)
# ──────────────────────────────────────────────────────────────────

def best_model_factory():
    return RandomForestRegressor(
        n_estimators=50, max_depth=3,
        min_samples_leaf=3, random_state=42,
    )


def best_model_factory_gbr():
    return GradientBoostingRegressor(
        n_estimators=50, max_depth=2,
        learning_rate=0.1, subsample=0.8,
        min_samples_leaf=3, random_state=42,
    )


# ──────────────────────────────────────────────────────────────────
# EXPERIMENT 1: Feature Group Ablation
# ──────────────────────────────────────────────────────────────────

def experiment_feature_groups(X, y, model_factory, preprocess=scale_preprocess):
    """Train with each feature group alone and each removed."""
    print(f"\n{'=' * 90}")
    print("EXPERIMENT 1: FEATURE GROUP ABLATION")
    print(f"{'=' * 90}")

    all_indices = list(range(X.shape[1]))
    results = []

    # Full model (baseline)
    res = loocv_evaluate(X, y, model_factory, preprocess)
    res["name"] = "All features (64)"
    results.append(res)
    print(f"  {'All features (64)':<40} ρ={res['spearman_rho']:.3f}  MAE={res['mae']:.3f}")

    # Each group ALONE
    print(f"\n  --- Each group ALONE ---")
    for group_name, indices in FEATURE_GROUPS.items():
        def make_preprocess(idx):
            def pp(Xtr, Xte):
                scaler = StandardScaler()
                return scaler.fit_transform(Xtr[:, idx]), scaler.transform(Xte[:, idx])
            return pp

        res = loocv_evaluate(X, y, model_factory, make_preprocess(indices))
        res["name"] = f"Only {group_name} ({len(indices)} dims)"
        results.append(res)
        print(f"  {res['name']:<40} ρ={res['spearman_rho']:.3f}  MAE={res['mae']:.3f}")

    # Each group REMOVED
    print(f"\n  --- Each group REMOVED ---")
    for group_name, indices in FEATURE_GROUPS.items():
        remaining = [i for i in all_indices if i not in indices]

        def make_preprocess(idx):
            def pp(Xtr, Xte):
                scaler = StandardScaler()
                return scaler.fit_transform(Xtr[:, idx]), scaler.transform(Xte[:, idx])
            return pp

        res = loocv_evaluate(X, y, model_factory, make_preprocess(remaining))
        res["name"] = f"Without {group_name} ({64 - len(indices)} dims)"
        results.append(res)
        print(f"  {res['name']:<40} ρ={res['spearman_rho']:.3f}  MAE={res['mae']:.3f}")

    return results


# ──────────────────────────────────────────────────────────────────
# EXPERIMENT 2: Scoring Weight Sensitivity
# ──────────────────────────────────────────────────────────────────

def experiment_scoring_weights(gt_file):
    """Grid search over capability scoring weights."""
    import json

    print(f"\n{'=' * 90}")
    print("EXPERIMENT 2: SCORING WEIGHT SENSITIVITY")
    print(f"{'=' * 90}")

    # Load ground truth to get success rates and existing capability scores
    with open(gt_file) as f:
        gt = json.load(f)

    # Load cached features to get capability scores for scored datasets
    cache_dir = Path("benchmarks/cached_features")

    # We need to compute capability scores with different weights
    # For now, analyze the current weights by examining the task_primary_score feature
    # and its correlation with ground truth

    # Use the feature vectors directly
    features, targets, labels, types = load_data(profiled_only=False)
    if len(features) == 0:
        print("  No data available for weight sensitivity analysis")
        return []

    # Feature 52 is task_primary_score, 53 is visual_relevance, 54 is coverage_diversity,
    # 55 is data_quality, 59 is volume_score
    feature_map = {
        "task_primary_score (f52)": 52,
        "visual_relevance (f53)": 53,
        "coverage_diversity (f54)": 54,
        "data_quality (f55)": 55,
        "task_volume (f59)": 59,
    }

    print(f"\n  Single-feature correlations with ground truth success rate:")
    for name, idx in feature_map.items():
        vals = features[:, idx]
        # Skip if all zeros
        if np.std(vals) < 1e-8:
            print(f"  {name:<35} ρ=N/A (constant)")
            continue
        rho, p = stats.spearmanr(vals, targets)
        r, _ = stats.pearsonr(vals, targets)
        print(f"  {name:<35} ρ={rho:.3f} (p={p:.4f})  r={r:.3f}")

    # Grid search over weight combinations
    print(f"\n  Grid search over scoring weight combinations:")
    print(f"  (visual_rel, quality, coverage, volume)")

    best_combos = []
    vis_range = [0.1, 0.2, 0.3, 0.4, 0.5]
    qual_range = [0.1, 0.2, 0.3, 0.4, 0.5]
    cov_range = [0.05, 0.1, 0.2, 0.3]
    vol_range = [0.05, 0.1, 0.2]

    for vis in vis_range:
        for qual in qual_range:
            for cov in cov_range:
                for vol in vol_range:
                    if abs(vis + qual + cov + vol - 1.0) > 0.01:
                        continue

                    # Compute weighted score from component features
                    weighted = (vis * features[:, 53] +
                                qual * features[:, 55] +
                                cov * features[:, 54] +
                                vol * features[:, 59])

                    if np.std(weighted) < 1e-8:
                        continue

                    rho, p = stats.spearmanr(weighted, targets)
                    best_combos.append({
                        "weights": (vis, qual, cov, vol),
                        "rho": rho,
                        "p": p,
                    })

    best_combos.sort(key=lambda x: x["rho"], reverse=True)

    print(f"\n  Top 5 weight configurations:")
    for i, combo in enumerate(best_combos[:5]):
        w = combo["weights"]
        print(f"    {i + 1}. vis={w[0]:.2f} qual={w[1]:.2f} cov={w[2]:.2f} vol={w[3]:.2f}  "
              f"ρ={combo['rho']:.3f} (p={combo['p']:.4f})")

    print(f"\n  Current weights: vis=0.35 qual=0.35 cov=0.20 vol=0.10")
    # Find current in results
    for combo in best_combos:
        w = combo["weights"]
        if abs(w[0] - 0.35) < 0.01 and abs(w[1] - 0.35) < 0.01:
            print(f"  Current rank: ρ={combo['rho']:.3f}")
            break

    return best_combos


# ──────────────────────────────────────────────────────────────────
# EXPERIMENT 3: Learning Curve
# ──────────────────────────────────────────────────────────────────

def experiment_learning_curve(X, y, model_factory, preprocess=scale_preprocess, n_repeats=20):
    """Learning curve: performance vs training set size."""
    print(f"\n{'=' * 90}")
    print("EXPERIMENT 3: LEARNING CURVE")
    print(f"{'=' * 90}")

    n = len(y)
    k_values = [k for k in [10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80] if k < n]
    k_values.append(n)  # full dataset (LOOCV)

    results = []
    rng = np.random.RandomState(42)

    for k in k_values:
        if k == n:
            # Full LOOCV
            res = loocv_evaluate(X, y, model_factory, preprocess)
            rho_mean = res["spearman_rho"]
            mae_mean = res["mae"]
            rho_ci = (rho_mean, rho_mean)  # no CI for single run
            mae_ci = (mae_mean, mae_mean)
            rhos = [rho_mean]
            maes = [mae_mean]
        else:
            rhos = []
            maes = []
            for rep in range(n_repeats):
                # Random train/test split
                idx = rng.permutation(n)
                train_idx = idx[:k]
                test_idx = idx[k:]

                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                if preprocess is not None:
                    X_train_p, X_test_p = preprocess(X_train, X_test)
                else:
                    X_train_p, X_test_p = X_train, X_test

                model = model_factory()
                model.fit(X_train_p, y_train)
                predictions = np.clip(model.predict(X_test_p), 0.0, 1.0)

                if len(set(y_test)) > 1 and len(y_test) > 2:
                    rho, _ = stats.spearmanr(predictions, y_test)
                    rhos.append(rho)
                maes.append(float(np.mean(np.abs(predictions - y_test))))

            rho_mean = np.mean(rhos) if rhos else 0.0
            mae_mean = np.mean(maes)
            rho_ci = (np.percentile(rhos, 5), np.percentile(rhos, 95)) if rhos else (0, 0)
            mae_ci = (np.percentile(maes, 5), np.percentile(maes, 95))

        results.append({
            "k": k,
            "rho_mean": rho_mean,
            "rho_ci_low": rho_ci[0],
            "rho_ci_high": rho_ci[1],
            "mae_mean": mae_mean,
            "mae_ci_low": mae_ci[0],
            "mae_ci_high": mae_ci[1],
        })

        ci_str = f"[{rho_ci[0]:.3f}, {rho_ci[1]:.3f}]" if k != n else "LOOCV"
        print(f"  k={k:>3}:  ρ={rho_mean:.3f} {ci_str:<24}  MAE={mae_mean:.3f}")

    # Assess: data-starved or saturated?
    if len(results) >= 3:
        rho_last_3 = [r["rho_mean"] for r in results[-3:]]
        slope = rho_last_3[-1] - rho_last_3[0]
        if slope > 0.03:
            print(f"\n  DIAGNOSIS: Learning curve still CLIMBING (Δρ={slope:+.3f} over last 3 points)")
            print(f"  → DATA-STARVED: more training data will likely improve performance")
        else:
            print(f"\n  DIAGNOSIS: Learning curve FLATTENING (Δρ={slope:+.3f} over last 3 points)")
            print(f"  → SATURATING: more data alone won't help much — need better features")

    return results


# ──────────────────────────────────────────────────────────────────
# EXPERIMENT 4: Feature Importance
# ──────────────────────────────────────────────────────────────────

def experiment_feature_importance(X, y, labels):
    """Permutation importance + SHAP values."""
    from sklearn.inspection import permutation_importance

    print(f"\n{'=' * 90}")
    print("EXPERIMENT 4: FEATURE IMPORTANCE")
    print(f"{'=' * 90}")

    # Train RF on full data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(
        n_estimators=50, max_depth=3,
        min_samples_leaf=3, random_state=42,
    )
    model.fit(X_scaled, y)

    # Permutation importance
    print(f"\n  Permutation importance (100 repeats):")
    result = permutation_importance(
        model, X_scaled, y,
        n_repeats=100, random_state=42,
        scoring="neg_mean_absolute_error",
    )

    importances = result.importances_mean
    sorted_idx = np.argsort(importances)[::-1]

    print(f"\n  {'Rank':>4} {'Feature':<30} {'Importance':>12} {'Std':>10}")
    print(f"  {'─' * 60}")
    for rank, idx in enumerate(sorted_idx[:20]):
        name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"feat_{idx}"
        print(f"  {rank + 1:>4}. {name:<28} {importances[idx]:>12.4f} "
              f"±{result.importances_std[idx]:>8.4f}")

    # RF built-in importance
    print(f"\n  RF Gini importance (top 20):")
    rf_imp = model.feature_importances_
    rf_sorted = np.argsort(rf_imp)[::-1]
    for rank, idx in enumerate(rf_sorted[:20]):
        name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"feat_{idx}"
        print(f"  {rank + 1:>4}. {name:<28} {rf_imp[idx]:.4f}")

    # Try SHAP
    try:
        import shap
        print(f"\n  SHAP values (TreeExplainer):")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        shap_sorted = np.argsort(mean_abs_shap)[::-1]

        for rank, idx in enumerate(shap_sorted[:20]):
            name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"feat_{idx}"
            print(f"  {rank + 1:>4}. {name:<28} |SHAP|={mean_abs_shap[idx]:.4f}")

        # Save SHAP bar chart
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 8))
            top_k = 20
            top_idx = shap_sorted[:top_k][::-1]
            names = [FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"f{i}" for i in top_idx]
            ax.barh(names, mean_abs_shap[top_idx], color="#2563eb", alpha=0.7)
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title("Feature Importance (SHAP, Top 20)")
            plt.tight_layout()
            os.makedirs("docs", exist_ok=True)
            plt.savefig("docs/feature_importance.png", dpi=150, bbox_inches="tight")
            print(f"\n  SHAP plot saved to docs/feature_importance.png")
            plt.close()
        except Exception as e:
            print(f"  Plot failed: {e}")

    except ImportError:
        print(f"\n  SHAP not installed — skipping SHAP analysis")
        print(f"  Install with: pip install shap")

    return importances, sorted_idx


# ──────────────────────────────────────────────────────────────────
# EXPERIMENT 5: Error Analysis
# ──────────────────────────────────────────────────────────────────

def experiment_error_analysis(X, y, labels, types):
    """Analyze patterns in prediction errors."""
    print(f"\n{'=' * 90}")
    print("EXPERIMENT 5: ERROR ANALYSIS")
    print(f"{'=' * 90}")

    # Get LOOCV predictions from best model
    res = loocv_evaluate(X, y, best_model_factory, scale_preprocess)
    predictions = res["predictions"]
    residuals = predictions - y

    print(f"\n  Residual statistics:")
    print(f"    Mean: {np.mean(residuals):+.4f}")
    print(f"    Median: {np.median(residuals):+.4f}")
    print(f"    Std: {np.std(residuals):.4f}")
    print(f"    Min: {np.min(residuals):+.4f}")
    print(f"    Max: {np.max(residuals):+.4f}")

    # Is there systematic bias?
    t_stat, t_p = stats.ttest_1samp(residuals, 0.0)
    print(f"\n  One-sample t-test (H0: mean residual = 0):")
    print(f"    t = {t_stat:.3f}, p = {t_p:.4f}")
    if t_p < 0.05:
        print(f"    → SIGNIFICANT systematic bias detected!")
    else:
        print(f"    → No significant systematic bias")

    # Correlation with dataset size
    log_episodes = X[:, 44]  # scale_log_episodes
    rho_size, p_size = stats.spearmanr(np.abs(residuals), log_episodes)
    print(f"\n  |Error| vs log(episodes): ρ={rho_size:.3f} (p={p_size:.4f})")
    if p_size < 0.05:
        print(f"    → Errors ARE correlated with dataset size")
    else:
        print(f"    → Errors are NOT correlated with dataset size")

    # Correlation with target value (do we under-predict high and over-predict low?)
    rho_target, p_target = stats.spearmanr(residuals, y)
    print(f"\n  Residual vs actual success rate: ρ={rho_target:.3f} (p={p_target:.4f})")
    if rho_target < -0.3 and p_target < 0.05:
        print(f"    → REGRESSION TO MEAN: underpredicts high values, overpredicts low values")
    elif rho_target > 0.3 and p_target < 0.05:
        print(f"    → AMPLIFICATION: overpredicts high values, underpredicts low values")
    else:
        print(f"    → No strong relationship between residuals and target")

    # Error by data type (profiled vs estimated)
    type_arr = np.array(types)
    for dtype in np.unique(type_arr):
        mask = type_arr == dtype
        if mask.sum() < 3:
            continue
        subset_residuals = residuals[mask]
        subset_abs_err = np.abs(subset_residuals)
        print(f"\n  Type '{dtype}' (n={mask.sum()}):")
        print(f"    Mean |error|: {np.mean(subset_abs_err):.3f}")
        print(f"    Mean residual: {np.mean(subset_residuals):+.3f}")

    # 10 worst predictions
    print(f"\n  10 WORST PREDICTIONS:")
    worst_idx = np.argsort(np.abs(residuals))[-10:][::-1]
    print(f"  {'Dataset':<42} {'Actual':>7} {'Pred':>7} {'Error':>7} {'Type':>12}")
    print(f"  {'─' * 75}")
    for idx in worst_idx:
        print(f"  {labels[idx]:<40} {y[idx]:>7.3f} {predictions[idx]:>7.3f} "
              f"{residuals[idx]:>+7.3f} {types[idx]:>12}")

    # Fraction within tolerance bands
    for tol in [0.05, 0.10, 0.15, 0.20, 0.25]:
        frac = np.mean(np.abs(residuals) < tol)
        print(f"\n  Within ±{tol:.2f}: {frac * 100:.1f}% ({int(frac * len(y))}/{len(y)})")

    return residuals


# ──────────────────────────────────────────────────────────────────
# Save all results to markdown
# ──────────────────────────────────────────────────────────────────

def save_ablation_markdown(
    fg_results, weight_results, lc_results, importances, err_residuals,
    targets, labels, types,
    output_path="docs/ablation_results.md",
):
    """Write publication-ready ablation results."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write("# Ablation Study Results\n\n")

        # Experiment 1
        f.write("## Experiment 1: Feature Group Ablation\n\n")
        f.write("| Configuration | Spearman ρ | MAE | RMSE |\n")
        f.write("|--------------|-----------|-----|------|\n")
        for r in fg_results:
            f.write(f"| {r['name']} | {r['spearman_rho']:.3f} | {r['mae']:.3f} | {r['rmse']:.3f} |\n")

        # Experiment 2
        f.write("\n## Experiment 2: Scoring Weight Sensitivity\n\n")
        if weight_results:
            f.write("| Rank | Visual Rel. | Quality | Coverage | Volume | Spearman ρ |\n")
            f.write("|------|------------|---------|----------|--------|----------|\n")
            for i, combo in enumerate(weight_results[:10]):
                w = combo["weights"]
                f.write(f"| {i + 1} | {w[0]:.2f} | {w[1]:.2f} | {w[2]:.2f} | {w[3]:.2f} | "
                        f"{combo['rho']:.3f} |\n")

        # Experiment 3
        f.write("\n## Experiment 3: Learning Curve\n\n")
        f.write("| Training Size | Spearman ρ (mean) | 90% CI | MAE |\n")
        f.write("|--------------|------------------|--------|-----|\n")
        for r in lc_results:
            ci = f"[{r['rho_ci_low']:.3f}, {r['rho_ci_high']:.3f}]"
            f.write(f"| {r['k']} | {r['rho_mean']:.3f} | {ci} | {r['mae_mean']:.3f} |\n")

        # Experiment 5
        f.write("\n## Experiment 5: Error Analysis\n\n")
        f.write(f"- Mean residual: {np.mean(err_residuals):+.4f}\n")
        f.write(f"- Std residual: {np.std(err_residuals):.4f}\n")
        for tol in [0.10, 0.15, 0.20]:
            frac = np.mean(np.abs(err_residuals) < tol)
            f.write(f"- Within ±{tol:.2f}: {frac * 100:.1f}%\n")

    print(f"\nAll ablation results saved to {output_path}")


# ──────────────────────────────────────────────────────────────────
# Learning curve plot
# ──────────────────────────────────────────────────────────────────

def plot_learning_curve(lc_results, output_path="docs/learning_curve.png"):
    """Save learning curve plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ks = [r["k"] for r in lc_results]
        rhos = [r["rho_mean"] for r in lc_results]
        rho_lows = [r["rho_ci_low"] for r in lc_results]
        rho_highs = [r["rho_ci_high"] for r in lc_results]
        maes = [r["mae_mean"] for r in lc_results]
        mae_lows = [r["mae_ci_low"] for r in lc_results]
        mae_highs = [r["mae_ci_high"] for r in lc_results]

        ax1.plot(ks, rhos, "o-", color="#2563eb", linewidth=2, markersize=6)
        ax1.fill_between(ks, rho_lows, rho_highs, alpha=0.2, color="#2563eb")
        ax1.set_xlabel("Training Set Size", fontsize=12)
        ax1.set_ylabel("Spearman ρ", fontsize=12)
        ax1.set_title("Learning Curve — Correlation", fontsize=12)
        ax1.axhline(0.7, color="green", linestyle="--", alpha=0.3, label="Target ρ=0.7")
        ax1.legend()
        ax1.grid(True, alpha=0.2)

        ax2.plot(ks, maes, "o-", color="#dc2626", linewidth=2, markersize=6)
        ax2.fill_between(ks, mae_lows, mae_highs, alpha=0.2, color="#dc2626")
        ax2.set_xlabel("Training Set Size", fontsize=12)
        ax2.set_ylabel("MAE", fontsize=12)
        ax2.set_title("Learning Curve — Error", fontsize=12)
        ax2.grid(True, alpha=0.2)

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\nLearning curve saved to {output_path}")
        plt.close()
    except ImportError:
        print("matplotlib not available — skipping plot")


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Ablation studies")
    parser.add_argument("--use-all", action="store_true")
    parser.add_argument("--cache-dir", default="benchmarks/cached_features")
    parser.add_argument("--gt-file", default="orbit/benchmarks/ground_truth_comprehensive.json")
    args = parser.parse_args()

    features, targets, labels, types = load_data(
        cache_dir=args.cache_dir,
        gt_file=args.gt_file,
        profiled_only=not args.use_all,
    )

    print(f"Loaded {len(features)} datasets")

    # Choose best model based on Phase 1 results
    n = len(features)
    if n >= 50:
        model_factory = best_model_factory_gbr
        print(f"Using GBR (best for n≥50)")
    else:
        model_factory = best_model_factory
        print(f"Using RF (best for n<50)")

    # Experiment 1: Feature groups
    fg_results = experiment_feature_groups(features, targets, model_factory)

    # Experiment 2: Scoring weights
    weight_results = experiment_scoring_weights(args.gt_file)

    # Experiment 3: Learning curve
    lc_results = experiment_learning_curve(features, targets, model_factory)

    # Experiment 4: Feature importance
    importances, sorted_idx = experiment_feature_importance(features, targets, labels)

    # Experiment 5: Error analysis
    residuals = experiment_error_analysis(features, targets, labels, types)

    # Save results
    save_ablation_markdown(
        fg_results, weight_results, lc_results,
        importances, residuals, targets, labels, types,
    )

    # Plot learning curve
    plot_learning_curve(lc_results)


if __name__ == "__main__":
    main()
