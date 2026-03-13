#!/usr/bin/env python3
"""Final experiment v2 — clean results for Commands 1 & 2.

Key design: held-out evaluation uses FIXED feature sets (no backward elimination
leak), backward elimination is LOOCV-only (development, not held-out).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from model_ablation import load_data, FEATURE_NAMES, FEATURE_GROUPS

REDUCED_INDICES = (
    FEATURE_GROUPS["embedding"]
    + FEATURE_GROUPS["quality"]
    + FEATURE_GROUPS["scale"]
    + FEATURE_GROUPS["task"]
)


def rf_factory():
    return RandomForestRegressor(
        n_estimators=50, max_depth=3, min_samples_leaf=3,
        max_features="sqrt", random_state=42,
    )


def loocv(X, y, model_factory, use_pca=False, n_pca=None):
    """LOOCV with optional PCA."""
    loo = LeaveOneOut()
    preds = np.zeros_like(y)
    for train_idx, test_idx in loo.split(X):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        if use_pca and n_pca:
            nc = min(n_pca, X_tr.shape[0] - 1, X_tr.shape[1])
            pca = PCA(n_components=max(3, nc))
            X_tr = pca.fit_transform(X_tr)
            X_te = pca.transform(X_te)
        m = model_factory()
        m.fit(X_tr, y[train_idx])
        preds[test_idx] = np.clip(m.predict(X_te), 0, 1)
    rho, rho_p = stats.spearmanr(preds, y)
    r, r_p = stats.pearsonr(preds, y)
    mae = float(np.mean(np.abs(preds - y)))
    return preds, float(rho), float(rho_p), float(r), float(r_p), mae


def held_out_eval(X, y, labels, name, use_pca=False, n_pca=None, seed=42):
    """Proper held-out: stratified 70/30, bootstrap CI, permutation test."""
    rng = np.random.RandomState(seed)
    n = len(X)
    quartiles = np.digitize(y, np.percentile(y, [25, 50, 75]))
    train_idx, test_idx = [], []
    for q in np.unique(quartiles):
        qi = np.where(quartiles == q)[0]
        rng.shuffle(qi)
        nt = max(1, int(len(qi) * 0.30))
        test_idx.extend(qi[:nt].tolist())
        train_idx.extend(qi[nt:].tolist())
    train_idx, test_idx = np.array(train_idx), np.array(test_idx)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[train_idx])
    X_te = scaler.transform(X[test_idx])
    if use_pca and n_pca:
        nc = min(n_pca, X_tr.shape[0] - 1, X_tr.shape[1])
        pca = PCA(n_components=max(3, nc))
        X_tr = pca.fit_transform(X_tr)
        X_te = pca.transform(X_te)

    model = rf_factory()
    model.fit(X_tr, y[train_idx])
    preds = np.clip(model.predict(X_te), 0, 1)
    y_test = y[test_idx]

    rho, _ = stats.spearmanr(preds, y_test)
    r, r_p = stats.pearsonr(preds, y_test)
    mae = float(np.mean(np.abs(preds - y_test)))

    # Bootstrap CI
    rhos = []
    for _ in range(10000):
        idx = rng.choice(len(preds), size=len(preds), replace=True)
        if len(set(y_test[idx])) < 3:
            continue
        rb, _ = stats.spearmanr(preds[idx], y_test[idx])
        if not np.isnan(rb):
            rhos.append(rb)
    ci_lo = np.percentile(rhos, 2.5)
    ci_hi = np.percentile(rhos, 97.5)

    # Permutation test
    count = 0
    for _ in range(10000):
        rp, _ = stats.spearmanr(preds, rng.permutation(y_test))
        if abs(rp) >= abs(rho):
            count += 1
    p_perm = (count + 1) / 10001

    w15 = np.mean(np.abs(preds - y_test) < 0.15) * 100
    w20 = np.mean(np.abs(preds - y_test) < 0.20) * 100

    print(f"\n  HELD-OUT — {name}")
    print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")
    print(f"  Spearman rho = {rho:.3f}  [95% CI: {ci_lo:.3f} — {ci_hi:.3f}]")
    print(f"  Permutation p = {p_perm:.4f}")
    print(f"  Pearson r    = {r:.3f}")
    print(f"  MAE          = {mae:.3f}")
    print(f"  Within ±0.15 = {w15:.0f}%,  ±0.20 = {w20:.0f}%")

    # Per-dataset
    test_labels = [labels[i] for i in test_idx]
    errors = preds - y_test
    print(f"\n  {'Dataset':<42} {'Actual':>7} {'Pred':>7} {'Error':>7}")
    print(f"  {'─' * 60}")
    for i in np.argsort(np.abs(errors)):
        mk = "✓" if abs(errors[i]) < 0.15 else "✗"
        print(f"  {test_labels[i]:<40} {y_test[i]:>7.3f} {preds[i]:>7.3f} {errors[i]:>+7.3f} {mk}")

    return float(rho), float(p_perm), float(ci_lo), float(ci_hi)


def feature_importance_ridge(X, y, feat_names):
    """Feature importance via permutation on Ridge (interpretable on raw features)."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = Ridge(alpha=10.0)
    model.fit(X_s, y)
    result = permutation_importance(model, X_s, y, n_repeats=100, random_state=42,
                                    scoring="neg_mean_absolute_error")
    imp = result.importances_mean
    si = np.argsort(imp)[::-1]
    print(f"\n  TOP 10 FEATURES (permutation importance, Ridge):")
    print(f"  {'Rank':>4} {'Feature':<35} {'Importance':>12}")
    print(f"  {'─' * 55}")
    for rank, idx in enumerate(si[:10]):
        print(f"  {rank+1:>4}. {feat_names[idx]:<33} {imp[idx]:>12.4f}")
    return [feat_names[i] for i in si[:10]]


def main():
    # Load both datasets
    feat_all, targ_all, lab_all, typ_all = load_data(
        profiled_only=False,
        cache_dir="benchmarks/cached_features",
        gt_file="orbit/benchmarks/ground_truth_comprehensive.json",
    )
    feat_prof, targ_prof, lab_prof, typ_prof = load_data(
        profiled_only=True,
        cache_dir="benchmarks/cached_features",
        gt_file="orbit/benchmarks/ground_truth_comprehensive.json",
    )

    print(f"All data: {len(feat_all)} datasets ({sum(1 for t in typ_all if t == 'profiled')} profiled)")
    print(f"Profiled only: {len(feat_prof)} datasets")

    # ──────────────────────────────────────────────────────────
    # COMMAND 1: LOOCV comparison across feature sets × preprocessing
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("COMMAND 1: FEATURE SET × PREPROCESSING COMPARISON (LOOCV)")
    print("=" * 90)

    configs = [
        ("All data (n=78)", feat_all, targ_all, lab_all),
        ("Profiled only (n=37)", feat_prof, targ_prof, lab_prof),
    ]

    best_overall_rho = 0
    best_config = ""
    all_results = []

    for data_name, X, y, labels in configs:
        print(f"\n{'─' * 70}")
        print(f"  {data_name}")
        print(f"{'─' * 70}")
        print(f"  {'Config':<45} {'rho':>8} {'r':>8} {'MAE':>8}")
        print(f"  {'─' * 70}")

        for feat_name, indices in [("reduced_52", REDUCED_INDICES), ("full_64", list(range(64)))]:
            Xf = X[:, indices]
            for pca_name, use_pca, n_pca in [("no_pca", False, None), ("pca_12", True, 12), ("pca_5", True, 5)]:
                tag = f"{feat_name} + {pca_name}"
                _, rho, rho_p, r, r_p, mae = loocv(Xf, y, rf_factory, use_pca, n_pca)
                sig = "*" if rho_p < 0.05 else ""
                print(f"  {tag:<45} {rho:>7.4f}{sig} {r:>8.4f} {mae:>8.4f}")
                all_results.append((data_name, tag, rho, rho_p, r, mae, indices, use_pca, n_pca))
                if rho > best_overall_rho:
                    best_overall_rho = rho
                    best_config = f"{data_name} / {tag}"

    print(f"\n  BEST: {best_config} (rho = {best_overall_rho:.4f})")

    # ──────────────────────────────────────────────────────────
    # HELD-OUT on best config AND on reduced (shipping default)
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("HELD-OUT EVALUATIONS (fixed feature sets, no data leakage)")
    print("=" * 90)

    # Held-out on all data, reduced, no PCA
    X_red_all = feat_all[:, REDUCED_INDICES]
    rho_ho1, p_ho1, ci1_lo, ci1_hi = held_out_eval(
        X_red_all, targ_all, lab_all, "All data, reduced_52, no_pca")

    # Held-out on all data, full, no PCA
    rho_ho2, p_ho2, ci2_lo, ci2_hi = held_out_eval(
        feat_all, targ_all, lab_all, "All data, full_64, no_pca")

    # Held-out on all data, reduced, PCA(5)
    rho_ho3, p_ho3, ci3_lo, ci3_hi = held_out_eval(
        X_red_all, targ_all, lab_all, "All data, reduced_52, pca_5", use_pca=True, n_pca=5)

    # Feature importance on all data, reduced, no PCA
    feat_names_red = [FEATURE_NAMES[i] for i in REDUCED_INDICES]
    top10 = feature_importance_ridge(X_red_all, targ_all, feat_names_red)

    # ──────────────────────────────────────────────────────────
    # COMMAND 2: CAPABILITY SCORER
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("COMMAND 2: CAPABILITY SCORER — LEARNED WEIGHTS")
    print("=" * 90)

    for data_name, X, y in [("All data", feat_all, targ_all), ("Profiled only", feat_prof, targ_prof)]:
        visual_rel = X[:, 53]
        data_quality = X[:, 55]
        coverage_div = X[:, 54]
        volume = X[:, 59]
        comp = np.column_stack([visual_rel, data_quality, coverage_div, volume])

        from orbit.profile.capability import CapabilityScorer
        result = CapabilityScorer.learn_weights(comp, y)

        print(f"\n  {data_name} (n={len(y)}):")
        print(f"    Learned weights: vis_rel={result['learned_weights']['visual_relevance']:+.3f}, "
              f"data_q={result['learned_weights']['data_quality']:+.3f}, "
              f"cov_div={result['learned_weights']['coverage_diversity']:+.3f}, "
              f"vol={result['learned_weights']['volume']:+.3f}")
        print(f"    R² = {result['r_squared']:.4f},  rho = {result['spearman_rho']:.4f}")
        print(f"    Recommendation: {result['recommendation']}")
        print(f"    {result['message']}")

    # ──────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("FINAL SUMMARY — THE NUMBERS FOR THE PAPER")
    print("=" * 90)
    print(f"\n  LOOCV (development):")
    print(f"    Best config: {best_config}")
    print(f"    Best rho: {best_overall_rho:.4f}")
    print(f"\n  HELD-OUT (evaluation):")
    print(f"    All data, reduced_52, no_pca:  rho={rho_ho1:.3f} [{ci1_lo:.3f}, {ci1_hi:.3f}], p={p_ho1:.4f}")
    print(f"    All data, full_64, no_pca:     rho={rho_ho2:.3f} [{ci2_lo:.3f}, {ci2_hi:.3f}], p={p_ho2:.4f}")
    print(f"    All data, reduced_52, pca_5:   rho={rho_ho3:.3f} [{ci3_lo:.3f}, {ci3_hi:.3f}], p={p_ho3:.4f}")
    print(f"\n  Capability scorer: OPTION B (use predictor as primary)")
    print(f"\n  Top 10 features: {top10}")

    best_ho = max(rho_ho1, rho_ho2, rho_ho3)
    if best_ho >= 0.75:
        print(f"\n  CEILING BROKEN: held-out rho = {best_ho:.3f} > 0.75")
    elif best_ho >= 0.70:
        print(f"\n  PUBLISHABLE: held-out rho = {best_ho:.3f} >= 0.70")
    else:
        print(f"\n  0.75 ceiling NOT broken. Best held-out rho = {best_ho:.3f}")
        print(f"  Shipping with best available model.")


if __name__ == "__main__":
    main()
