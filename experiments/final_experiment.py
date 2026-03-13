#!/usr/bin/env python3
"""Final experiment: Feature set comparison + capability scorer + held-out evaluation.

Commands 1 & 2 from the shipping checklist.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from model_ablation import load_data, FEATURE_NAMES, FEATURE_GROUPS

# ──────────────────────────────────────────────────────────────────
# Feature set definitions
# ──────────────────────────────────────────────────────────────────

REDUCED_INDICES = (
    FEATURE_GROUPS["embedding"]
    + FEATURE_GROUPS["quality"]
    + FEATURE_GROUPS["scale"]
    + FEATURE_GROUPS["task"]
)  # 52 features (no action)

FULL_INDICES = list(range(64))  # all 64


def loocv(X, y, model_factory, use_pca=True):
    """LOOCV returning predictions, rho, r, MAE.  Scale→PCA→model per fold."""
    loo = LeaveOneOut()
    preds = np.zeros_like(y)
    for train_idx, test_idx in loo.split(X):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        if use_pca:
            n_comp = min(12, X_tr.shape[0] - 1, X_tr.shape[1])
            pca = PCA(n_components=max(3, n_comp))
            X_tr = pca.fit_transform(X_tr)
            X_te = pca.transform(X_te)
        model = model_factory()
        model.fit(X_tr, y[train_idx])
        preds[test_idx] = np.clip(model.predict(X_te), 0, 1)
    rho, rho_p = stats.spearmanr(preds, y)
    r, r_p = stats.pearsonr(preds, y)
    mae = float(np.mean(np.abs(preds - y)))
    return preds, float(rho), float(rho_p), float(r), float(r_p), mae


def rf_factory():
    return RandomForestRegressor(
        n_estimators=50, max_depth=3, min_samples_leaf=3,
        max_features="sqrt", random_state=42,
    )


# ──────────────────────────────────────────────────────────────────
# Command 1: Feature set comparison
# ──────────────────────────────────────────────────────────────────

def run_command_1(features, targets, labels):
    print("\n" + "=" * 90)
    print("COMMAND 1: FEATURE SET COMPARISON")
    print("=" * 90)

    # --- Reduced (52 features) ---
    X_reduced = features[:, REDUCED_INDICES]
    preds_r, rho_r, rho_p_r, r_r, r_p_r, mae_r = loocv(X_reduced, targets, rf_factory)
    print(f"\n  REDUCED (52 features, no action):")
    print(f"    Spearman rho = {rho_r:.4f} (p={rho_p_r:.4f})")
    print(f"    Pearson r    = {r_r:.4f} (p={r_p_r:.4f})")
    print(f"    MAE          = {mae_r:.4f}")

    # --- Full (64 features) ---
    X_full = features[:, FULL_INDICES]
    preds_f, rho_f, rho_p_f, r_f, r_p_f, mae_f = loocv(X_full, targets, rf_factory)
    print(f"\n  FULL (64 features, with action):")
    print(f"    Spearman rho = {rho_f:.4f} (p={rho_p_f:.4f})")
    print(f"    Pearson r    = {r_f:.4f} (p={r_p_f:.4f})")
    print(f"    MAE          = {mae_f:.4f}")

    # --- Extended features note ---
    print(f"\n  EXTENDED (52 reduced + 9 new = 61 features):")
    print(f"    NOTE: Extended features require re-profiling all datasets from raw data")
    print(f"    (FAISS indices needed for temporal/geometry/cross-episode features).")
    print(f"    Only 7 of 37 datasets have cached raw profiles.")
    print(f"    Cannot extract extended features from cached 64-dim vectors.")
    print(f"    Proceeding with backward elimination on reduced set instead.")

    # --- Backward elimination on reduced features ---
    print(f"\n{'─' * 70}")
    print(f"BACKWARD ELIMINATION on reduced (52 features)")
    print(f"{'─' * 70}")

    reduced_names = [FEATURE_NAMES[i] for i in REDUCED_INDICES]
    best_indices = list(range(len(REDUCED_INDICES)))  # indices into X_reduced
    best_rho = rho_r
    best_set_name = "reduced_52"
    elimination_log = []

    round_num = 0
    while len(best_indices) > 5:
        round_num += 1
        best_removal = None
        best_new_rho = best_rho

        for idx_to_remove in best_indices:
            trial_indices = [i for i in best_indices if i != idx_to_remove]
            X_trial = X_reduced[:, trial_indices]
            _, trial_rho, _, _, _, _ = loocv(X_trial, targets, rf_factory)
            if trial_rho > best_new_rho:
                best_new_rho = trial_rho
                best_removal = idx_to_remove

        if best_removal is None:
            print(f"  Round {round_num}: No improvement possible. Stopping.")
            break

        removed_name = reduced_names[best_removal]
        best_indices = [i for i in best_indices if i != best_removal]
        improvement = best_new_rho - best_rho
        elimination_log.append((removed_name, best_new_rho, improvement))
        best_rho = best_new_rho
        print(f"  Round {round_num}: Removed '{removed_name}' → rho = {best_rho:.4f} (+{improvement:.4f}), {len(best_indices)} features left")

        if improvement < 0.001:
            print(f"  Marginal improvement. Stopping.")
            break

    # Final best_subset evaluation
    X_best = X_reduced[:, best_indices]
    preds_b, rho_b, rho_p_b, r_b, r_p_b, mae_b = loocv(X_best, targets, rf_factory)
    best_subset_names = [reduced_names[i] for i in best_indices]

    print(f"\n  BEST SUBSET ({len(best_indices)} features):")
    print(f"    Spearman rho = {rho_b:.4f} (p={rho_p_b:.4f})")
    print(f"    Pearson r    = {r_b:.4f} (p={r_p_b:.4f})")
    print(f"    MAE          = {mae_b:.4f}")
    print(f"    Features: {best_subset_names}")

    # --- Summary table ---
    print(f"\n{'=' * 90}")
    print(f"COMMAND 1 SUMMARY")
    print(f"{'=' * 90}")
    print(f"  {'Feature Set':<25} {'Dims':>5} {'rho':>8} {'r':>8} {'MAE':>8}")
    print(f"  {'─' * 55}")
    print(f"  {'Reduced':<25} {52:>5} {rho_r:>8.4f} {r_r:>8.4f} {mae_r:>8.4f}")
    print(f"  {'Full':<25} {64:>5} {rho_f:>8.4f} {r_f:>8.4f} {mae_f:>8.4f}")
    print(f"  {'Best Subset':<25} {len(best_indices):>5} {rho_b:>8.4f} {r_b:>8.4f} {mae_b:>8.4f}")

    # Determine winner
    results = [
        ("reduced", rho_r, 52, REDUCED_INDICES, preds_r),
        ("full", rho_f, 64, FULL_INDICES, preds_f),
        ("best_subset", rho_b, len(best_indices), [REDUCED_INDICES[i] for i in best_indices], preds_b),
    ]
    results.sort(key=lambda x: x[1], reverse=True)
    winner_name, winner_rho, winner_dims, winner_indices, winner_preds = results[0]
    print(f"\n  WINNER: {winner_name} (rho={winner_rho:.4f}, {winner_dims} dims)")

    # --- Held-out evaluation on winner ---
    if winner_rho > rho_r or True:  # always run held-out
        print(f"\n{'=' * 90}")
        print(f"HELD-OUT EVALUATION — {winner_name} ({winner_dims} features)")
        print(f"{'=' * 90}")
        X_winner = features[:, winner_indices]
        run_held_out(X_winner, targets, labels, winner_name)

    # --- Feature importance ---
    print(f"\n{'=' * 90}")
    print(f"TOP 10 FEATURES BY PERMUTATION IMPORTANCE — {winner_name}")
    print(f"{'=' * 90}")
    run_feature_importance(features[:, winner_indices], targets,
                           [FEATURE_NAMES[i] for i in winner_indices])

    return winner_name, winner_rho, winner_indices


def run_held_out(X, y, labels, feature_set_name):
    """Stratified 70/30 split, bootstrap CI, permutation test."""
    n = len(X)
    rng = np.random.RandomState(42)

    # Stratified split
    quartiles = np.digitize(y, np.percentile(y, [25, 50, 75]))
    train_idx, test_idx = [], []
    for q in np.unique(quartiles):
        q_indices = np.where(quartiles == q)[0]
        rng.shuffle(q_indices)
        n_q_test = max(1, int(len(q_indices) * 0.30))
        test_idx.extend(q_indices[:n_q_test].tolist())
        train_idx.extend(q_indices[n_q_test:].tolist())
    train_idx, test_idx = np.array(train_idx), np.array(test_idx)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    n_comp = min(12, X_train_s.shape[0] - 1, X_train_s.shape[1])
    pca = PCA(n_components=max(3, n_comp))
    X_train_s = pca.fit_transform(X_train_s)
    X_test_s = pca.transform(X_test_s)

    model = rf_factory()
    model.fit(X_train_s, y_train)
    preds = np.clip(model.predict(X_test_s), 0, 1)

    rho_obs, _ = stats.spearmanr(preds, y_test)
    r_obs, r_p = stats.pearsonr(preds, y_test)
    mae = float(np.mean(np.abs(preds - y_test)))
    rmse = float(np.sqrt(np.mean((preds - y_test) ** 2)))

    # Bootstrap CI for rho
    rhos = []
    for _ in range(10000):
        idx = rng.choice(len(preds), size=len(preds), replace=True)
        if len(set(y_test[idx])) < 3:
            continue
        r_boot, _ = stats.spearmanr(preds[idx], y_test[idx])
        if not np.isnan(r_boot):
            rhos.append(r_boot)
    ci_low = np.percentile(rhos, 2.5)
    ci_high = np.percentile(rhos, 97.5)

    # Permutation test
    count_extreme = 0
    for _ in range(10000):
        shuffled = rng.permutation(y_test)
        rho_perm, _ = stats.spearmanr(preds, shuffled)
        if abs(rho_perm) >= abs(rho_obs):
            count_extreme += 1
    p_perm = (count_extreme + 1) / 10001

    within_15 = np.mean(np.abs(preds - y_test) < 0.15) * 100
    within_20 = np.mean(np.abs(preds - y_test) < 0.20) * 100

    print(f"\n  Train: {len(train_idx)}, Test: {len(test_idx)}")
    print(f"  y_test: mean={y_test.mean():.3f}, std={y_test.std():.3f}")
    print(f"\n  Spearman rho = {rho_obs:.3f}  [95% CI: {ci_low:.3f} — {ci_high:.3f}]")
    print(f"  Permutation p = {p_perm:.4f}")
    print(f"  Pearson r    = {r_obs:.3f} (p={r_p:.4f})")
    print(f"  MAE          = {mae:.3f}")
    print(f"  RMSE         = {rmse:.3f}")
    print(f"  Within ±0.15 = {within_15:.0f}%")
    print(f"  Within ±0.20 = {within_20:.0f}%")

    # Per-dataset
    test_labels = [labels[i] for i in test_idx]
    errors = preds - y_test
    print(f"\n  {'Dataset':<42} {'Actual':>7} {'Pred':>7} {'Error':>7}")
    print(f"  {'─' * 60}")
    for i in np.argsort(np.abs(errors)):
        marker = "✓" if abs(errors[i]) < 0.15 else "✗"
        print(f"  {test_labels[i]:<40} {y_test[i]:>7.3f} {preds[i]:>7.3f} {errors[i]:>+7.3f} {marker}")

    # Verdict
    print(f"\n  {'=' * 60}")
    if rho_obs >= 0.75:
        print(f"  VERDICT: rho = {rho_obs:.3f} ≥ 0.75 → CEILING BROKEN!")
    elif rho_obs >= 0.70:
        print(f"  VERDICT: rho = {rho_obs:.3f} ≥ 0.70 → PUBLISHABLE")
    elif rho_obs >= 0.50:
        print(f"  VERDICT: rho = {rho_obs:.3f} ≥ 0.50 → USABLE")
    else:
        print(f"  VERDICT: rho = {rho_obs:.3f} → needs more data")
    print(f"  {'=' * 60}")

    return rho_obs, p_perm


def run_feature_importance(X, y, feature_names):
    """Permutation importance on full training data (no PCA — raw features)."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    # Use Ridge for interpretable feature importance on original features
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=10.0)
    model.fit(X_s, y)

    result = permutation_importance(
        model, X_s, y, n_repeats=100, random_state=42,
        scoring="neg_mean_absolute_error",
    )
    importances = result.importances_mean
    sorted_idx = np.argsort(importances)[::-1]

    # Extended feature names for checking Phase 7
    EXTENDED_NAMES = [
        "temp_state_autocorrelation", "temp_coverage_rate", "temp_action_temporal_entropy",
        "cross_inter_episode_overlap", "cross_episode_diversity_index",
        "geom_intrinsic_dimensionality", "geom_isotropy", "geom_hub_score",
        "adv_coverage_action_ratio",
    ]

    print(f"\n  {'Rank':>4} {'Feature':<35} {'Importance':>12}")
    print(f"  {'─' * 55}")
    for rank, idx in enumerate(sorted_idx[:10]):
        name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
        is_new = name in EXTENDED_NAMES
        marker = " ★ NEW" if is_new else ""
        print(f"  {rank+1:>4}. {name:<33} {importances[idx]:>12.4f}{marker}")

    # Check if any extended features are in top 10
    top10_names = [feature_names[i] for i in sorted_idx[:10]]
    new_in_top10 = [n for n in top10_names if n in EXTENDED_NAMES]
    if new_in_top10:
        print(f"\n  Phase 7 features in top 10: {new_in_top10}")
    else:
        print(f"\n  No Phase 7 extended features in top 10 (they weren't available for this run)")


# ──────────────────────────────────────────────────────────────────
# Command 2: Capability scorer learned weights
# ──────────────────────────────────────────────────────────────────

def run_command_2(features, targets, labels):
    print("\n" + "=" * 90)
    print("COMMAND 2: CAPABILITY SCORER — LEARNED WEIGHTS")
    print("=" * 90)

    # Extract component scores from feature vector
    # Index 53 = visual_relevance, 55 = data_quality, 54 = coverage_diversity, 59 = volume
    visual_rel = features[:, 53]
    data_quality = features[:, 55]
    coverage_div = features[:, 54]
    volume = features[:, 59]

    component_scores = np.column_stack([visual_rel, data_quality, coverage_div, volume])

    # Run learn_weights
    from orbit.profile.capability import CapabilityScorer
    result = CapabilityScorer.learn_weights(component_scores, targets)

    print(f"\n  Samples: {result['n_samples']}")
    print(f"\n  Learned weights:")
    for k, v in result['learned_weights'].items():
        print(f"    {k:<25}: {v:>+8.4f}")
    print(f"    intercept:                {result['intercept']:>+8.4f}")
    print(f"\n  R-squared:    {result['r_squared']:.4f}")
    print(f"  Spearman rho: {result['spearman_rho']:.4f} (p={result['spearman_p']:.4f})")
    print(f"\n  Recommendation: {result['recommendation']}")
    print(f"  Message: {result['message']}")

    if result['recommendation'] == 'option_b':
        print(f"\n  → OPTION B: Kill the scorer as primary output.")
        print(f"    The predictor becomes the headline. Capability score is a")
        print(f"    human-readable summary only.")
        print(f"    Action: Update report card to demote capability score.")
    else:
        print(f"\n  → OPTION A: Use learned weights.")
        print(f"    Replace hand-tuned weights with learned weights.")

    return result


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    # ---- Run on ALL data (n=90, profiled + estimated) ----
    print("\n" + "#" * 90)
    print("# RUN A: ALL DATA (profiled + estimated)")
    print("#" * 90)
    features_all, targets_all, labels_all, types_all = load_data(
        cache_dir="benchmarks/cached_features",
        gt_file="orbit/benchmarks/ground_truth_comprehensive.json",
        profiled_only=False,
    )
    n_profiled_all = sum(1 for t in types_all if t == "profiled")
    print(f"Loaded {len(features_all)} datasets ({n_profiled_all} profiled, "
          f"{len(features_all) - n_profiled_all} estimated)")
    print(f"Target distribution: mean={targets_all.mean():.3f}, std={targets_all.std():.3f}")

    winner_all, rho_all, indices_all = run_command_1(features_all, targets_all, labels_all)

    # ---- Run on PROFILED only (n=37) ----
    print("\n" + "#" * 90)
    print("# RUN B: PROFILED ONLY (n=37)")
    print("#" * 90)
    features, targets, labels, types = load_data(
        cache_dir="benchmarks/cached_features",
        gt_file="orbit/benchmarks/ground_truth_comprehensive.json",
        profiled_only=True,
    )
    print(f"Loaded {len(features)} datasets (all profiled)")
    print(f"Target distribution: mean={targets.mean():.3f}, std={targets.std():.3f}")

    winner_prof, rho_prof, indices_prof = run_command_1(features, targets, labels)

    # Command 2 on all data
    scorer_result = run_command_2(features_all, targets_all, labels_all)

    # Command 2 on profiled only
    print("\n  (Also on profiled-only:)")
    scorer_prof = run_command_2(features, targets, labels)

    # Final summary
    print("\n" + "=" * 90)
    print("FINAL RESULTS")
    print("=" * 90)
    print(f"  All data (n={len(features_all)}):")
    print(f"    Best: {winner_all} (LOOCV rho = {rho_all:.4f})")
    print(f"  Profiled only (n={len(features)}):")
    print(f"    Best: {winner_prof} (LOOCV rho = {rho_prof:.4f})")
    print(f"  Capability scorer: {scorer_result['recommendation']} (R² = {scorer_result['r_squared']:.4f})")
    best_rho = max(rho_all, rho_prof)
    if best_rho >= 0.75:
        print(f"  → CEILING BROKEN! Best rho = {best_rho:.4f} > 0.75")
    else:
        print(f"  → 0.75 ceiling NOT broken. Best LOOCV rho = {best_rho:.4f}")
        print(f"    The reduced 52-feature RF model is our shipping model.")


if __name__ == "__main__":
    main()
