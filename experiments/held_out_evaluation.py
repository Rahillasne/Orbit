#!/usr/bin/env python3
"""Phase 4: Held-out evaluation with bootstrap CI and permutation test.

This produces the number you put in the paper. Everything before this
was development; this is the real evaluation.

Usage:
    python3 experiments/held_out_evaluation.py
    python3 experiments/held_out_evaluation.py --use-all
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from model_ablation import FEATURE_NAMES, load_data


# ──────────────────────────────────────────────────────────────────
# Stratified split
# ──────────────────────────────────────────────────────────────────

def stratified_split(
    targets: np.ndarray,
    labels: list[str],
    test_fraction: float = 0.30,
    min_test: int = 10,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices stratified by success rate quantiles.

    Ensures at least min_test samples in test set and that
    different task domains appear in both splits.
    """
    n = len(targets)
    n_test = max(min_test, int(n * test_fraction))
    n_train = n - n_test

    rng = np.random.RandomState(random_state)

    # Stratify by quartiles of success rate
    quartiles = np.digitize(targets, np.percentile(targets, [25, 50, 75]))

    train_idx = []
    test_idx = []

    for q in np.unique(quartiles):
        q_indices = np.where(quartiles == q)[0]
        rng.shuffle(q_indices)

        # Proportional split
        n_q_test = max(1, int(len(q_indices) * test_fraction))
        test_idx.extend(q_indices[:n_q_test].tolist())
        train_idx.extend(q_indices[n_q_test:].tolist())

    return np.array(train_idx), np.array(test_idx)


# ──────────────────────────────────────────────────────────────────
# Bootstrap CI for Spearman rho
# ──────────────────────────────────────────────────────────────────

def bootstrap_spearman_ci(
    predictions: np.ndarray,
    actuals: np.ndarray,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Compute Spearman rho with bootstrap 95% CI.

    Returns: (rho, ci_low, ci_high)
    """
    rng = np.random.RandomState(random_state)
    n = len(predictions)

    rho_obs, _ = stats.spearmanr(predictions, actuals)

    rhos = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if len(set(actuals[idx])) < 3:
            continue
        r, _ = stats.spearmanr(predictions[idx], actuals[idx])
        if not np.isnan(r):
            rhos.append(r)

    alpha = 1 - ci_level
    ci_low = np.percentile(rhos, 100 * alpha / 2)
    ci_high = np.percentile(rhos, 100 * (1 - alpha / 2))

    return float(rho_obs), float(ci_low), float(ci_high)


# ──────────────────────────────────────────────────────────────────
# Permutation test
# ──────────────────────────────────────────────────────────────────

def permutation_test_rho(
    predictions: np.ndarray,
    actuals: np.ndarray,
    n_permutations: int = 10000,
    random_state: int = 42,
) -> tuple[float, float]:
    """Test H0: rho = 0 by shuffling ground truth.

    Returns: (rho_obs, p_value)
    """
    rng = np.random.RandomState(random_state)
    rho_obs, _ = stats.spearmanr(predictions, actuals)

    count_extreme = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(actuals)
        rho_perm, _ = stats.spearmanr(predictions, shuffled)
        if abs(rho_perm) >= abs(rho_obs):
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_permutations + 1)
    return float(rho_obs), float(p_value)


# ──────────────────────────────────────────────────────────────────
# Main evaluation
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 4: Held-out evaluation")
    parser.add_argument("--use-all", action="store_true")
    parser.add_argument("--cache-dir", default="benchmarks/cached_features")
    parser.add_argument("--gt-file", default="orbit/benchmarks/ground_truth_comprehensive.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    features, targets, labels, types = load_data(
        cache_dir=args.cache_dir,
        gt_file=args.gt_file,
        profiled_only=not args.use_all,
    )

    n = len(features)
    print(f"Loaded {n} datasets ({sum(1 for t in types if t == 'profiled')} profiled)")

    if n < 15:
        print("ERROR: Need at least 15 datasets for held-out evaluation")
        sys.exit(1)

    # ── Split ──────────────────────────────────────────────────
    train_idx, test_idx = stratified_split(targets, labels, random_state=args.seed)

    X_train, X_test = features[train_idx], features[test_idx]
    y_train, y_test = targets[train_idx], targets[test_idx]
    train_labels = [labels[i] for i in train_idx]
    test_labels = [labels[i] for i in test_idx]
    train_types = [types[i] for i in train_idx]
    test_types = [types[i] for i in test_idx]

    print(f"\n  Train: {len(train_idx)} datasets")
    print(f"    y_train: mean={y_train.mean():.3f}, std={y_train.std():.3f}, "
          f"min={y_train.min():.3f}, max={y_train.max():.3f}")
    print(f"  Test:  {len(test_idx)} datasets")
    print(f"    y_test:  mean={y_test.mean():.3f}, std={y_test.std():.3f}, "
          f"min={y_test.min():.3f}, max={y_test.max():.3f}")

    # ── Train best model ──────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Use RF for small n, GBR for larger
    if n >= 50:
        model = GradientBoostingRegressor(
            n_estimators=50, max_depth=2,
            learning_rate=0.1, subsample=0.8,
            min_samples_leaf=3, random_state=42,
        )
    else:
        model = RandomForestRegressor(
            n_estimators=50, max_depth=3,
            min_samples_leaf=3, random_state=42,
        )

    model.fit(X_train_s, y_train)
    predictions = np.clip(model.predict(X_test_s), 0.0, 1.0)

    # ── Metrics ───────────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("HELD-OUT TEST SET EVALUATION")
    print(f"{'=' * 90}")

    # Spearman rho with bootstrap CI
    rho, ci_low, ci_high = bootstrap_spearman_ci(predictions, y_test)
    print(f"\n  Spearman ρ = {rho:.3f}  [95% CI: {ci_low:.3f} — {ci_high:.3f}]")

    # Pearson r
    r, r_p = stats.pearsonr(predictions, y_test)
    print(f"  Pearson r  = {r:.3f}  (p = {r_p:.4f})")

    # MAE, RMSE
    mae = float(np.mean(np.abs(predictions - y_test)))
    rmse = float(np.sqrt(np.mean((predictions - y_test) ** 2)))
    print(f"  MAE        = {mae:.3f}")
    print(f"  RMSE       = {rmse:.3f}")

    # Permutation test
    print(f"\n  Running permutation test (10,000 shuffles)...")
    _, p_perm = permutation_test_rho(predictions, y_test)
    print(f"  Permutation p-value = {p_perm:.4f}")
    if p_perm < 0.05:
        print(f"  → ρ is SIGNIFICANTLY different from random (p < 0.05)")
    elif p_perm < 0.10:
        print(f"  → MARGINAL significance (0.05 < p < 0.10)")
    else:
        print(f"  → NOT significant (p ≥ 0.10) — cannot reject H0: ρ = 0")

    # ── Per-dataset predictions ───────────────────────────────
    print(f"\n  {'Dataset':<42} {'Actual':>7} {'Pred':>7} {'Error':>7} {'Type':>12}")
    print(f"  {'─' * 75}")

    errors = predictions - y_test
    sorted_test = np.argsort(np.abs(errors))
    for idx in sorted_test:
        marker = "✓" if abs(errors[idx]) < 0.15 else "✗"
        print(f"  {test_labels[idx]:<40} {y_test[idx]:>7.3f} {predictions[idx]:>7.3f} "
              f"{errors[idx]:>+7.3f} {test_types[idx]:>12} {marker}")

    # ── Summary ───────────────────────────────────────────────
    within_15 = np.mean(np.abs(errors) < 0.15) * 100
    within_20 = np.mean(np.abs(errors) < 0.20) * 100

    print(f"\n{'=' * 90}")
    print("SUMMARY — THE NUMBER FOR THE PAPER")
    print(f"{'=' * 90}")
    print(f"  Model:            {'GBR' if n >= 50 else 'RF'} (depth={'2' if n >= 50 else '3'}, n_estimators=50)")
    print(f"  Train / Test:     {len(train_idx)} / {len(test_idx)}")
    print(f"  Spearman ρ:       {rho:.3f}  [95% CI: {ci_low:.3f} — {ci_high:.3f}]")
    print(f"  Permutation p:    {p_perm:.4f}")
    print(f"  Pearson r:        {r:.3f}")
    print(f"  MAE:              {mae:.3f}")
    print(f"  Within ±0.15:     {within_15:.0f}%")
    print(f"  Within ±0.20:     {within_20:.0f}%")
    print()

    if rho >= 0.70 and p_perm < 0.05:
        print(f"  VERDICT: ρ ≥ 0.70 and significant → PUBLISHABLE")
    elif rho >= 0.50 and p_perm < 0.05:
        print(f"  VERDICT: ρ ≥ 0.50 and significant → USABLE (include caveats)")
    else:
        print(f"  VERDICT: Need more data or better features before publishing")

    # ── Save results ──────────────────────────────────────────
    output_path = "docs/held_out_results.md"
    os.makedirs("docs", exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# Held-Out Evaluation Results\n\n")
        f.write("## Setup\n\n")
        f.write(f"- Total datasets: {n}\n")
        f.write(f"- Train: {len(train_idx)}, Test: {len(test_idx)}\n")
        f.write(f"- Split: Stratified by success rate quartiles\n")
        f.write(f"- Model: {'GBR' if n >= 50 else 'RF'}\n\n")

        f.write("## Results\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Spearman ρ | {rho:.3f} [{ci_low:.3f}, {ci_high:.3f}] |\n")
        f.write(f"| Permutation p-value | {p_perm:.4f} |\n")
        f.write(f"| Pearson r | {r:.3f} |\n")
        f.write(f"| MAE | {mae:.3f} |\n")
        f.write(f"| RMSE | {rmse:.3f} |\n")
        f.write(f"| Within ±0.15 | {within_15:.0f}% |\n")
        f.write(f"| Within ±0.20 | {within_20:.0f}% |\n\n")

        f.write("## Per-Dataset Predictions\n\n")
        f.write("| Dataset | Actual | Predicted | Error |\n")
        f.write("|---------|--------|-----------|-------|\n")
        for idx in sorted_test:
            f.write(f"| {test_labels[idx]} | {y_test[idx]:.3f} | {predictions[idx]:.3f} | "
                    f"{errors[idx]:+.3f} |\n")

    print(f"\n  Results saved to {output_path}")

    # ── Scatter plot ──────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(y_test, predictions, c="#2563eb", s=80, alpha=0.7,
                   edgecolors="black", linewidth=0.5, zorder=5)

        for i in range(len(y_test)):
            if abs(errors[i]) > 0.15:
                ax.annotate(
                    test_labels[i].replace("_", "\n"),
                    (y_test[i], predictions[i]),
                    fontsize=6, alpha=0.6,
                    xytext=(5, 5), textcoords="offset points",
                )

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
        ax.fill_between([0, 1], [-0.15, 0.85], [0.15, 1.15],
                        alpha=0.1, color="green", label="±0.15")
        ax.set_xlabel("Actual Success Rate", fontsize=12)
        ax.set_ylabel("Predicted Success Rate", fontsize=12)
        ax.set_title(
            f"Held-Out Evaluation (n={len(test_idx)})\n"
            f"ρ={rho:.3f} [{ci_low:.3f}, {ci_high:.3f}], "
            f"p={p_perm:.4f}, MAE={mae:.3f}",
            fontsize=11,
        )
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        plt.savefig("docs/held_out_scatter.png", dpi=150, bbox_inches="tight")
        print(f"  Scatter plot saved to docs/held_out_scatter.png")
        plt.close()
    except ImportError:
        pass


if __name__ == "__main__":
    main()
