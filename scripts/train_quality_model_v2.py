#!/usr/bin/env python3
"""Train the enterprise-grade quality prediction model.

Usage:
    python3 scripts/train_quality_model_v2.py
    python3 scripts/train_quality_model_v2.py --real-only
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Train V2 quality model with PCA, calibration, and bootstrap CIs"
    )
    parser.add_argument(
        "--real-only",
        action="store_true",
        help="Train only on real profiled datasets",
    )
    parser.add_argument("--cache-dir", default="benchmarks/cached_features")
    parser.add_argument(
        "--gt-file",
        default="orbit/benchmarks/ground_truth_comprehensive.json",
    )
    parser.add_argument(
        "--output",
        default="orbit/profile/pretrained/quality_model_v2.pkl",
    )
    args = parser.parse_args()

    # Load ground truth
    with open(args.gt_file) as f:
        gt = json.load(f)
    gt_lookup = {}
    for e in gt["benchmarks"]:
        gt_lookup[e["id"]] = e

    # Load cached features
    features_list = []
    success_rates = []
    labels = []
    types = []

    for npy_file in sorted(Path(args.cache_dir).glob("*.npy")):
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

        # If --real-only, skip estimated features
        if args.real_only and feat_type != "profiled":
            continue

        feat = np.load(npy_file)
        features_list.append(feat)

        sr = entry.get(
            "reported_success_rate", meta.get("success_rate", 0.0)
        )
        # Normalize D4RL scores (normalized_score 0-110+ -> 0-1)
        if entry.get("metric_type") == "normalized_score":
            sr = min(sr / 100.0, 1.0)

        success_rates.append(sr)
        labels.append(entry_id)
        types.append(feat_type)

    if len(features_list) < 10:
        print(f"ERROR: Only {len(features_list)} datasets. Need at least 10.")
        print("Run 'python3 scripts/profile_training_data.py' first.")
        return

    features = np.array(features_list)
    targets = np.array(success_rates)

    n_profiled = sum(1 for t in types if t == "profiled")
    n_estimated = len(types) - n_profiled
    print(
        f"Loaded {len(features)} datasets: "
        f"{n_profiled} profiled, {n_estimated} estimated"
    )

    # Train the model
    from orbit.profile.predictor_v2 import DatasetQualityModelV2

    model = DatasetQualityModelV2()
    model.fit(features, targets, feature_types=types)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    model.save(args.output)

    # Determine which samples were actually used in training
    # (the model filters to profiled-only if >= 15 profiled exist)
    real_mask = np.array([t == "profiled" for t in types])
    if real_mask.sum() >= 15 and not args.real_only:
        train_mask = real_mask
    else:
        train_mask = np.ones(len(types), dtype=bool)

    train_targets = targets[train_mask]
    train_labels = [l for l, m in zip(labels, train_mask) if m]
    train_types = [t for t, m in zip(types, train_mask) if m]

    # Generate plots
    try:
        _make_plots(model, train_targets, train_labels, train_types)
    except Exception as e:
        print(f"Plot generation failed: {e}")

    # Print per-dataset table with labels
    v = model.validation_results
    cv_pred = v["cv_predictions"]

    print(
        f"\n{'Dataset':<42} {'Actual':>7} {'Pred':>7} "
        f"{'Error':>7} {'Type':>15}"
    )
    print("─" * 85)

    sorted_idx = np.argsort(np.abs(cv_pred - train_targets))
    for idx in sorted_idx:
        err = cv_pred[idx] - train_targets[idx]
        marker = "✓" if abs(err) < 0.15 else "✗"
        print(
            f"  {train_labels[idx]:<40} {train_targets[idx]:>7.2f} {cv_pred[idx]:>7.2f} "
            f"{err:>+7.2f} {train_types[idx]:>15} {marker}"
        )

    # Summary
    rho = v["spearman_rho"]
    print(f"\n{'=' * 70}")
    print(
        f"FINAL: rho={rho:.3f}, r={v['pearson_r']:.3f}, "
        f"MAE={v['mae']:.3f}, Rank Accuracy={v['rank_accuracy'] * 100:.1f}%"
    )

    if rho >= 0.80:
        print("STATUS: Ready for production")
    elif rho >= 0.70:
        print("STATUS: Usable — would benefit from more training data")
    else:
        print("STATUS: Needs improvement — profile more real datasets")


def _make_plots(model, targets, labels, types):
    """Generate calibration + residual + feature importance plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v = model.validation_results
    cv_pred = v["cv_predictions"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Calibration (predicted vs actual)
    ax = axes[0]
    for dtype, color, marker in [
        ("profiled", "#2563eb", "o"),
        ("estimated", "#9ca3af", "s"),
        ("estimated_fallback", "#d1d5db", "^"),
    ]:
        mask = [t == dtype for t in types]
        if any(mask):
            ax.scatter(
                [targets[i] for i in range(len(mask)) if mask[i]],
                [cv_pred[i] for i in range(len(mask)) if mask[i]],
                c=color,
                marker=marker,
                s=60,
                alpha=0.7,
                label=dtype,
                edgecolors="black",
                linewidth=0.5,
            )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect")
    ax.set_xlabel("Actual Success Rate")
    ax.set_ylabel("Predicted (Cross-Validated)")
    ax.set_title(f"Calibration — ρ={v['spearman_rho']:.3f}")
    ax.legend(fontsize=8)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # Plot 2: Residual distribution
    ax = axes[1]
    residuals = cv_pred - targets
    ax.hist(residuals, bins=20, color="#2563eb", alpha=0.7, edgecolor="black")
    ax.axvline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Prediction Error (Pred - Actual)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residuals — MAE={v['mae']:.3f}, RMSE={v['rmse']:.3f}")

    # Plot 3: Feature importance (PCA-space)
    ax = axes[2]
    if hasattr(model.model, "named_estimators_"):
        gbr = model.model.named_estimators_.get("gbr")
        if gbr is not None and hasattr(gbr, "feature_importances_"):
            importances = gbr.feature_importances_
            top_k = min(10, len(importances))
            top_idx = np.argsort(importances)[-top_k:]
            ax.barh(
                [f"PC{i + 1}" for i in top_idx],
                importances[top_idx],
                color="#2563eb",
                alpha=0.7,
            )
            ax.set_xlabel("Feature Importance (GBR)")
            ax.set_title("Top Principal Components")

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/quality_model_v2_report.png", dpi=150)
    print(f"\nPlots saved to results/quality_model_v2_report.png")


if __name__ == "__main__":
    main()
