#!/usr/bin/env python3
"""Train the quality prediction model from cached features. Runs locally, no GPU needed.

Usage:
    python3 scripts/train_quality_model_local.py
    python3 scripts/train_quality_model_local.py --cache-dir benchmarks/cached_features
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Train quality model from cached features")
    parser.add_argument("--cache-dir", default="benchmarks/cached_features")
    parser.add_argument("--gt-file", default="orbit/benchmarks/ground_truth_comprehensive.json")
    parser.add_argument("--output", default="orbit/profile/pretrained/quality_model.pkl")
    args = parser.parse_args()

    # Load ground truth
    with open(args.gt_file) as f:
        gt = json.load(f)

    # Build ID -> entry lookup
    gt_lookup = {}
    for e in gt["benchmarks"]:
        gt_lookup[e["id"]] = e

    # Load all cached features
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

        feat = np.load(npy_file)
        features_list.append(feat)

        sr = entry.get("reported_success_rate", meta.get("success_rate", 0.0))
        # Normalize D4RL scores (normalized_score 0-110+ -> 0-1)
        if entry.get("metric_type") == "normalized_score":
            sr = min(sr / 100.0, 1.0)

        success_rates.append(sr)
        labels.append(entry_id)
        types.append(meta.get("type", "unknown"))

    if len(features_list) < 5:
        print(f"ERROR: Only {len(features_list)} datasets cached. Need at least 5.")
        print(f"Run 'python3 scripts/profile_training_data.py' first.")
        return

    features = np.array(features_list)
    targets = np.array(success_rates)

    n_profiled = sum(1 for t in types if t == "profiled")
    n_estimated = sum(1 for t in types if t != "profiled")
    print(f"Training on {len(features)} datasets ({n_profiled} profiled, {n_estimated} estimated)")

    # Handle NaN/Inf in features
    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)

    # Build sample weights: profiled datasets get 3x weight
    sample_weights = np.array(
        [3.0 if t == "profiled" else 1.0 for t in types],
        dtype=np.float32,
    )

    # Train the model
    from orbit.profile.predictor import DatasetQualityModel

    model = DatasetQualityModel()
    model.fit(features, targets, sample_weights=sample_weights)

    # Print LOOCV results
    loocv = model.loocv_results
    print(f"\n{'=' * 60}")
    print("TRAINING RESULTS (Leave-One-Out Cross-Validation)")
    print(f"{'=' * 60}")
    print(f"Spearman rho: {loocv['spearman_rho']:.3f} (p={loocv['spearman_p']:.4f})")
    print(f"Pearson r:    {loocv['pearson_r']:.3f} (p={loocv['pearson_p']:.4f})")
    print(f"MAE:          {loocv['mae']:.3f}")
    print(f"N samples:    {loocv['n_samples']}")

    # Print per-dataset predictions
    print(f"\n{'Dataset':<40} {'Actual':>7} {'Predicted':>10} {'Error':>7} {'Type':>10}")
    print("-" * 80)
    for label, actual, predicted, dtype in zip(
        labels, loocv["actuals"], loocv["predictions"], types
    ):
        error = predicted - actual
        marker = " ok" if abs(error) < 0.15 else " !!"
        print(f"{label:<40} {actual:>7.2f} {predicted:>10.2f} {error:>+7.2f} {dtype:>10}{marker}")

    # Save model
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    model.save(args.output)
    print(f"\nModel saved to {args.output}")

    # Generate calibration plot
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Color by type
        for dtype, color, marker in [
            ("profiled", "blue", "o"),
            ("estimated", "gray", "s"),
            ("estimated_fallback", "lightgray", "^"),
            ("unknown", "orange", "x"),
        ]:
            mask = [t == dtype for t in types]
            if any(mask):
                ax.scatter(
                    [a for a, m in zip(loocv["actuals"], mask) if m],
                    [p for p, m in zip(loocv["predictions"], mask) if m],
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
        ax.set_ylabel("Predicted Success Rate (LOOCV)")
        ax.set_title(
            f"ORBIT Quality Model - rho={loocv['spearman_rho']:.2f}, "
            f"r={loocv['pearson_r']:.2f}, MAE={loocv['mae']:.2f}"
        )
        ax.legend()
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()

        os.makedirs("results", exist_ok=True)
        plt.savefig("results/quality_model_calibration.png", dpi=150)
        print(f"Calibration plot saved to results/quality_model_calibration.png")
    except ImportError:
        print("matplotlib not installed - skipping calibration plot")


if __name__ == "__main__":
    main()
