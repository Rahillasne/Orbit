#!/usr/bin/env python3
"""Train the ORBIT quality predictor from locally cached features using Modal.

This script uploads pre-cached feature vectors (from profile_training_data.py)
to Modal and trains the sklearn ensemble there. No GPU profiling is done on
Modal — all profiling happens locally first.

Usage:
    # Step 1: Profile datasets locally (slow, but reliable)
    python3 scripts/profile_training_data.py --device mps --max-episodes 10

    # Step 2: Train the model (fast, can run locally or on Modal)
    python3 scripts/train_quality_model_local.py   # preferred: fully local
    modal run scripts/modal_train_predictor.py      # alternative: on Modal

Requires:
    pip install modal
    modal token set  # one-time auth
"""

from __future__ import annotations

import json
import os

import modal

# ---------------------------------------------------------------------------
# Modal image: lightweight — only sklearn + scipy, no torch/GPU needed
# ---------------------------------------------------------------------------

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "scikit-learn>=1.2.0",
        "scipy>=1.10.0",
        "numpy>=1.24.0",
        "joblib>=1.3.0",
        "matplotlib>=3.7.0",
    )
    .add_local_dir("orbit", remote_path="/root/orbit_pkg/orbit", copy=True)
    .add_local_file("pyproject.toml", remote_path="/root/orbit_pkg/pyproject.toml", copy=True)
    .add_local_file("README.md", remote_path="/root/orbit_pkg/README.md", copy=True)
    .run_commands("cd /root/orbit_pkg && pip install -e '.[profile]'")
)

app = modal.App("orbit-train-predictor", image=train_image)


# ---------------------------------------------------------------------------
# CPU function: train model from cached features (no GPU needed)
# ---------------------------------------------------------------------------


@app.function(timeout=300)
def train_from_cached(
    feature_data: list[dict],
    gt_benchmarks: list[dict],
) -> dict:
    """Train the quality model from pre-cached feature vectors.

    Parameters
    ----------
    feature_data:
        List of dicts with keys: id, features (list[float]), type (str).
    gt_benchmarks:
        Ground truth benchmark entries with id, reported_success_rate, metric_type.
    """
    import base64
    import io

    import joblib
    import numpy as np

    from orbit.profile.predictor import DatasetQualityModel

    # Build lookup
    gt_lookup = {e["id"]: e for e in gt_benchmarks}

    features_list = []
    success_rates = []
    labels = []
    types = []

    for item in feature_data:
        entry_id = item["id"]
        if entry_id not in gt_lookup:
            continue

        entry = gt_lookup[entry_id]
        sr = entry.get("reported_success_rate", 0.0)
        if entry.get("metric_type") == "normalized_score":
            sr = min(sr / 100.0, 1.0)

        features_list.append(item["features"])
        success_rates.append(sr)
        labels.append(entry_id)
        types.append(item.get("type", "unknown"))

    features = np.array(features_list, dtype=np.float32)
    targets = np.array(success_rates, dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)

    print(f"Training on {len(features)} datasets")

    model = DatasetQualityModel()
    model.fit(features, targets)

    loocv = model.loocv_results
    print(f"Spearman rho: {loocv['spearman_rho']:.3f}")
    print(f"Pearson r:    {loocv['pearson_r']:.3f}")
    print(f"MAE:          {loocv['mae']:.3f}")

    # Generate calibration plot
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.scatter(
        loocv["actuals"],
        loocv["predictions"],
        s=60,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
    )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")
    ax.set_xlabel("Actual Success Rate")
    ax.set_ylabel("Predicted Success Rate (LOOCV)")
    ax.set_title(
        f"ORBIT Quality Model - LOOCV "
        f"(rho={loocv['spearman_rho']:.2f}, r={loocv['pearson_r']:.2f})"
    )
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()

    plot_buf = io.BytesIO()
    plt.savefig(plot_buf, format="png", dpi=150)
    plot_bytes = base64.b64encode(plot_buf.getvalue()).decode()
    plt.close()

    # Serialize model
    model_buf = io.BytesIO()
    joblib.dump(
        {"models": model.models, "weights": model.weights, "loocv": model.loocv_results},
        model_buf,
        compress=3,
    )
    model_bytes = base64.b64encode(model_buf.getvalue()).decode()

    return {
        "loocv": loocv,
        "n_datasets": len(features),
        "labels": labels,
        "model_bytes": model_bytes,
        "plot_bytes": plot_bytes,
    }


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main():
    """Run from CLI: modal run scripts/modal_train_predictor.py"""
    import base64
    from pathlib import Path

    import numpy as np

    cache_dir = Path("benchmarks/cached_features")
    gt_file = Path("orbit/benchmarks/ground_truth_comprehensive.json")

    if not cache_dir.exists():
        print("ERROR: No cached features found.")
        print("Run 'python3 scripts/profile_training_data.py' first.")
        return

    # Load ground truth
    with open(gt_file) as f:
        gt = json.load(f)

    # Load cached features
    feature_data = []
    for npy_file in sorted(cache_dir.glob("*.npy")):
        entry_id = npy_file.stem
        meta_file = npy_file.with_suffix(".json")
        meta = {}
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)

        feat = np.load(npy_file)
        feature_data.append({
            "id": entry_id,
            "features": feat.tolist(),
            "type": meta.get("type", "unknown"),
        })

    print(f"Uploading {len(feature_data)} cached feature vectors to Modal...")
    print("Training model (CPU only, should take <1 minute)...\n")

    result = train_from_cached.remote(feature_data, gt["benchmarks"])

    # Save model locally
    os.makedirs("orbit/profile/pretrained", exist_ok=True)
    model_data = base64.b64decode(result["model_bytes"])
    with open("orbit/profile/pretrained/quality_model.pkl", "wb") as f:
        f.write(model_data)

    # Save calibration plot
    os.makedirs("results", exist_ok=True)
    plot_data = base64.b64decode(result["plot_bytes"])
    with open("results/quality_model_calibration.png", "wb") as f:
        f.write(plot_data)

    loocv = result["loocv"]
    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Trained on {result['n_datasets']} datasets")
    print(f"Spearman rho: {loocv['spearman_rho']:.3f}")
    print(f"Pearson r:    {loocv['pearson_r']:.3f}")
    print(f"MAE:          {loocv['mae']:.3f}")
    print(f"\nModel saved to orbit/profile/pretrained/quality_model.pkl")
    print(f"Calibration plot saved to results/quality_model_calibration.png")
