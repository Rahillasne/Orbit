#!/usr/bin/env python3
"""Validate ORBIT R3M profiler on real LeRobot datasets using Modal (serverless GPU).

Usage:
    modal run scripts/modal_validate.py

Requires:
    pip install modal
    modal token set  # one-time auth
"""

from __future__ import annotations

import json
import time

import modal

# ---------------------------------------------------------------------------
# Modal image: install ORBIT + GPU dependencies
# ---------------------------------------------------------------------------

orbit_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "faiss-cpu>=1.7.4",
        "scikit-learn>=1.2.0",
        "scipy>=1.10.0",
        "hdbscan>=0.8.33",
        "Pillow>=9.5.0",
        "transformers>=4.30.0",
        "open-clip-torch>=2.20.0",
        "umap-learn>=0.5.3",
        "sentencepiece>=0.1.99",
        "numpy>=1.24.0",
        "h5py>=3.8.0",
        "rich>=13.0.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "filelock>=3.12.0",
        "tqdm>=4.65.0",
        "click>=8.1.0",
        "pyyaml>=6.0",
        "datasets>=2.14.0",
        "huggingface_hub>=0.20.0",
        "pandas>=2.0.0",
        "opencv-python-headless>=4.8.0",
        "lerobot>=0.4.0",
        "joblib>=1.3.0",
    )
    .add_local_dir("orbit", remote_path="/root/orbit_pkg/orbit", copy=True)
    .add_local_file("pyproject.toml", remote_path="/root/orbit_pkg/pyproject.toml", copy=True)
    .add_local_file("README.md", remote_path="/root/orbit_pkg/README.md", copy=True)
    .run_commands("cd /root/orbit_pkg && pip install -e '.[profile]'")
)

app = modal.App("orbit-r3m-validation", image=orbit_image)

# ---------------------------------------------------------------------------
# Dataset configs
# ---------------------------------------------------------------------------

DATASETS = {
    "pusht": {
        "repo": "lerobot/pusht",
        "tasks": ["push block to target", "precise positioning"],
    },
    "aloha_transfer": {
        "repo": "lerobot/aloha_sim_transfer_cube_human",
        "tasks": ["pick up cube", "bimanual handover", "place cube"],
    },
    "aloha_insertion": {
        "repo": "lerobot/aloha_sim_insertion_human",
        "tasks": ["align peg", "insert peg"],
    },
    "xarm_lift": {
        "repo": "lerobot/xarm_lift_medium_replay",
        "tasks": ["grasp object", "lift object"],
    },
}

GROUND_TRUTH = {
    "pusht": 0.91,
    "aloha_transfer": 0.82,
    "aloha_insertion": 0.86,
    "xarm_lift": 0.65,
}


# ---------------------------------------------------------------------------
# GPU function: profile a single dataset
# ---------------------------------------------------------------------------


@app.function(gpu="T4", timeout=1800)
def profile_dataset(name: str, config: dict) -> dict:
    """Profile a single LeRobot dataset on a T4 GPU using the ORBIT pipeline."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{name}] Device: {device}")
    if device == "cuda":
        print(f"[{name}] GPU: {torch.cuda.get_device_name(0)}")

    start = time.time()

    try:
        from orbit.profile.profiler import DatasetProfiler
        from orbit.profile.report_card import ReportCardGenerator

        profiler = DatasetProfiler(
            embedding_model="r3m",
            device=device,
        )

        print(f"[{name}] Profiling {config['repo']} via ORBIT pipeline...")
        profile = profiler.profile_from_hub(
            repo_id=config["repo"],
            task_descriptions=config["tasks"],
            max_episodes=50,
            cache_dir="/tmp/orbit_cache",
        )

        report_gen = ReportCardGenerator()
        report = report_gen.generate(profile)

        elapsed = time.time() - start
        print(f"\n[{name}] Profiled in {elapsed:.1f}s")
        print(report_gen.render_cli(report))

        return {
            "name": name,
            "overall_grade": report.overall_grade,
            "overall_score": report.overall_score,
            "coverage_grade": report.coverage_grade,
            "coverage_score": report.coverage_score,
            "quality_grade": report.quality_grade,
            "quality_score": report.quality_score,
            "diversity_grade": report.diversity_grade,
            "diversity_score": report.diversity_score,
            "volume_grade": report.volume_grade,
            "volume_score": report.volume_score,
            "time_seconds": round(elapsed, 1),
            "device": device,
            "num_images": profile.num_frames,
            "num_episodes": profile.num_episodes,
            "report_card": report.to_dict(),
        }

    except Exception as exc:
        elapsed = time.time() - start
        print(f"[{name}] FAILED after {elapsed:.1f}s: {exc}")
        import traceback

        traceback.print_exc()
        return {
            "name": name,
            "error": str(exc),
            "time_seconds": round(elapsed, 1),
        }


# ---------------------------------------------------------------------------
# Orchestrator: run all datasets + correlation analysis
# ---------------------------------------------------------------------------


@app.function(timeout=3600)
def run_full_validation() -> dict:
    """Launch all dataset profiles in parallel, then compute correlation."""
    from scipy import stats

    # Launch all profiles in parallel on T4 GPUs
    futures = []
    for name, config in DATASETS.items():
        futures.append(profile_dataset.spawn(name, config))

    # Collect results
    results = {}
    for future in futures:
        result = future.get()
        results[result["name"]] = result

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Dataset':<20} {'Grade':>6} {'Score':>6} {'Coverage':>9} {'Time':>8}")
    print("-" * 70)
    for name in DATASETS:
        if name in results:
            r = results[name]
            if "error" in r:
                print(f"{name:<20} ERROR: {r['error']}")
            else:
                print(
                    f"{name:<20} {r['overall_grade']:>6} "
                    f"{r['overall_score']:>6.2f} "
                    f"{r['coverage_score']:>9.2f} "
                    f"{r['time_seconds']:>7.1f}s"
                )

    # Correlation with ground truth
    coverage_scores = []
    gt_scores = []
    for name in GROUND_TRUTH:
        if name in results and "error" not in results[name]:
            coverage_scores.append(results[name]["coverage_score"])
            gt_scores.append(GROUND_TRUTH[name])

    correlation = {}
    if len(coverage_scores) >= 3:
        rho, p_val = stats.spearmanr(coverage_scores, gt_scores)
        pearson_r, pearson_p = stats.pearsonr(coverage_scores, gt_scores)
        correlation = {
            "spearman_rho": round(float(rho), 3),
            "spearman_p": round(float(p_val), 4),
            "pearson_r": round(float(pearson_r), 3),
            "pearson_p": round(float(pearson_p), 4),
        }
        print(f"\nSpearman rho: {rho:.3f} (p={p_val:.4f})")
        print(f"Pearson r:    {pearson_r:.3f} (p={pearson_p:.4f})")

    return {
        "datasets": results,
        "ground_truth": GROUND_TRUTH,
        "correlation": correlation,
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main():
    """Run the full validation and save results locally."""
    print("Launching ORBIT R3M validation on Modal (T4 GPUs)...")
    print(f"Profiling {len(DATASETS)} datasets in parallel\n")

    output = run_full_validation.remote()

    # Save results
    out_path = "results/modal_r3m_validation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {out_path}")

    # Print final summary
    corr = output.get("correlation", {})
    if corr:
        print(f"\nValidation: Spearman rho = {corr['spearman_rho']}")
        if corr["spearman_rho"] > 0.6:
            print("ORBIT R3M scores correlate well with ground truth success rates!")
        else:
            print("Correlation is moderate — may need more datasets for validation.")
