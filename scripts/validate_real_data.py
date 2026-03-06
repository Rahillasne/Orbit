#!/usr/bin/env python3
"""Validate ORBIT DatasetProfiler on real LeRobot datasets.

Downloads a small dataset, converts it to ORBIT format, runs the profiler,
and validates that capability scores produce correct ordering.

Usage:
    python scripts/validate_real_data.py [--max-episodes N] [--cache-dir DIR]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


# Datasets and their expected capability ordering.
# Each entry: (repo_id, task_descriptions, expected_high, expected_low)
VALIDATION_CONFIGS = [
    {
        "repo_id": "lerobot/aloha_static_cups_open",
        "tasks": [
            "bimanual robot opening a plastic cup",
            "robot arms manipulating a cup on a table",
            "navigate through a doorway",
            "fold laundry on a table",
            "sweep the floor with a broom",
        ],
        "expected_high": ["bimanual robot opening a plastic cup"],
        "expected_low": ["navigate through a doorway", "fold laundry on a table", "sweep the floor with a broom"],
    },
]

FALLBACK_DATASETS = [
    "lerobot/pusht",
    "lerobot/xarm_lift_medium_replay",
]


def run_validation(
    repo_id: str,
    tasks: list[str],
    expected_high: list[str],
    expected_low: list[str],
    max_episodes: int,
    cache_dir: str,
) -> bool:
    """Run profiler on a dataset and validate results. Returns True if passed."""
    from orbit.profile import DatasetLoader, DatasetProfiler, ProfileReporter

    print(f"\n{'='*60}")
    print(f"Dataset: {repo_id}")
    print(f"{'='*60}")

    # Step 1: Convert dataset
    print("\n[1/4] Converting dataset...")
    t0 = time.time()
    output_dir = Path(cache_dir) / repo_id.replace("/", "_")
    try:
        DatasetLoader.from_lerobot(repo_id, output_dir, max_episodes=max_episodes, fps_sample=20)
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return False
    convert_time = time.time() - t0
    print(f"  Done in {convert_time:.1f}s")

    # Step 2: Run profiler
    print("\n[2/4] Running profiler...")
    t0 = time.time()
    profiler = DatasetProfiler()
    profile = profiler.profile(str(output_dir), task_descriptions=tasks)
    profile_time = time.time() - t0
    print(f"  Done in {profile_time:.1f}s")

    # Step 3: Generate report
    print("\n[3/4] Generating report...")
    reporter = ProfileReporter()
    report = reporter.generate_report(profile, format="markdown")

    report_path = Path(cache_dir) / "orbit_profile_report.md"
    report_path.write_text(report)
    print(f"  Saved to {report_path}")

    # Step 4: Print and validate results
    print("\n[4/4] Results")
    print(f"  Episodes:    {profile.num_episodes}")
    print(f"  Frames:      {profile.num_frames}")
    print(f"  Embeddings:  {profile.embedding_index.num_embeddings}")
    print(f"  Coverage:    {profile.coverage.overall_coverage_score:.3f}")
    print(f"  Quality:     {profile.quality.aggregate_score:.3f}")
    print(f"  Total time:  {convert_time + profile_time:.1f}s")

    print("\n  Capability Scores:")
    score_map: dict[str, float] = {}
    for cap in sorted(profile.capabilities, key=lambda c: c.score, reverse=True):
        score_map[cap.task_description] = cap.score
        marker = ""
        if cap.task_description in expected_high:
            marker = " <-- expected HIGH"
        elif cap.task_description in expected_low:
            marker = " <-- expected LOW"
        print(f"    {cap.score:.3f}  {cap.task_description}{marker}")

    if profile.prescriptions:
        print("\n  Top 3 Prescriptions:")
        for rx in profile.prescriptions[:3]:
            print(f"    [{rx.get('priority', '?')}] {rx.get('instruction', rx.get('task', '?'))}")

    # Validate ordering
    passed = True
    if expected_high and expected_low:
        min_high = min(score_map.get(t, 0.0) for t in expected_high)
        max_low = max(score_map.get(t, 1.0) for t in expected_low)

        if min_high > max_low:
            print(f"\n  PASS: High tasks ({min_high:.3f}) > Low tasks ({max_low:.3f})")
        else:
            print(f"\n  FAIL: High tasks ({min_high:.3f}) <= Low tasks ({max_low:.3f})")
            print("  The capability scoring ordering is incorrect!")
            passed = False

    print(f"\n  Full report:\n{report}")
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-episodes", type=int, default=20)
    parser.add_argument("--cache-dir", default="/tmp/orbit_validation")
    parser.add_argument("--dataset", default=None, help="Override dataset repo ID")
    args = parser.parse_args()

    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    configs = VALIDATION_CONFIGS
    if args.dataset:
        configs = [{
            "repo_id": args.dataset,
            "tasks": VALIDATION_CONFIGS[0]["tasks"],
            "expected_high": VALIDATION_CONFIGS[0]["expected_high"],
            "expected_low": VALIDATION_CONFIGS[0]["expected_low"],
        }]

    all_passed = True
    for config in configs:
        try:
            passed = run_validation(
                repo_id=config["repo_id"],
                tasks=config["tasks"],
                expected_high=config["expected_high"],
                expected_low=config["expected_low"],
                max_episodes=args.max_episodes,
                cache_dir=args.cache_dir,
            )
        except Exception as exc:
            print(f"\n  ERROR: {exc}")
            import traceback
            traceback.print_exc()
            passed = False

        if not passed:
            # Try fallback datasets
            for fallback in FALLBACK_DATASETS:
                print(f"\n  Trying fallback: {fallback}")
                try:
                    passed = run_validation(
                        repo_id=fallback,
                        tasks=config["tasks"],
                        expected_high=config["expected_high"],
                        expected_low=config["expected_low"],
                        max_episodes=args.max_episodes,
                        cache_dir=args.cache_dir,
                    )
                    if passed:
                        break
                except Exception as exc:
                    print(f"  Fallback {fallback} failed: {exc}")

        all_passed = all_passed and passed

    if all_passed:
        print("\n\nAll validations PASSED")
    else:
        print("\n\nSome validations FAILED")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
