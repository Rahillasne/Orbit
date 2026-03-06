#!/usr/bin/env python3
"""Generate pre-computed demo profile data for HuggingFace Spaces.

Produces:
  spaces/data/demo_profile.json
  spaces/data/demo_report.md

No ML dependencies required -- all data is synthetic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SPACES_DATA = Path(__file__).resolve().parent.parent / "spaces" / "data"

# ── Capability scores (hand-crafted) ──────────────────────────────────────

CAPABILITIES = [
    {
        "task": "pick up red cup",
        "score": 0.85,
        "confidence": 0.90,
        "supporting_episodes": 24,
        "action_diversity": 0.78,
        "environment_diversity": 0.65,
        "verdict": "Strong",
        "gap_description": None,
    },
    {
        "task": "open drawer",
        "score": 0.78,
        "confidence": 0.85,
        "supporting_episodes": 20,
        "action_diversity": 0.72,
        "environment_diversity": 0.60,
        "verdict": "Strong",
        "gap_description": None,
    },
    {
        "task": "stack blocks",
        "score": 0.62,
        "confidence": 0.75,
        "supporting_episodes": 15,
        "action_diversity": 0.58,
        "environment_diversity": 0.50,
        "verdict": "Adequate",
        "gap_description": None,
    },
    {
        "task": "wipe surface with sponge",
        "score": 0.55,
        "confidence": 0.70,
        "supporting_episodes": 12,
        "action_diversity": 0.45,
        "environment_diversity": 0.42,
        "verdict": "Adequate",
        "gap_description": None,
    },
    {
        "task": "pour water into glass",
        "score": 0.35,
        "confidence": 0.60,
        "supporting_episodes": 8,
        "action_diversity": 0.30,
        "environment_diversity": 0.28,
        "verdict": "Weak",
        "gap_description": "Limited coverage of pouring motions and liquid handling.",
    },
    {
        "task": "fold cloth napkin",
        "score": 0.22,
        "confidence": 0.50,
        "supporting_episodes": 5,
        "action_diversity": 0.20,
        "environment_diversity": 0.18,
        "verdict": "Weak",
        "gap_description": "Few demonstrations of deformable object manipulation.",
    },
    {
        "task": "navigate to kitchen",
        "score": 0.08,
        "confidence": 0.30,
        "supporting_episodes": 2,
        "action_diversity": 0.05,
        "environment_diversity": 0.10,
        "verdict": "No Coverage",
        "gap_description": "No visual coverage of navigation scenarios. Dataset is tabletop-only.",
    },
    {
        "task": "fly a helicopter",
        "score": 0.02,
        "confidence": 0.15,
        "supporting_episodes": 0,
        "action_diversity": 0.01,
        "environment_diversity": 0.02,
        "verdict": "No Coverage",
        "gap_description": "Completely outside the dataset's domain.",
    },
]


def _generate_prescriptions(capabilities: list[dict]) -> list[dict]:
    """Generate prescriptions for tasks with score < 0.5."""
    prescriptions = []
    for cap in capabilities:
        if cap["score"] >= 0.5:
            continue
        gap_score = 1.0 - cap["score"]
        estimated_demos = max(5, int(gap_score * 30))
        gap_desc = cap["gap_description"] or f"Insufficient coverage for '{cap['task']}'."
        instruction = (
            f"Collect {estimated_demos} demonstrations of '{cap['task']}'. "
            f"Focus on: {gap_desc}"
        )
        prescriptions.append({
            "priority": 0,
            "task": cap["task"],
            "instruction": instruction,
            "estimated_demos": estimated_demos,
            "gap_description": gap_desc,
            "current_capability": round(cap["score"], 3),
            "target_capability": round(min(cap["score"] + 0.3, 1.0), 3),
            "_sort_key": gap_score,
        })
    prescriptions.sort(key=lambda p: p["_sort_key"], reverse=True)
    for i, p in enumerate(prescriptions):
        p["priority"] = i + 1
        del p["_sort_key"]
    return prescriptions


def _generate_umap_points(rng: np.random.Generator) -> tuple[list, list]:
    """Generate 400 UMAP-like 2D points in 3 clusters."""
    centers = [(-3.0, 0.5), (2.5, 2.0), (3.0, -2.5)]
    sizes = [160, 140, 100]
    points = []
    labels = []
    for cluster_id, (center, size) in enumerate(zip(centers, sizes)):
        cx, cy = center
        pts = rng.normal(loc=[cx, cy], scale=0.8, size=(size, 2))
        points.append(pts)
        labels.extend([cluster_id] * size)
    points = np.vstack(points)
    return points.round(4).tolist(), labels


def _generate_quality_scores(rng: np.random.Generator, n: int = 40) -> list[float]:
    """Generate per-episode quality scores."""
    scores = rng.normal(loc=0.71, scale=0.15, size=n)
    scores = np.clip(scores, 0.05, 0.99)
    return scores.round(3).tolist()


def _render_markdown(data: dict) -> str:
    """Render the report as markdown (mirrors ProfileReporter._render_markdown)."""
    lines = []
    lines.append(f"# Dataset Profile: {data['dataset_name']}")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- **Episodes:** {data['num_episodes']}")
    lines.append(f"- **Frames:** {data['num_frames']}")
    lines.append(f"- **Overall Coverage:** {data['overall_coverage_score']:.3f}")
    lines.append(f"- **Quality (MI):** {data['quality']['mutual_information']:.3f}")
    lines.append("")
    if data["strengths"]:
        lines.append(f"**Top strengths:** {', '.join(data['strengths'])}")
    if data["gaps"]:
        lines.append(f"**Top gaps:** {', '.join(data['gaps'])}")
    lines.append("")

    lines.append("## Coverage Analysis")
    lines.append("")
    lines.append(f"- Dense regions: {data['coverage']['num_dense_regions']}")
    lines.append(f"- Sparse regions: {data['coverage']['num_sparse_regions']}")
    lines.append(f"- Overall coverage score: {data['coverage']['overall_score']:.3f}")
    lines.append("")

    lines.append("## Capability Breakdown")
    lines.append("")
    lines.append("| Task | Score | Confidence | Episodes | Verdict |")
    lines.append("|------|-------|------------|----------|---------|")
    for cap in data["capabilities"]:
        lines.append(
            f"| {cap['task']} | {cap['score']:.3f} | "
            f"{cap['confidence']:.3f} | {cap['supporting_episodes']} | "
            f"{cap['verdict']} |"
        )
    lines.append("")

    lines.append("## Quality Assessment")
    lines.append("")
    lines.append(f"- Aggregate quality: {data['quality']['aggregate_score']:.3f}")
    lines.append(f"- Mutual information: {data['quality']['mutual_information']:.3f}")
    if data["quality"]["low_quality_episodes"]:
        lines.append(f"- Low quality episodes: {data['quality']['low_quality_episodes']}")
    lines.append("")

    lines.append("## Prescriptions")
    lines.append("")
    if data["prescriptions"]:
        for p in data["prescriptions"]:
            lines.append(
                f"{p['priority']}. **[{p['task']}]** "
                f"(current: {p['current_capability']:.3f}) \u2014 "
                f"{p['instruction']}"
            )
    else:
        lines.append("No prescriptions \u2014 dataset looks good!")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    rng = np.random.default_rng(42)
    SPACES_DATA.mkdir(parents=True, exist_ok=True)

    quality_scores = _generate_quality_scores(rng)
    quality_mean = round(float(np.mean(quality_scores)), 3)
    quality_std = round(float(np.std(quality_scores)), 3)
    low_quality = [i for i, s in enumerate(quality_scores) if s < 0.4]

    prescriptions = _generate_prescriptions(CAPABILITIES)
    umap_points, cluster_labels = _generate_umap_points(rng)

    sorted_caps = sorted(CAPABILITIES, key=lambda c: c["score"], reverse=True)
    strengths = [c["task"] for c in sorted_caps[:3] if c["score"] >= 0.5]
    gaps = [c["task"] for c in sorted_caps if c["score"] < 0.5][-3:]

    report_data = {
        "dataset_name": "orbit_synthetic_benchmark",
        "num_episodes": 40,
        "num_frames": 400,
        "overall_coverage_score": 0.72,
        "strengths": strengths,
        "gaps": gaps,
        "capabilities": CAPABILITIES,
        "quality": {
            "aggregate_score": quality_mean,
            "mutual_information": 0.62,
            "low_quality_episodes": low_quality,
            "num_episodes_scored": 40,
        },
        "coverage": {
            "num_dense_regions": 3,
            "num_sparse_regions": 2,
            "overall_score": 0.72,
        },
        "prescriptions": prescriptions,
        "timestamp": "2026-03-06T00:00:00",
    }

    dashboard_data = {
        "summary_stats": {
            "dataset_name": "orbit_synthetic_benchmark",
            "num_episodes": 40,
            "num_frames": 400,
            "overall_coverage": 0.72,
            "aggregate_quality": quality_mean,
        },
        "coverage_umap": umap_points,
        "cluster_labels": cluster_labels,
        "capability_bars": [
            {"task": c["task"], "score": c["score"], "confidence": c["confidence"]}
            for c in CAPABILITIES
        ],
        "quality_histogram": {
            "scores": quality_scores,
            "mean": quality_mean,
            "std": quality_std,
        },
        "prescription_table": prescriptions,
    }

    profile = {"report": report_data, "dashboard": dashboard_data}

    json_path = SPACES_DATA / "demo_profile.json"
    json_path.write_text(json.dumps(profile, indent=2))
    print(f"Wrote {json_path} ({json_path.stat().st_size} bytes)")

    md_path = SPACES_DATA / "demo_report.md"
    md_path.write_text(_render_markdown(report_data))
    print(f"Wrote {md_path} ({md_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
