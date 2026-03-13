#!/usr/bin/env python3
"""Run ORBIT validation pipeline against LeRobot datasets.

Downloads datasets, runs the profiler and sim2real analysis, and exports
results as JSON, CSV, and plots.

Usage:
    python scripts/run_lerobot_validation.py --output results/
    python scripts/run_lerobot_validation.py --datasets lerobot/pusht,lerobot/xarm_lift_medium_replay
    python scripts/run_lerobot_validation.py --skip-sim2real --max-episodes 10
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

log = logging.getLogger("orbit.validation")

# ---------------------------------------------------------------------------
# Default datasets and task descriptions
# ---------------------------------------------------------------------------

DEFAULT_DATASETS: dict[str, list[str]] = {
    "lerobot/pusht": ["push block to target", "precise positioning"],
    "lerobot/aloha_sim_transfer_cube_human": [
        "pick up cube",
        "bimanual handover",
        "place cube",
    ],
    "lerobot/xarm_lift_medium_replay": [
        "grasp object",
        "lift object",
        "stable hold",
    ],
    "lerobot/aloha_sim_insertion_human": [
        "align peg",
        "insert peg",
        "precision manipulation",
    ],
    "lerobot/umi_cup_in_the_wild": [
        "grasp cup",
        "pour liquid",
        "place cup",
    ],
}

SIM2REAL_PAIRS: list[tuple[str, str, list[str]]] = [
    (
        "lerobot/aloha_sim_transfer_cube_human",
        "lerobot/aloha_sim_insertion_human",
        ["bimanual manipulation", "object transfer"],
    ),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def detect_device() -> str:
    import torch

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        log.info("GPU detected: %s (%.1f GB)", name, mem)
        return "cuda"
    log.info("No GPU detected — running on CPU")
    return "cpu"


def clear_gpu_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def profile_dataset(
    repo_id: str,
    tasks: list[str],
    profiler,
    max_episodes: int,
    fps_sample: int,
    cache_dir: Path,
) -> dict | None:
    """Download, convert, and profile a single dataset. Returns result dict or None."""
    from orbit.profile.loaders import DatasetLoader

    short_name = repo_id.split("/", 1)[-1]
    output_dir = cache_dir / repo_id.replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Convert from LeRobot format
    log.info("[%s] Converting (max %d episodes)...", short_name, max_episodes)
    try:
        DatasetLoader.from_lerobot(
            repo_id, str(output_dir), max_episodes=max_episodes, fps_sample=fps_sample
        )
    except Exception as exc:
        log.error("[%s] Download/conversion failed: %s", short_name, exc)
        return None

    # Step 2: Profile (with OOM retry)
    log.info("[%s] Profiling with tasks: %s", short_name, tasks)
    t0 = time.time()
    profile = None
    episodes_tried = max_episodes

    for attempt in range(2):
        try:
            profile = profiler.profile(data_dir=str(output_dir), task_descriptions=tasks)
            break
        except (RuntimeError,) as exc:
            if "CUDA out of memory" in str(exc) or "OutOfMemoryError" in type(exc).__name__:
                clear_gpu_cache()
                episodes_tried = max(1, episodes_tried // 2)
                log.warning(
                    "[%s] GPU OOM — retrying with %d episodes (attempt %d/2)",
                    short_name,
                    episodes_tried,
                    attempt + 2,
                )
                # Re-convert with fewer episodes
                try:
                    DatasetLoader.from_lerobot(
                        repo_id,
                        str(output_dir),
                        max_episodes=episodes_tried,
                        fps_sample=fps_sample,
                    )
                except Exception:
                    log.error("[%s] Re-conversion failed after OOM", short_name)
                    return None
            else:
                log.error("[%s] Profiling failed: %s", short_name, exc)
                return None
        except Exception as exc:
            log.error("[%s] Profiling failed: %s", short_name, exc)
            return None

    if profile is None:
        log.error("[%s] Profiling failed after retries", short_name)
        return None

    elapsed = time.time() - t0
    log.info(
        "[%s] Done in %.1fs — %d episodes, %d frames, coverage=%.3f, quality=%.3f",
        short_name,
        elapsed,
        profile.num_episodes,
        profile.num_frames,
        profile.coverage.overall_coverage_score,
        profile.quality.aggregate_score,
    )
    clear_gpu_cache()

    return {
        "repo_id": repo_id,
        "profile": profile,
        "data_dir": str(output_dir),
        "elapsed": elapsed,
    }


def run_sim2real(
    sim_dir: str,
    real_dir: str,
    tasks: list[str],
    device: str,
) -> dict | None:
    """Run sim2real analysis. Returns report dict or None."""
    from orbit.sim2real_profiler import Sim2RealProfiler

    log.info("Sim2Real: comparing datasets...")
    try:
        s2r = Sim2RealProfiler(
            embedding_model="google/siglip-base-patch16-224", device=device
        )
        t0 = time.time()
        report = s2r.analyze(
            sim_dir=sim_dir, real_dir=real_dir, task_descriptions=tasks
        )
        elapsed = time.time() - t0
        log.info(
            "Sim2Real done in %.1fs — transfer score: %.3f",
            elapsed,
            report.overall_transfer_score,
        )
        clear_gpu_cache()
        return {"report": report, "elapsed": elapsed}
    except Exception as exc:
        log.error("Sim2Real analysis failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def export_json_reports(results: dict, output_dir: Path) -> None:
    """Save per-dataset JSON reports and combined summary."""
    from orbit.profile.report import ProfileReporter

    reporter = ProfileReporter()
    summaries = []

    for repo_id, res in results.items():
        profile = res["profile"]
        name = repo_id.split("/", 1)[-1]
        report_dict = reporter.generate_report(profile, format="dict")
        report_path = output_dir / f"{name}_report.json"
        with open(report_path, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)
        log.info("Saved %s", report_path)

        summaries.append(
            {
                "dataset": repo_id,
                "num_episodes": profile.num_episodes,
                "num_frames": profile.num_frames,
                "coverage_score": round(profile.coverage.overall_coverage_score, 3),
                "quality_score": round(profile.quality.aggregate_score, 3),
                "capabilities": [
                    {"task": c.task_description, "score": round(c.score, 3)}
                    for c in profile.capabilities
                ],
                "num_prescriptions": len(profile.prescriptions),
                "profiling_time_s": round(res["elapsed"], 1),
            }
        )

    combined = {
        "orbit_version": _get_orbit_version(),
        "num_datasets": len(summaries),
        "embedding_model": "google/siglip-base-patch16-224",
        "datasets": summaries,
    }
    combined_path = output_dir / "combined_summary.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    log.info("Saved %s", combined_path)


def export_csv(results: dict, output_dir: Path) -> None:
    """Save benchmark_results.csv."""
    rows = []
    for repo_id, res in results.items():
        profile = res["profile"]
        top_caps = ", ".join(
            f"{c.task_description} ({c.score:.2f})"
            for c in sorted(profile.capabilities, key=lambda c: c.score, reverse=True)
            if c.score >= 0.5
        )
        gaps = ", ".join(
            f"{c.task_description} ({c.score:.2f})"
            for c in profile.capabilities
            if c.score < 0.5
        )
        rows.append(
            {
                "Dataset": repo_id.split("/", 1)[-1],
                "Episodes": profile.num_episodes,
                "Frames": profile.num_frames,
                "Coverage": round(profile.coverage.overall_coverage_score, 3),
                "Quality": round(profile.quality.aggregate_score, 3),
                "Top Capabilities": top_caps or "None",
                "Gaps Found": gaps or "None",
                "Time (s)": round(res["elapsed"], 1),
            }
        )

    csv_path = output_dir / "benchmark_results.csv"

    # Try pandas first, fall back to csv module
    try:
        import pandas as pd

        pd.DataFrame(rows).to_csv(csv_path, index=False)
    except ImportError:
        if rows:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

    log.info("Saved %s", csv_path)


def export_plots(results: dict, output_dir: Path) -> None:
    """Generate radar charts and heatmap. Skips if matplotlib unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        log.warning("matplotlib not installed — skipping plot generation")
        return

    profiles = {k: v["profile"] for k, v in results.items()}

    # -- Radar charts --
    datasets_with_caps = {k: v for k, v in profiles.items() if v.capabilities}
    n = len(datasets_with_caps)
    if n > 0:
        cols = min(n, 3)
        rows_count = (n + cols - 1) // cols
        fig, axes = plt.subplots(
            rows_count, cols,
            figsize=(6 * cols, 5 * rows_count),
            subplot_kw={"projection": "polar"},
        )
        if n == 1:
            axes = [axes]
        else:
            axes = list(np.array(axes).flat)

        for idx, (repo_id, profile) in enumerate(datasets_with_caps.items()):
            ax = axes[idx]
            labels = [c.task_description for c in profile.capabilities]
            scores = [c.score for c in profile.capabilities]
            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
            scores_plot = scores + [scores[0]]
            angles_plot = angles + [angles[0]]

            ax.fill(angles_plot, scores_plot, alpha=0.25, color="steelblue")
            ax.plot(angles_plot, scores_plot, "o-", color="steelblue", linewidth=2)
            ax.set_xticks(angles)
            ax.set_xticklabels(labels, size=8)
            ax.set_ylim(0, 1)
            ax.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], size=7)
            ax.set_title(
                repo_id.split("/", 1)[-1], pad=20, fontsize=11, fontweight="bold"
            )

        for idx in range(n, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(
            "ORBIT Capability Profiles per LeRobot Dataset",
            fontsize=14, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        path = output_dir / "capability_radar_charts.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved %s", path)

    # -- Heatmap --
    if profiles:
        try:
            import pandas as pd
        except ImportError:
            log.warning("pandas not installed — skipping heatmap")
            return

        heatmap_data = {}
        for repo_id, profile in profiles.items():
            name = repo_id.split("/", 1)[-1]
            row = {
                "Coverage": profile.coverage.overall_coverage_score,
                "Quality": profile.quality.aggregate_score,
            }
            for cap in profile.capabilities:
                row[cap.task_description] = cap.score
            heatmap_data[name] = row

        heatmap_df = pd.DataFrame(heatmap_data).T.fillna(0)

        fig, ax = plt.subplots(
            figsize=(max(10, len(heatmap_df.columns) * 1.2), len(heatmap_df) * 0.8 + 2)
        )
        im = ax.imshow(heatmap_df.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(heatmap_df.columns)))
        ax.set_xticklabels(heatmap_df.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(heatmap_df.index)))
        ax.set_yticklabels(heatmap_df.index, fontsize=10)

        for i in range(len(heatmap_df.index)):
            for j in range(len(heatmap_df.columns)):
                val = heatmap_df.values[i, j]
                color = "white" if val < 0.4 or val > 0.8 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)

        plt.colorbar(im, ax=ax, label="Score", shrink=0.8)
        ax.set_title(
            "ORBIT Profiler Scores Across LeRobot Datasets",
            fontsize=13, fontweight="bold", pad=15,
        )
        plt.tight_layout()
        path = output_dir / "cross_dataset_heatmap.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved %s", path)


def print_summary_table(results: dict, s2r_result: dict | None) -> None:
    """Print a formatted summary table using rich."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # -- Dataset results table --
    table = Table(title="ORBIT Validation Results", show_lines=True)
    table.add_column("Dataset", style="bold cyan")
    table.add_column("Episodes", justify="right")
    table.add_column("Frames", justify="right")
    table.add_column("Coverage", justify="right")
    table.add_column("Quality", justify="right")
    table.add_column("Top Capability", style="green")
    table.add_column("Time (s)", justify="right")

    for repo_id, res in results.items():
        profile = res["profile"]
        top_cap = max(profile.capabilities, key=lambda c: c.score) if profile.capabilities else None
        top_cap_str = f"{top_cap.task_description} ({top_cap.score:.2f})" if top_cap else "—"

        table.add_row(
            repo_id.split("/", 1)[-1],
            str(profile.num_episodes),
            str(profile.num_frames),
            f"{profile.coverage.overall_coverage_score:.3f}",
            f"{profile.quality.aggregate_score:.3f}",
            top_cap_str,
            f"{res['elapsed']:.1f}",
        )

    console.print()
    console.print(table)

    # -- Sim2Real result --
    if s2r_result:
        report = s2r_result["report"]
        console.print()
        console.print(
            f"[bold]Sim2Real Transfer Score:[/bold] "
            f"[{'green' if report.overall_transfer_score >= 0.5 else 'red'}]"
            f"{report.overall_transfer_score:.3f}[/]"
            f"  ({s2r_result['elapsed']:.1f}s)"
        )

    # -- Summary --
    total_time = sum(r["elapsed"] for r in results.values())
    console.print()
    console.print(
        f"[bold]Profiled {len(results)} dataset(s) in {total_time:.1f}s total[/bold]"
    )


def _get_orbit_version() -> str:
    try:
        import orbit

        return getattr(orbit, "__version__", "unknown")
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ORBIT validation pipeline against LeRobot datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Output directory for results (default: results/)",
    )
    parser.add_argument(
        "--datasets",
        default="all",
        help='Comma-separated dataset repo IDs, or "all" for all defaults (default: all)',
    )
    parser.add_argument(
        "--skip-sim2real",
        action="store_true",
        help="Skip sim-to-real transfer analysis",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=20,
        help="Max episodes per dataset (default: 20)",
    )
    parser.add_argument(
        "--fps-sample",
        type=int,
        default=2,
        help="FPS subsampling rate during conversion (default: 2)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device: cuda or cpu (default: auto-detect)",
    )
    parser.add_argument(
        "--cache-dir",
        default="orbit_data",
        help="Cache directory for converted datasets (default: orbit_data/)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    # Resolve datasets
    if args.datasets == "all":
        datasets = dict(DEFAULT_DATASETS)
    else:
        datasets = {}
        for ds in args.datasets.split(","):
            ds = ds.strip()
            if ds in DEFAULT_DATASETS:
                datasets[ds] = DEFAULT_DATASETS[ds]
            else:
                log.warning(
                    "Dataset %s not in defaults — using generic tasks. "
                    "Provide tasks via DEFAULT_DATASETS if needed.",
                    ds,
                )
                datasets[ds] = ["general manipulation"]

    # Device
    device = args.device or detect_device()

    # Output dir
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    log.info("ORBIT Validation Pipeline")
    log.info("  Datasets:     %d", len(datasets))
    log.info("  Device:       %s", device)
    log.info("  Max episodes: %d", args.max_episodes)
    log.info("  Output:       %s", output_dir)

    # -- Import profiler (lazy, to surface import errors early) --
    try:
        from orbit.profile.profiler import DatasetProfiler
    except ImportError as exc:
        log.error(
            "Failed to import ORBIT profiler. Install with: "
            "pip install orbit-robotics[full]\n  %s",
            exc,
        )
        return 1

    profiler = DatasetProfiler(
        embedding_model="google/siglip-base-patch16-224",
        device=device,
    )

    # -- Profile each dataset --
    results: dict[str, dict] = {}
    for repo_id, tasks in datasets.items():
        res = profile_dataset(
            repo_id=repo_id,
            tasks=tasks,
            profiler=profiler,
            max_episodes=args.max_episodes,
            fps_sample=args.fps_sample,
            cache_dir=cache_dir,
        )
        if res is not None:
            results[repo_id] = res

    if not results:
        log.error("No datasets were profiled successfully. Exiting.")
        return 1

    log.info("Profiled %d/%d datasets successfully.", len(results), len(datasets))

    # -- Sim2Real analysis --
    s2r_result = None
    if not args.skip_sim2real:
        for sim_repo, real_repo, s2r_tasks in SIM2REAL_PAIRS:
            if sim_repo in results and real_repo in results:
                s2r_result = run_sim2real(
                    sim_dir=results[sim_repo]["data_dir"],
                    real_dir=results[real_repo]["data_dir"],
                    tasks=s2r_tasks,
                    device=device,
                )
                if s2r_result is not None:
                    report_path = output_dir / "sim2real_report.json"
                    with open(report_path, "w") as f:
                        f.write(s2r_result["report"].to_json(indent=2))
                    log.info("Saved %s", report_path)
            else:
                missing = [
                    r for r in (sim_repo, real_repo) if r not in results
                ]
                log.warning(
                    "Skipping sim2real pair — missing datasets: %s", missing
                )

    # -- Export results --
    log.info("Exporting results to %s ...", output_dir)
    export_json_reports(results, output_dir)
    export_csv(results, output_dir)
    export_plots(results, output_dir)

    # -- Print summary --
    print_summary_table(results, s2r_result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
