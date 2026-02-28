"""CLI entry point for ``orbit-detect``."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from uuid import UUID

import click

from orbit.detector.heuristic import (
    DetectorPipeline,
    GripperDropDetector,
    OutOfBoundsDetector,
    PipelineResult,
    RewardThresholdDetector,
    StallDetector,
    TimeoutDetector,
    load_pipeline_from_yaml,
)
from orbit.logger.schemas import LoggerConfig
from orbit.logger.storage import HDF5Storage


def _default_pipeline() -> DetectorPipeline:
    """Build a pipeline with all detectors using default configs."""
    return DetectorPipeline(
        detectors=[
            GripperDropDetector(),
            StallDetector(),
            OutOfBoundsDetector(),
            TimeoutDetector(),
            RewardThresholdDetector(),
        ]
    )


def _load_episodes_from_h5(path: Path) -> list:
    """Load all episodes from an HDF5 session file."""
    config = LoggerConfig(storage_dir=str(path.parent))
    storage = HDF5Storage(config)

    # Extract session_id from filename pattern session_{uuid}.h5
    stem = path.stem
    session_id = UUID(stem.replace("session_", ""))

    episode_pairs = storage.list_episodes(session_id=session_id)
    episodes = []
    for sid, eid in episode_pairs:
        episodes.append(storage.load_episode(sid, eid))
    storage.close()
    return episodes


def _print_table(results: list[PipelineResult]) -> None:
    """Print a summary table to stdout."""
    click.echo(f"{'Episode ID':<40} {'Fail?':<7} {'Prob':>6} {'Detections':>10}")
    click.echo("-" * 70)
    for r in results:
        ep_str = str(r.episode_id)[:36]
        fail = "YES" if r.is_failure else "no"
        click.echo(f"{ep_str:<40} {fail:<7} {r.failure_probability:>5.0%} {len(r.detections):>10}")

    # Summary
    n_fail = sum(1 for r in results if r.is_failure)
    click.echo(f"\n{len(results)} episodes analyzed, {n_fail} failures detected.")

    # Detail for failures
    for r in results:
        if r.is_failure:
            click.echo(f"\n--- Episode {str(r.episode_id)[:8]}... ---")
            for name, summary in r.detector_summaries.items():
                if "No failures" not in summary:
                    click.echo(summary)


def _print_json(results: list[PipelineResult]) -> None:
    """Print results as JSON."""
    output = []
    for r in results:
        output.append(
            {
                "episode_id": str(r.episode_id),
                "is_failure": r.is_failure,
                "failure_probability": r.failure_probability,
                "detections": [
                    {
                        "detector_name": d.detector_name,
                        "confidence": d.confidence,
                        "frame_idx": d.frame_idx,
                        "description": d.description,
                    }
                    for d in r.detections
                ],
            }
        )
    click.echo(json.dumps(output, indent=2))


@click.command("detect")
@click.option(
    "--session",
    required=True,
    type=click.Path(exists=True),
    help="Path to a session .h5 file.",
)
@click.option(
    "--config",
    "config_path",
    required=False,
    type=click.Path(exists=True),
    default=None,
    help="Path to a detector YAML config file. Uses defaults if omitted.",
)
@click.option("--json-output", is_flag=True, help="Output results as JSON.")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
def main(
    session: str,
    config_path: str | None,
    json_output: bool,
    verbose: bool,
) -> None:
    """Run heuristic failure detection on a saved session file."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    # 1. Build pipeline
    if config_path:
        pipeline = load_pipeline_from_yaml(config_path)
    else:
        pipeline = _default_pipeline()

    # 2. Load episodes from session file
    session_path = Path(session)
    episodes = _load_episodes_from_h5(session_path)

    if not episodes:
        click.echo("No episodes found in session file.")
        sys.exit(0)

    # 3. Run detection
    results = pipeline.run_batch(episodes)

    # 4. Output
    if json_output:
        _print_json(results)
    else:
        _print_table(results)
