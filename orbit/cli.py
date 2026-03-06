"""Unified CLI for Orbit — ``orbit dashboard``, ``orbit detect``."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click


@click.group()
@click.version_option(package_name="orbit")
def cli() -> None:
    """Orbit — Robot learning failure analysis toolkit."""


@cli.command()
@click.option(
    "--data-dir",
    default="./orbit_data",
    type=click.Path(),
    help="Directory containing session .h5 files.",
)
@click.option("--port", default=8501, type=int, help="Streamlit server port.")
def dashboard(data_dir: str, port: int) -> None:
    """Launch the Orbit Streamlit dashboard."""
    app_path = str(Path(__file__).parent / "dashboard" / "app.py")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_path,
        f"--server.port={port}",
        "--server.headless=true",
        "--",
        f"--data-dir={data_dir}",
    ]
    subprocess.run(cmd, check=True)


@cli.command()
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
    help="Path to a detector YAML config file.",
)
@click.option("--json-output", is_flag=True, help="Output results as JSON.")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
def detect(session: str, config_path: str | None, json_output: bool, verbose: bool) -> None:
    """Run heuristic failure detection on a saved session."""
    from orbit.detector.cli import main as detect_main

    # Build sys.argv for the click command in detector.cli
    args = ["detect", "--session", session]
    if config_path:
        args.extend(["--config", config_path])
    if json_output:
        args.append("--json-output")
    if verbose:
        args.append("--verbose")

    detect_main(args, standalone_mode=False)


@cli.command()
@click.option(
    "--data-dir",
    type=click.Path(),
    default=None,
    help="Local dataset directory to profile.",
)
@click.option("--hub-repo", default=None, help="HuggingFace Hub repo ID (e.g. lerobot/aloha_sim).")
@click.option("--tasks", multiple=True, help="Task descriptions (repeat for multiple).")
@click.option("--output", "-o", type=click.Path(), default=None, help="Output file path.")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["markdown", "json"]),
    default="markdown",
    help="Report format.",
)
def profile(
    data_dir: str | None,
    hub_repo: str | None,
    tasks: tuple[str, ...],
    output: str | None,
    fmt: str,
) -> None:
    """Profile a robot dataset for task capabilities and coverage gaps."""
    if not data_dir and not hub_repo:
        raise click.UsageError("Provide either --data-dir or --hub-repo.")
    if data_dir and hub_repo:
        raise click.UsageError("Provide only one of --data-dir or --hub-repo, not both.")

    from orbit.profile.profiler import DatasetProfiler
    from orbit.profile.report import ProfileReporter

    profiler = DatasetProfiler()
    task_list = list(tasks) if tasks else None

    if data_dir:
        click.echo(f"Profiling local dataset: {data_dir}")
        result = profiler.profile(data_dir, task_descriptions=task_list)
    else:
        click.echo(f"Profiling HuggingFace dataset: {hub_repo}")
        result = profiler.profile_from_hub(hub_repo, task_descriptions=task_list)

    reporter = ProfileReporter()
    report = reporter.generate_report(result, format=fmt)

    if output:
        Path(output).write_text(report if isinstance(report, str) else str(report))
        click.echo(f"Report saved to {output}")
    else:
        click.echo(report if isinstance(report, str) else str(report))


@cli.command(name="profile-compare")
@click.option(
    "--dataset-a",
    required=True,
    type=click.Path(exists=True),
    help="Path to first dataset.",
)
@click.option(
    "--dataset-b",
    required=True,
    type=click.Path(exists=True),
    help="Path to second dataset.",
)
@click.option("--tasks", multiple=True, help="Task descriptions (repeat for multiple).")
@click.option("--output", "-o", type=click.Path(), default=None, help="Output file path.")
def profile_compare(
    dataset_a: str,
    dataset_b: str,
    tasks: tuple[str, ...],
    output: str | None,
) -> None:
    """Compare two dataset profiles side-by-side."""
    from orbit.profile.capability import CapabilityScorer
    from orbit.profile.profiler import DatasetProfiler
    from orbit.profile.report import ProfileReporter

    profiler = DatasetProfiler()
    reporter = ProfileReporter()
    task_list = list(tasks) if tasks else None

    click.echo(f"Profiling dataset A: {dataset_a}")
    profile_a = profiler.profile(dataset_a, task_descriptions=task_list)

    click.echo(f"Profiling dataset B: {dataset_b}")
    profile_b = profiler.profile(dataset_b, task_descriptions=task_list)

    scorer = CapabilityScorer()
    comparison = scorer.compare_profiles(profile_a, profile_b)

    report_a = reporter.generate_report(profile_a, format="dict")
    report_b = reporter.generate_report(profile_b, format="dict")

    import json

    combined = {
        "dataset_a": report_a,
        "dataset_b": report_b,
        "comparison": comparison,
    }
    result_str = json.dumps(combined, indent=2, default=str)

    if output:
        Path(output).write_text(result_str)
        click.echo(f"Comparison saved to {output}")
    else:
        click.echo(result_str)


if __name__ == "__main__":
    cli()
