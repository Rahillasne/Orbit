"""Unified CLI for Orbit — dataset profiling, gap analysis, and deployment diagnostics."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click


@click.group()
@click.version_option(package_name="orbit")
def cli() -> None:
    """Orbit — Analyze robot datasets for task capability and coverage gaps."""


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
@click.option(
    "--fast",
    is_flag=True,
    help="Use fast CPU-friendly embeddings (OpenCLIP ViT-B/32).",
)
@click.option(
    "--embedding-model",
    type=click.Choice(["auto", "r3m", "siglip", "openclip"]),
    default="auto",
    help="Embedding model for visual features. 'auto' prefers R3M if available.",
)
@click.option(
    "--report-card",
    is_flag=True,
    help="Generate graded report card output instead of raw scores.",
)
def profile(
    data_dir: str | None,
    hub_repo: str | None,
    tasks: tuple[str, ...],
    output: str | None,
    fmt: str,
    fast: bool,
    embedding_model: str,
    report_card: bool,
) -> None:
    """Profile a robot dataset for task capabilities and coverage gaps."""
    if not data_dir and not hub_repo:
        raise click.UsageError("Provide either --data-dir or --hub-repo.")
    if data_dir and hub_repo:
        raise click.UsageError("Provide only one of --data-dir or --hub-repo, not both.")

    from orbit.profile.profiler import DatasetProfiler
    from orbit.profile.report import ProfileReporter

    profiler = DatasetProfiler(embedding_model=embedding_model, fast_mode=fast)
    task_list = list(tasks) if tasks else None

    source = data_dir or hub_repo
    click.echo(f"Profiling {'local dataset' if data_dir else 'HuggingFace dataset'}: {source}")
    if fast:
        click.echo("Using fast embedding mode (OpenCLIP ViT-B/32)")
    elif embedding_model != "auto":
        click.echo(f"Using embedding model: {embedding_model}")

    if data_dir:
        result = profiler.profile(data_dir, task_descriptions=task_list)
    else:
        result = profiler.profile_from_hub(hub_repo, task_descriptions=task_list)

    # Report card output
    if report_card:
        from orbit.profile.report_card import ReportCardGenerator

        generator = ReportCardGenerator()
        card = generator.generate(result)
        if fmt == "json":
            import json

            card_dict = card.to_dict()
            # Attach V2 prediction if available
            if result.predicted_success_rate is not None:
                card_dict["predicted_success_rate"] = round(result.predicted_success_rate, 3)
                card_dict["prediction_ci"] = [
                    round(result.prediction_ci_low, 3),
                    round(result.prediction_ci_high, 3),
                ]
                card_dict["prediction_confidence"] = result.prediction_confidence
            report_output = json.dumps(card_dict, indent=2)
        else:
            report_output = generator.render_cli(card)
            # Append V2 prediction box
            if result.predicted_success_rate is not None:
                pred_pct = int(result.predicted_success_rate * 100)
                ci_lo = int(result.prediction_ci_low * 100)
                ci_hi = int(result.prediction_ci_high * 100)
                conf = (result.prediction_confidence or "unknown").capitalize()
                report_output += (
                    f"\n╔══════════════════════════════════════════╗"
                    f"\n║  Predicted Success Rate: {pred_pct}% (CI: {ci_lo}-{ci_hi}%)"
                    f"\n║  Confidence: {conf:<28}║"
                    f"\n╚══════════════════════════════════════════╝"
                )

        if output:
            Path(output).write_text(report_output)
            click.echo(f"Report card saved to {output}")
        else:
            click.echo(report_output)
        return

    # Standard report output
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


@cli.command(name="sim2real")
@click.option(
    "--sim-data",
    required=True,
    type=click.Path(exists=True),
    help="Path to simulation dataset directory.",
)
@click.option(
    "--real-data",
    required=True,
    type=click.Path(exists=True),
    help="Path to real (or expected-real) dataset directory.",
)
@click.option("--tasks", multiple=True, help="Task descriptions (repeat for multiple).")
@click.option("--output", "-o", type=click.Path(), default=None, help="Output JSON file path.")
def sim2real(
    sim_data: str,
    real_data: str,
    tasks: tuple[str, ...],
    output: str | None,
) -> None:
    """Analyze sim-to-real transfer readiness between two datasets."""
    from orbit.sim2real_profiler import Sim2RealProfiler

    profiler = Sim2RealProfiler()
    task_list = list(tasks) if tasks else None

    click.echo(f"Analyzing sim-to-real transfer: {sim_data} → {real_data}")
    report = profiler.analyze(sim_data, real_data, task_descriptions=task_list)

    # Print summary
    click.echo(f"\nOverall transfer score: {report.overall_transfer_score:.3f}")
    for task, scores in report.per_task_scores.items():
        click.echo(f"  {task}: {scores['score']:.3f}")

    if report.prescription:
        click.echo("\nPrescriptions:")
        for p in report.prescription:
            click.echo(f"  {p['priority']}. [{p['task']}] {p['action']}")

    result_json = report.to_json()
    if output:
        Path(output).write_text(result_json)
        click.echo(f"\nReport saved to {output}")
    else:
        click.echo(f"\n{result_json}")


@cli.command(name="validate-benchmark")
@click.option("--cache-dir", default=None, type=click.Path(), help="Cache directory for profiles.")
@click.option("--output", "-o", default=None, type=click.Path(), help="Output file path.")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["markdown", "json"]),
    default="markdown",
    help="Report format.",
)
@click.option("--max-episodes", default=None, type=int, help="Override max episodes per dataset.")
@click.option("--dataset", multiple=True, help="Specific dataset IDs to validate (default: all).")
@click.option("--force", is_flag=True, help="Ignore cached scores and re-profile.")
@click.option("--device", default="cpu", help="Torch device (cpu or cuda).")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
def validate_benchmark(
    cache_dir: str | None,
    output: str | None,
    fmt: str,
    max_episodes: int | None,
    dataset: tuple[str, ...],
    force: bool,
    device: str,
    verbose: bool,
) -> None:
    """Validate ORBIT profiler predictions against known benchmark results."""
    import logging

    from orbit.benchmarks.validate_profiler import BenchmarkValidator

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    validator = BenchmarkValidator(
        cache_dir=cache_dir,
        max_episodes=max_episodes,
        device=device,
    )
    dataset_ids = list(dataset) if dataset else None
    report = validator.run(dataset_ids=dataset_ids, force=force)

    if fmt == "json":
        text = report.to_json()
    else:
        text = report.to_markdown()

    if output:
        Path(output).write_text(text)
        click.echo(f"Report saved to {output}")
    else:
        click.echo(text)


if __name__ == "__main__":
    cli()
