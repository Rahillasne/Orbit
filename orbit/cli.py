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


if __name__ == "__main__":
    cli()
