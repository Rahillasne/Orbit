"""Smoke tests for the Orbit Streamlit dashboard.

Tests that the dashboard app can be imported, data loading works,
and the Streamlit server can start headlessly.
"""

from __future__ import annotations

import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from orbit.logger.episode_logger import EpisodeLogger
from orbit.logger.schemas import LoggerConfig, Outcome

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_data_dir(tmp_path):
    """Generate a small synthetic deployment for testing."""
    data_dir = tmp_path / "dashboard_test_data"
    config = LoggerConfig(
        storage_dir=str(data_dir),
        task_name="test_task",
        robot_dof=6,
        save_images=True,
    )
    rng = np.random.default_rng(123)

    with EpisodeLogger(config) as logger:
        # 3 success episodes
        for _ in range(3):
            logger.start_episode()
            for step in range(10):
                img = Image.fromarray(rng.integers(100, 200, (32, 32, 3), dtype=np.uint8))
                logger.log_frame(
                    joint_positions=(rng.standard_normal(6) * 0.1).tolist(),
                    gripper_state=float(np.clip(rng.random(), 0, 1)),
                    action=(rng.standard_normal(6) * 0.01).tolist(),
                    reward=0.1,
                    images={"front": img},
                )
            logger.end_episode(outcome=Outcome.SUCCESS)

        # 2 failure episodes
        for _ in range(2):
            logger.start_episode()
            for step in range(8):
                img = Image.fromarray(rng.integers(10, 40, (32, 32, 3), dtype=np.uint8))
                logger.log_frame(
                    joint_positions=(rng.standard_normal(6) * 0.1).tolist(),
                    gripper_state=float(np.clip(rng.random(), 0, 1)),
                    action=(rng.standard_normal(6) * 0.001).tolist(),
                    reward=-0.5,
                    images={"front": img},
                )
            logger.end_episode(outcome=Outcome.FAILURE)

    return str(data_dir)


def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Tests: data loading
# ---------------------------------------------------------------------------


class TestDataLoader:
    """Test the dashboard data loading utilities."""

    def test_discover_sessions(self, synthetic_data_dir):
        from orbit.dashboard.data_loader import discover_sessions

        sessions = discover_sessions(synthetic_data_dir)
        assert len(sessions) == 1
        assert sessions[0]["episode_count"] == 5

    def test_load_session_episodes(self, synthetic_data_dir):
        from orbit.dashboard.data_loader import (
            discover_sessions,
            load_session_episodes,
        )

        sessions = discover_sessions(synthetic_data_dir)
        ep_dicts = load_session_episodes(sessions[0]["file_path"])
        assert len(ep_dicts) == 5

    def test_episodes_to_summary_df(self, synthetic_data_dir):
        from orbit.dashboard.data_loader import (
            deserialize_episodes,
            discover_sessions,
            episodes_to_summary_df,
            load_session_episodes,
        )

        sessions = discover_sessions(synthetic_data_dir)
        ep_dicts = load_session_episodes(sessions[0]["file_path"])
        episodes = deserialize_episodes(ep_dicts)
        df = episodes_to_summary_df(episodes)

        assert len(df) == 5
        assert "outcome" in df.columns
        assert "num_frames" in df.columns
        assert (df["outcome"] == "success").sum() == 3
        assert (df["outcome"] == "failure").sum() == 2

    def test_run_detector_pipeline(self, synthetic_data_dir):
        from orbit.dashboard.data_loader import (
            deserialize_episodes,
            discover_sessions,
            load_session_episodes,
            run_detector_pipeline,
        )

        sessions = discover_sessions(synthetic_data_dir)
        ep_dicts = load_session_episodes(sessions[0]["file_path"])
        episodes = deserialize_episodes(ep_dicts)
        results = run_detector_pipeline(episodes)

        assert len(results) == 5
        for r in results:
            assert "is_failure" in r
            assert "detections" in r

    def test_run_prescriber(self, synthetic_data_dir):
        from orbit.dashboard.data_loader import (
            deserialize_episodes,
            discover_sessions,
            load_session_episodes,
            run_detector_pipeline,
            run_prescriber,
        )

        sessions = discover_sessions(synthetic_data_dir)
        ep_dicts = load_session_episodes(sessions[0]["file_path"])
        episodes = deserialize_episodes(ep_dicts)
        detection_results = run_detector_pipeline(episodes)
        report = run_prescriber(detection_results, episodes)

        assert "prescriptions" in report
        assert "summary" in report
        assert isinstance(report["prescriptions"], list)
        # If ML deps are missing, prescriptions list may be empty
        # but the structure should still be valid

    def test_discover_sessions_missing_dir(self, tmp_path):
        from orbit.dashboard.data_loader import discover_sessions

        sessions = discover_sessions(str(tmp_path / "nonexistent"))
        assert sessions == []

    def test_get_failure_type_counts(self, synthetic_data_dir):
        from orbit.dashboard.data_loader import (
            deserialize_episodes,
            discover_sessions,
            get_failure_type_counts,
            load_session_episodes,
            run_detector_pipeline,
        )

        sessions = discover_sessions(synthetic_data_dir)
        ep_dicts = load_session_episodes(sessions[0]["file_path"])
        episodes = deserialize_episodes(ep_dicts)
        results = run_detector_pipeline(episodes)
        counts = get_failure_type_counts(results)

        assert isinstance(counts, dict)


# ---------------------------------------------------------------------------
# Tests: module imports
# ---------------------------------------------------------------------------


class TestImports:
    """Verify that dashboard modules can be imported without errors."""

    def test_import_data_loader(self):
        import orbit.dashboard.data_loader  # noqa: F401

    def test_import_app(self):
        import orbit.dashboard.app  # noqa: F401

    def test_import_cli(self):
        import orbit.cli  # noqa: F401


# ---------------------------------------------------------------------------
# Tests: Streamlit headless smoke test
# ---------------------------------------------------------------------------


class TestStreamlitServer:
    """Launch Streamlit headlessly and verify it starts."""

    @pytest.mark.slow
    def test_streamlit_starts_headlessly(self, synthetic_data_dir):
        """Verify the Streamlit app starts and responds to health checks."""
        port = _find_free_port()
        app_path = str(Path(__file__).parent.parent / "orbit" / "dashboard" / "app.py")

        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                app_path,
                f"--server.port={port}",
                "--server.headless=true",
                "--",
                f"--data-dir={synthetic_data_dir}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            # Wait for Streamlit to start (up to 30 seconds)
            health_url = f"http://localhost:{port}/_stcore/health"
            started = False
            for _ in range(30):
                try:
                    resp = urllib.request.urlopen(health_url, timeout=2)
                    if resp.status == 200:
                        started = True
                        break
                except Exception:
                    time.sleep(1)

            assert started, "Streamlit server failed to start within 30 seconds"

            # Verify the main page loads
            main_url = f"http://localhost:{port}/"
            resp = urllib.request.urlopen(main_url, timeout=5)
            assert resp.status == 200

        finally:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=10)
