"""Shared pytest fixtures for Orbit tests."""

import os

# Prevent FAISS segfault on macOS when torch is imported first:
# both ship competing OpenMP runtimes that crash during threaded search.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest
from PIL import Image

from orbit.logger.schemas import (
    Episode,
    EpisodeFrame,
    EpisodeRecord,
    LoggerConfig,
    Outcome,
    StepRecord,
)


# ---------------------------------------------------------------------------
# Directory fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_storage_dir(tmp_path):
    """Provide a temporary directory for storage tests."""
    return tmp_path / "orbit_test_data"


# ---------------------------------------------------------------------------
# New-model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def new_config(tmp_storage_dir):
    """LoggerConfig for use with the new EpisodeLogger."""
    return LoggerConfig(
        storage_dir=str(tmp_storage_dir),
        task_name="test_pick_place",
        robot_dof=6,
    )


@pytest.fixture
def sample_frame():
    """A single EpisodeFrame."""
    return EpisodeFrame(
        timestamp=1000.0,
        joint_positions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        gripper_state=0.5,
        action=[0.01, -0.02, 0.03, -0.01, 0.0, 0.0],
        reward=0.1,
    )


@pytest.fixture
def sample_new_episode():
    """A complete Episode with 20 frames."""
    frames = []
    for i in range(20):
        frames.append(
            EpisodeFrame(
                timestamp=1000.0 + i * 0.033,
                joint_positions=(np.random.randn(6) * 0.1).tolist(),
                gripper_state=float(np.clip(np.random.random(), 0, 1)),
                action=(np.random.randn(6) * 0.01).tolist(),
                reward=0.1 if i < 15 else 1.0,
            )
        )
    return Episode(
        task_name="test_pick_place",
        robot_id="test_robot",
        frames=frames,
        outcome=Outcome.SUCCESS,
    )


# ---------------------------------------------------------------------------
# Legacy fixtures (kept for detector / analyzer / prescriber tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def parquet_config(tmp_storage_dir):
    """LoggerConfig using Parquet storage in a temp directory (legacy)."""
    return LoggerConfig(
        storage_dir=str(tmp_storage_dir),
        task_name="test_pick_place",
    )


@pytest.fixture
def hdf5_config(tmp_storage_dir):
    """LoggerConfig using HDF5 storage in a temp directory (legacy)."""
    return LoggerConfig(
        storage_dir=str(tmp_storage_dir),
        task_name="test_pick_place",
    )


@pytest.fixture
def sample_step():
    """A single sample StepRecord (legacy)."""
    return StepRecord(
        step_index=0,
        timestamp=1000.0,
        observation={"joint_positions": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]},
        action=[0.01, -0.02, 0.03, -0.01, 0.0, 0.0],
        reward=0.1,
        done=False,
        images={
            "front": Image.fromarray(
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            )
        },
    )


def _make_episode(episode_id, n_steps, reward_fn, action_fn, success, task="test_pick_place"):
    """Helper to build a legacy EpisodeRecord."""
    steps = []
    for i in range(n_steps):
        step = StepRecord(
            step_index=i,
            timestamp=1000.0 + i * 0.033,
            observation={"joint_positions": (np.random.randn(6) * 0.1).tolist()},
            action=action_fn(i),
            reward=reward_fn(i),
            done=(i == n_steps - 1),
            images={
                "front": Image.fromarray(
                    np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                )
            },
        )
        steps.append(step)

    return EpisodeRecord(
        episode_id=episode_id,
        task=task,
        steps=steps,
        total_reward=sum(s.reward for s in steps),
        success=success,
        num_steps=n_steps,
    )


@pytest.fixture
def sample_episode():
    """A complete successful sample episode with 20 steps (legacy)."""
    return _make_episode(
        episode_id=0,
        n_steps=20,
        reward_fn=lambda i: 0.1 if i < 15 else 1.0,
        action_fn=lambda i: (np.random.randn(6) * 0.1).tolist(),
        success=True,
    )


@pytest.fixture
def failure_episode():
    """A failure episode (low reward, stuck actions) (legacy)."""
    return _make_episode(
        episode_id=1,
        n_steps=10,
        reward_fn=lambda _: -0.5,
        action_fn=lambda _: [0.0] * 6,
        success=False,
    )


# ---------------------------------------------------------------------------
# New-model fixtures with image files on disk
# ---------------------------------------------------------------------------


@pytest.fixture
def deployment_episodes_with_images(tmp_path):
    """Create success + failure episodes with actual image files."""
    episodes = []
    for outcome, prefix in [(Outcome.SUCCESS, "suc"), (Outcome.FAILURE, "fail")]:
        frames = []
        for i in range(10):
            img = Image.fromarray(
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            )
            img_path = tmp_path / "images" / f"{prefix}_{i:04d}.png"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(img_path)
            frames.append(
                EpisodeFrame(
                    timestamp=1000.0 + i * 0.033,
                    joint_positions=(np.random.randn(6) * 0.1).tolist(),
                    gripper_state=float(np.clip(np.random.random(), 0, 1)),
                    image_path=str(img_path),
                    action=(np.random.randn(6) * 0.01).tolist(),
                    reward=0.1,
                )
            )
        episodes.append(
            Episode(task_name="test", frames=frames, outcome=outcome)
        )
    return episodes
