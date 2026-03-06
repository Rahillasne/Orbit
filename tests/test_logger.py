"""Tests for orbit.logger module."""

from __future__ import annotations

import threading
import time

import h5py
import numpy as np
import pytest
from PIL import Image

from orbit.logger.episode_logger import EpisodeLogger
from orbit.logger.schemas import (
    DeploymentSession,
    Episode,
    EpisodeFrame,
    LoggerConfig,
    Outcome,
)
from orbit.logger.storage import HDF5Storage, LeRobotExporter

# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestSchemas:
    """Test new Pydantic models."""

    def test_episode_frame_creation(self):
        frame = EpisodeFrame(
            timestamp=1.0,
            joint_positions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            gripper_state=0.5,
            action=[0.01, -0.02, 0.03, -0.01, 0.0, 0.0],
            reward=0.1,
        )
        assert len(frame.joint_positions) == 6
        assert frame.gripper_state == 0.5
        assert frame.reward == 0.1

    def test_episode_frame_gripper_bounds(self):
        with pytest.raises(Exception):
            EpisodeFrame(
                timestamp=1.0,
                joint_positions=[0.1],
                gripper_state=1.5,  # out of bounds
                action=[0.0],
            )
        with pytest.raises(Exception):
            EpisodeFrame(
                timestamp=1.0,
                joint_positions=[0.1],
                gripper_state=-0.1,  # out of bounds
                action=[0.0],
            )

    def test_episode_frame_optional_reward(self):
        frame = EpisodeFrame(
            timestamp=1.0,
            joint_positions=[0.1],
            gripper_state=0.0,
            action=[0.0],
        )
        assert frame.reward is None

    def test_episode_json_serializable(self):
        episode = Episode(
            task_name="test",
            robot_id="r1",
            frames=[
                EpisodeFrame(
                    timestamp=1.0,
                    joint_positions=[0.1, 0.2],
                    gripper_state=0.5,
                    action=[0.0, 0.0],
                    reward=1.0,
                )
            ],
            outcome=Outcome.SUCCESS,
        )
        data = episode.model_dump(mode="json")
        assert isinstance(data, dict)
        assert isinstance(data["episode_id"], str)
        assert data["outcome"] == "success"
        assert len(data["frames"]) == 1
        # Verify round-trip
        reloaded = Episode.model_validate(data)
        assert reloaded.episode_id == episode.episode_id
        assert reloaded.outcome == Outcome.SUCCESS

    def test_deployment_session_structure(self):
        session = DeploymentSession(
            environment_description="sim_table",
            policy_version="v2.1",
        )
        assert len(session.episodes) == 0
        data = session.model_dump(mode="json")
        assert isinstance(data["session_id"], str)

    def test_episode_properties(self):
        frames = [
            EpisodeFrame(
                timestamp=1.0,
                joint_positions=[0.1, 0.2],
                gripper_state=0.5,
                action=[1.0, 0.0],
                reward=0.5,
            ),
            EpisodeFrame(
                timestamp=2.0,
                joint_positions=[0.3, 0.4],
                gripper_state=0.8,
                action=[0.0, 1.0],
                reward=1.5,
            ),
        ]
        ep = Episode(task_name="test", frames=frames, outcome=Outcome.SUCCESS)
        assert ep.num_frames == 2
        assert ep.total_reward == pytest.approx(2.0)
        assert ep.avg_action_magnitude > 0


# ---------------------------------------------------------------------------
# EpisodeLogger
# ---------------------------------------------------------------------------


class TestEpisodeLogger:
    """Test new EpisodeLogger API."""

    def test_context_manager(self, new_config):
        with EpisodeLogger(new_config) as logger:
            logger.start_episode()
            logger.log_frame(
                joint_positions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                gripper_state=0.5,
                action=[0.01, -0.02, 0.03, -0.01, 0.0, 0.0],
                reward=0.5,
            )
            ep = logger.end_episode(outcome=Outcome.SUCCESS)
            assert ep.num_frames == 1
            assert ep.outcome == Outcome.SUCCESS

    def test_start_end_episode(self, new_config):
        with EpisodeLogger(new_config) as logger:
            ep_id = logger.start_episode(task_name="grasp")
            assert logger.current_episode_id == ep_id
            logger.log_frame(joint_positions=[0.0] * 6, gripper_state=0.0, action=[0.0] * 6)
            ep = logger.end_episode(outcome=Outcome.FAILURE)
            assert ep.task_name == "grasp"
            assert ep.outcome == Outcome.FAILURE
            assert logger.current_episode_id is None

    def test_log_frame_records_data(self, new_config):
        with EpisodeLogger(new_config) as logger:
            logger.start_episode()
            joints = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
            action = [0.01, -0.02, 0.03, -0.01, 0.0, 0.0]
            logger.log_frame(
                joint_positions=joints,
                gripper_state=0.75,
                action=action,
                reward=1.0,
            )
            ep = logger.end_episode()
            frame = ep.frames[0]
            assert frame.joint_positions == joints
            assert frame.action == action
            assert frame.gripper_state == 0.75
            assert frame.reward == 1.0

    def test_double_start_raises(self, new_config):
        with EpisodeLogger(new_config) as logger:
            logger.start_episode()
            with pytest.raises(RuntimeError, match="already in progress"):
                logger.start_episode()

    def test_end_without_start_raises(self, new_config):
        with EpisodeLogger(new_config) as logger:
            with pytest.raises(RuntimeError, match="No episode in progress"):
                logger.end_episode()

    def test_log_frame_without_start_raises(self, new_config):
        with EpisodeLogger(new_config) as logger:
            with pytest.raises(RuntimeError, match="No episode in progress"):
                logger.log_frame(joint_positions=[0.0] * 6, gripper_state=0.0, action=[0.0] * 6)

    def test_dof_validation(self, new_config):
        with EpisodeLogger(new_config) as logger:
            logger.start_episode()
            with pytest.raises(ValueError, match="Expected 6 joint positions, got 3"):
                logger.log_frame(
                    joint_positions=[0.0, 0.0, 0.0],
                    gripper_state=0.0,
                    action=[0.0] * 6,
                )

    def test_auto_episode_id_increment(self, new_config):
        with EpisodeLogger(new_config) as logger:
            id1 = logger.start_episode()
            logger.log_frame(joint_positions=[0.0] * 6, gripper_state=0.0, action=[0.0] * 6)
            logger.end_episode()

            id2 = logger.start_episode()
            logger.log_frame(joint_positions=[0.0] * 6, gripper_state=0.0, action=[0.0] * 6)
            logger.end_episode()

            assert id1 != id2  # UUIDs should be unique

    def test_summary_method(self, new_config):
        with EpisodeLogger(new_config) as logger:
            logger.start_episode()
            for _ in range(10):
                logger.log_frame(
                    joint_positions=(np.random.randn(6) * 0.1).tolist(),
                    gripper_state=0.5,
                    action=(np.random.randn(6) * 0.01).tolist(),
                    reward=0.1,
                )
            logger.end_episode(outcome=Outcome.SUCCESS)

            summary = logger.summary()
            assert len(summary) == 1
            stats = list(summary.values())[0]
            assert stats["num_frames"] == 10
            assert stats["outcome"] == "success"
            assert stats["total_reward"] == pytest.approx(1.0)
            assert stats["avg_action_magnitude"] > 0

    def test_context_manager_auto_ends_on_exception(self, new_config):
        """Episode auto-ends as FAILURE when exception occurs."""
        episodes = []
        try:
            with EpisodeLogger(new_config) as logger:
                logger.start_episode()
                logger.log_frame(joint_positions=[0.0] * 6, gripper_state=0.0, action=[0.0] * 6)
                episodes.append(logger._completed_episodes)
                raise ValueError("simulated error")
        except ValueError:
            pass
        # The episode should have been auto-ended
        assert len(episodes[0]) == 1
        assert episodes[0][0].outcome == Outcome.FAILURE


# ---------------------------------------------------------------------------
# Mock deployment (5 episodes: 3 success, 2 failure)
# ---------------------------------------------------------------------------


class TestMockDeployment:
    """Simulate a realistic deployment with 5 episodes."""

    def test_five_episode_deployment(self, tmp_path):
        config = LoggerConfig(
            storage_dir=str(tmp_path / "deploy_data"),
            task_name="pick_place",
            robot_dof=6,
        )

        with EpisodeLogger(config) as logger:
            for i in range(5):
                success = i < 3
                logger.start_episode(task_name="pick_place")
                for frame_idx in range(20):
                    joints = (np.random.randn(6) * 0.1).tolist()
                    action = (np.random.randn(6) * 0.01).tolist()
                    gripper = float(np.clip(np.random.random(), 0, 1))
                    # Solid-color test images (different color per episode)
                    color = (i * 50) % 256
                    img = Image.fromarray(np.full((64, 64, 3), fill_value=color, dtype=np.uint8))
                    logger.log_frame(
                        joint_positions=joints,
                        gripper_state=gripper,
                        action=action,
                        reward=0.1 if success else -0.5,
                        images={"front": img},
                    )
                outcome = Outcome.SUCCESS if success else Outcome.FAILURE
                logger.end_episode(outcome=outcome)

            # Verify counts
            assert logger.num_episodes == 5

            # Verify summary
            summary = logger.summary()
            assert len(summary) == 5
            outcomes = [s["outcome"] for s in summary.values()]
            assert outcomes.count("success") == 3
            assert outcomes.count("failure") == 2

            # Verify each episode has 20 frames
            for stats in summary.values():
                assert stats["num_frames"] == 20


# ---------------------------------------------------------------------------
# HDF5 storage
# ---------------------------------------------------------------------------


class TestHDF5Storage:
    """Test HDF5 save/load/list cycle."""

    def _make_config(self, tmp_path):
        return LoggerConfig(
            storage_dir=str(tmp_path / "hdf5_data"),
            robot_dof=6,
        )

    def test_save_verify_structure(self, tmp_path):
        """Save an episode and verify the HDF5 file structure."""
        config = self._make_config(tmp_path)
        storage = HDF5Storage(config)
        session = DeploymentSession(environment_description="test", policy_version="v1")
        storage.create_session(session)

        episode = Episode(task_name="pick", robot_id="arm1", outcome=Outcome.SUCCESS)
        storage.begin_episode(session.session_id, episode)

        for i in range(10):
            frame = EpisodeFrame(
                timestamp=time.time(),
                joint_positions=[float(i)] * 6,
                gripper_state=float(i) / 10,
                action=[float(i) * 0.01] * 6,
                reward=float(i) * 0.1,
            )
            storage.append_frame(session.session_id, episode.episode_id, frame)

        import datetime

        storage.end_episode(
            session.session_id,
            episode.episode_id,
            Outcome.SUCCESS,
            datetime.datetime.now(),
        )

        # Verify HDF5 structure
        h5_path = storage.storage_dir / f"session_{session.session_id}.h5"
        assert h5_path.exists()

        with h5py.File(h5_path, "r") as f:
            assert "episodes" in f
            ep_grp = f[f"episodes/{episode.episode_id}"]
            assert ep_grp["timestamps"].shape == (10,)
            assert ep_grp["joint_positions"].shape == (10, 6)
            assert ep_grp["gripper_state"].shape == (10,)
            assert ep_grp["actions"].shape == (10, 6)
            assert ep_grp["rewards"].shape == (10,)
            assert ep_grp.attrs["task_name"] == "pick"
            assert ep_grp.attrs["outcome"] == "success"

        storage.close()

    def test_load_verify_data_match(self, tmp_path):
        """Save and load an episode, verify all data matches."""
        config = self._make_config(tmp_path)
        storage = HDF5Storage(config)
        session = DeploymentSession()
        storage.create_session(session)

        frames = []
        for i in range(5):
            frames.append(
                EpisodeFrame(
                    timestamp=1000.0 + i,
                    joint_positions=[float(i)] * 6,
                    gripper_state=float(i) / 5,
                    action=[float(i) * 0.1] * 6,
                    reward=float(i),
                )
            )
        episode = Episode(
            task_name="place",
            robot_id="arm2",
            frames=frames,
            outcome=Outcome.FAILURE,
            policy_checkpoint="ckpt_100",
        )
        import datetime

        episode.end_time = datetime.datetime.now()
        storage.save_episode(session.session_id, episode)

        loaded = storage.load_episode(session.session_id, episode.episode_id)
        assert loaded.task_name == "place"
        assert loaded.robot_id == "arm2"
        assert loaded.outcome == Outcome.FAILURE
        assert loaded.policy_checkpoint == "ckpt_100"
        assert len(loaded.frames) == 5

        for orig, load in zip(episode.frames, loaded.frames):
            assert orig.joint_positions == pytest.approx(load.joint_positions)
            assert orig.gripper_state == pytest.approx(load.gripper_state)
            assert orig.action == pytest.approx(load.action)
            assert orig.reward == pytest.approx(load.reward)

        storage.close()

    def test_incremental_frame_append(self, tmp_path):
        """Frames appended incrementally should match a batch write."""
        config = self._make_config(tmp_path)
        storage = HDF5Storage(config)
        session = DeploymentSession()
        storage.create_session(session)

        episode = Episode(task_name="test")
        storage.begin_episode(session.session_id, episode)

        for i in range(15):
            frame = EpisodeFrame(
                timestamp=float(i),
                joint_positions=[float(i)] * 6,
                gripper_state=0.5,
                action=[0.01] * 6,
                reward=0.1,
            )
            storage.append_frame(session.session_id, episode.episode_id, frame)

        import datetime

        storage.end_episode(
            session.session_id, episode.episode_id, Outcome.SUCCESS, datetime.datetime.now()
        )

        loaded = storage.load_episode(session.session_id, episode.episode_id)
        assert loaded.num_frames == 15
        assert loaded.frames[0].joint_positions == pytest.approx([0.0] * 6)
        assert loaded.frames[14].joint_positions == pytest.approx([14.0] * 6)

        storage.close()

    def test_list_episodes_with_filters(self, tmp_path):
        """list_episodes should support task and outcome filters."""
        config = self._make_config(tmp_path)
        storage = HDF5Storage(config)
        session = DeploymentSession()
        storage.create_session(session)

        for task, outcome in [("pick", Outcome.SUCCESS), ("place", Outcome.FAILURE)]:
            ep = Episode(task_name=task, outcome=outcome)
            ep.frames.append(
                EpisodeFrame(
                    timestamp=1.0,
                    joint_positions=[0.0] * 6,
                    gripper_state=0.0,
                    action=[0.0] * 6,
                )
            )
            import datetime

            ep.end_time = datetime.datetime.now()
            storage.save_episode(session.session_id, ep)

        all_eps = storage.list_episodes(session_id=session.session_id)
        assert len(all_eps) == 2

        pick_eps = storage.list_episodes(session_id=session.session_id, task="pick")
        assert len(pick_eps) == 1

        fail_eps = storage.list_episodes(session_id=session.session_id, outcome=Outcome.FAILURE)
        assert len(fail_eps) == 1

        storage.close()

    def test_concurrent_access_with_lock(self, tmp_path):
        """Two threads writing to the same session should not corrupt data."""
        config = self._make_config(tmp_path)
        storage = HDF5Storage(config)
        session = DeploymentSession()
        storage.create_session(session)

        errors: list[Exception] = []

        def write_episode(ep_idx: int) -> None:
            try:
                ep = Episode(task_name=f"task_{ep_idx}")
                storage.begin_episode(session.session_id, ep)
                for i in range(10):
                    frame = EpisodeFrame(
                        timestamp=float(i),
                        joint_positions=[float(ep_idx)] * 6,
                        gripper_state=0.5,
                        action=[0.01] * 6,
                        reward=0.1,
                    )
                    storage.append_frame(session.session_id, ep.episode_id, frame)
                import datetime

                storage.end_episode(
                    session.session_id, ep.episode_id, Outcome.SUCCESS, datetime.datetime.now()
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_episode, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        all_eps = storage.list_episodes(session_id=session.session_id)
        assert len(all_eps) == 3

        storage.close()


# ---------------------------------------------------------------------------
# LeRobot export
# ---------------------------------------------------------------------------


class TestLeRobotExport:
    """Test LeRobot format export."""

    def test_export_produces_parquet(self, tmp_path):
        frames = [
            EpisodeFrame(
                timestamp=float(i),
                joint_positions=[float(i)] * 6,
                gripper_state=0.5,
                action=[0.01] * 6,
                reward=0.1,
            )
            for i in range(10)
        ]
        episode = Episode(task_name="test", frames=frames, outcome=Outcome.SUCCESS)

        exporter = LeRobotExporter()
        output_dir = tmp_path / "lerobot_out"
        parquet_path = exporter.export_episode(episode, output_dir)

        assert parquet_path.exists()
        assert parquet_path.suffix == ".parquet"

        import pandas as pd

        df = pd.read_parquet(parquet_path)
        assert len(df) == 10
        assert "joint_position_0" in df.columns
        assert "action_0" in df.columns
        assert "gripper_state" in df.columns

    def test_export_session(self, tmp_path):
        episodes = []
        for _ in range(2):
            frames = [
                EpisodeFrame(
                    timestamp=1.0,
                    joint_positions=[0.0] * 6,
                    gripper_state=0.0,
                    action=[0.0] * 6,
                )
            ]
            episodes.append(Episode(task_name="test", frames=frames))

        session = DeploymentSession(episodes=episodes, policy_version="v1")
        exporter = LeRobotExporter()
        output = exporter.export_session(session, tmp_path / "session_out")

        parquet_files = list((output / "data").glob("*.parquet"))
        assert len(parquet_files) == 2


# ---------------------------------------------------------------------------
# Background image thread
# ---------------------------------------------------------------------------


class TestBackgroundImageThread:
    """Test background image saving."""

    def test_images_saved_after_wait(self, tmp_path):
        config = LoggerConfig(
            storage_dir=str(tmp_path / "img_data"),
            robot_dof=6,
            save_images=True,
        )
        with EpisodeLogger(config) as logger:
            logger.start_episode()
            for _ in range(5):
                img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
                logger.log_frame(
                    joint_positions=[0.0] * 6,
                    gripper_state=0.0,
                    action=[0.0] * 6,
                    images={"front": img},
                )
            logger.end_episode(outcome=Outcome.SUCCESS)
            logger.wait_for_images()

        # Check images on disk
        img_dir = tmp_path / "img_data" / "images"
        assert img_dir.exists()
        png_files = list(img_dir.rglob("*.png"))
        assert len(png_files) == 5

    def test_concurrent_frame_logging(self, tmp_path):
        """Simulate real-time logging by calling log_frame rapidly."""
        config = LoggerConfig(
            storage_dir=str(tmp_path / "concurrent_data"),
            robot_dof=6,
            save_images=True,
        )
        with EpisodeLogger(config) as logger:
            logger.start_episode()
            for i in range(50):
                img = Image.fromarray(np.full((16, 16, 3), fill_value=i % 256, dtype=np.uint8))
                logger.log_frame(
                    joint_positions=(np.random.randn(6) * 0.1).tolist(),
                    gripper_state=float(i % 2),
                    action=(np.random.randn(6) * 0.01).tolist(),
                    reward=0.01,
                    images={"front": img},
                )
            logger.end_episode(outcome=Outcome.SUCCESS)
            logger.wait_for_images()

        png_files = list((tmp_path / "concurrent_data" / "images").rglob("*.png"))
        assert len(png_files) == 50

    def test_thread_completion_on_context_exit(self, tmp_path):
        """Image thread should complete when context manager exits."""
        config = LoggerConfig(
            storage_dir=str(tmp_path / "exit_data"),
            robot_dof=6,
            save_images=True,
        )
        logger = EpisodeLogger(config)
        logger.start_episode()
        for _ in range(3):
            img = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))
            logger.log_frame(
                joint_positions=[0.0] * 6,
                gripper_state=0.0,
                action=[0.0] * 6,
                images={"front": img},
            )
        logger.end_episode()
        logger.wait_for_images()
        logger._shutdown_image_thread()
        logger._storage.close()

        png_files = list((tmp_path / "exit_data" / "images").rglob("*.png"))
        assert len(png_files) == 3
