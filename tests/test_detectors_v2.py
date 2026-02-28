"""Tests for Phase-2 heuristic failure detectors."""

from __future__ import annotations

import datetime

import numpy as np
import pytest

from orbit.detector.heuristic import (
    BaseDetector,
    DetectorPipeline,
    FailureDetection,
    GripperDropConfig,
    GripperDropDetector,
    OutOfBoundsConfig,
    OutOfBoundsDetector,
    PipelineResult,
    RewardThresholdConfig,
    RewardThresholdDetector,
    StallConfig,
    StallDetector,
    TimeoutConfig,
    TimeoutDetector,
    load_pipeline_from_yaml,
)
from orbit.logger.schemas import Episode, EpisodeFrame, LoggerConfig, Outcome


# ── Helpers ────────────────────────────────────────────────────────


def _make_frames(
    n: int,
    joint_fn=None,
    gripper_fn=None,
    action_fn=None,
    reward_fn=None,
    dt: float = 0.033,
) -> list[EpisodeFrame]:
    """Build synthetic frame sequences."""
    frames = []
    for i in range(n):
        frames.append(
            EpisodeFrame(
                timestamp=1000.0 + i * dt,
                joint_positions=joint_fn(i) if joint_fn else [0.1 * (i + 1)] * 6,
                gripper_state=gripper_fn(i) if gripper_fn else 0.5,
                action=action_fn(i) if action_fn else [0.01] * 6,
                reward=reward_fn(i) if reward_fn else 0.1,
            )
        )
    return frames


def _make_episode(frames: list[EpisodeFrame], **kwargs) -> Episode:
    start = datetime.datetime(2025, 1, 1, 0, 0, 0)
    end = start + datetime.timedelta(seconds=len(frames) * 0.033)
    return Episode(
        task_name=kwargs.pop("task_name", "test_task"),
        robot_id=kwargs.pop("robot_id", "test_robot"),
        start_time=start,
        end_time=end,
        frames=frames,
        **kwargs,
    )


# ── GripperDropDetector ───────────────────────────────────────────


class TestGripperDropDetector:
    def test_drop_detected(self):
        """Gripper grips (>0.8) for 12 frames then opens (<0.2) -> detection."""

        def gripper_fn(i):
            # 12 frames gripping, then release early in episode (not in last 15%)
            return 0.9 if i < 12 else 0.1

        # 40 frames total so frame 12 is at 30%, well before the last 15%
        frames = _make_frames(40, gripper_fn=gripper_fn)
        episode = _make_episode(frames)
        detector = GripperDropDetector(GripperDropConfig(min_closed_frames=10))
        detections = detector.detect(episode)
        assert len(detections) == 1
        assert detections[0].frame_idx == 12
        assert detections[0].detector_name == "GripperDropDetector"
        assert detections[0].confidence == 0.85

    def test_no_drop_when_gripper_stays_closed(self):
        """Gripper stays gripping throughout -> no detection."""
        frames = _make_frames(20, gripper_fn=lambda _: 0.9)
        episode = _make_episode(frames)
        detector = GripperDropDetector()
        assert detector.detect(episode) == []

    def test_no_drop_when_never_closed(self):
        """Gripper never grips -> no detection."""
        frames = _make_frames(20, gripper_fn=lambda _: 0.1)
        episode = _make_episode(frames)
        detector = GripperDropDetector()
        assert detector.detect(episode) == []

    def test_short_close_below_threshold(self):
        """Gripper grips for fewer than min_closed_frames -> no detection."""

        def gripper_fn(i):
            if 5 <= i < 8:
                return 0.9  # only 3 frames gripping
            return 0.1

        frames = _make_frames(40, gripper_fn=gripper_fn)
        episode = _make_episode(frames)
        detector = GripperDropDetector(GripperDropConfig(min_closed_frames=5))
        assert detector.detect(episode) == []

    def test_multiple_drops(self):
        """Two separate drop events in one episode."""

        def gripper_fn(i):
            # First grip-and-drop
            if i < 12:
                return 0.9  # gripping
            if i < 20:
                return 0.1  # dropped (stays open past reclose window)
            # Second grip-and-drop
            if i < 32:
                return 0.9  # gripping again
            if i < 40:
                return 0.1  # dropped again
            return 0.5

        # 80 frames so drops are well before last 15% (release zone starts at 68)
        frames = _make_frames(80, gripper_fn=gripper_fn)
        episode = _make_episode(frames)
        detector = GripperDropDetector(GripperDropConfig(min_closed_frames=10))
        detections = detector.detect(episode)
        assert len(detections) == 2

    def test_release_zone_not_flagged(self):
        """Gripper opens in last 15% of episode -> intentional release, no detection."""

        def gripper_fn(i):
            return 0.9 if i < 18 else 0.1

        frames = _make_frames(20, gripper_fn=gripper_fn)
        episode = _make_episode(frames)
        # Frame 18 is at 90% of episode (last 15% starts at frame 17)
        detector = GripperDropDetector(GripperDropConfig(min_closed_frames=10))
        detections = detector.detect(episode)
        assert len(detections) == 0

    def test_reclose_not_flagged(self):
        """Gripper briefly opens then re-closes within 5 frames -> sensor noise."""

        def gripper_fn(i):
            if i < 12:
                return 0.9  # gripping
            if i == 12:
                return 0.1  # momentary open
            return 0.9  # re-closes immediately

        frames = _make_frames(40, gripper_fn=gripper_fn)
        episode = _make_episode(frames)
        detector = GripperDropDetector(GripperDropConfig(min_closed_frames=10, reclose_window=5))
        detections = detector.detect(episode)
        assert len(detections) == 0

    def test_empty_episode(self):
        episode = _make_episode([])
        detector = GripperDropDetector()
        assert detector.detect(episode) == []

    def test_explain(self):
        def gripper_fn(i):
            return 0.9 if i < 12 else 0.1

        frames = _make_frames(40, gripper_fn=gripper_fn)
        episode = _make_episode(frames)
        detector = GripperDropDetector(GripperDropConfig(min_closed_frames=10))
        detections = detector.detect(episode)
        explanation = detector.explain(detections)
        assert "GripperDropDetector" in explanation
        assert "issue" in explanation.lower()


# ── StallDetector ─────────────────────────────────────────────────


class TestStallDetector:
    def test_stall_detected(self):
        """All joints at same position for 20 frames -> stall."""
        frames = _make_frames(20, joint_fn=lambda _: [0.5] * 6)
        episode = _make_episode(frames)
        detector = StallDetector(StallConfig(min_stall_frames=5))
        detections = detector.detect(episode)
        assert len(detections) >= 1
        assert detections[0].detector_name == "StallDetector"

    def test_no_stall_when_moving(self):
        """Joints change every frame -> no stall."""
        frames = _make_frames(20, joint_fn=lambda i: [0.1 * i] * 6)
        episode = _make_episode(frames)
        detector = StallDetector()
        assert detector.detect(episode) == []

    def test_stall_mid_episode(self):
        """Movement, then stall, then movement again."""

        def joint_fn(i):
            if i < 5:
                return [0.1 * i] * 6
            if i < 20:
                return [0.5] * 6  # stalled
            return [0.1 * i] * 6

        frames = _make_frames(25, joint_fn=joint_fn)
        episode = _make_episode(frames)
        detector = StallDetector(StallConfig(min_stall_frames=5))
        detections = detector.detect(episode)
        assert len(detections) >= 1

    def test_single_frame_episode(self):
        """Episode with one frame -> no stall (need >= 2 frames)."""
        frames = _make_frames(1)
        episode = _make_episode(frames)
        detector = StallDetector()
        assert detector.detect(episode) == []


# ── OutOfBoundsDetector ───────────────────────────────────────────


class TestOutOfBoundsDetector:
    def test_out_of_bounds_upper(self):
        """Joint exceeds upper limit -> detection."""
        frames = _make_frames(10, joint_fn=lambda i: [float(i)] * 6)
        episode = _make_episode(frames)
        config = OutOfBoundsConfig(
            joint_limits_lower=[-5.0] * 6,
            joint_limits_upper=[5.0] * 6,
        )
        detector = OutOfBoundsDetector(config)
        detections = detector.detect(episode)
        assert len(detections) == 1
        assert detections[0].detector_name == "OutOfBoundsDetector"

    def test_out_of_bounds_lower(self):
        """Joint below lower limit -> detection."""
        frames = _make_frames(10, joint_fn=lambda i: [-10.0 + i] * 6)
        episode = _make_episode(frames)
        config = OutOfBoundsConfig(
            joint_limits_lower=[-5.0] * 6,
            joint_limits_upper=[5.0] * 6,
        )
        detector = OutOfBoundsDetector(config)
        detections = detector.detect(episode)
        assert len(detections) == 1

    def test_in_bounds_no_detection(self):
        """All joints within limits -> no detection."""
        frames = _make_frames(10, joint_fn=lambda i: [0.1 * i] * 6)
        episode = _make_episode(frames)
        config = OutOfBoundsConfig(
            joint_limits_lower=[-5.0] * 6,
            joint_limits_upper=[5.0] * 6,
        )
        detector = OutOfBoundsDetector(config)
        assert detector.detect(episode) == []

    def test_no_limits_configured_skips(self):
        """When limits are empty, detector returns no detections."""
        frames = _make_frames(10)
        episode = _make_episode(frames)
        detector = OutOfBoundsDetector()
        assert detector.detect(episode) == []


# ── TimeoutDetector ───────────────────────────────────────────────


class TestTimeoutDetector:
    def test_timeout_by_duration(self):
        """Episode duration exceeds max -> detection."""
        frames = _make_frames(10)
        episode = _make_episode(frames)
        episode.end_time = episode.start_time + datetime.timedelta(seconds=100)
        detector = TimeoutDetector(TimeoutConfig(max_duration_seconds=60.0))
        detections = detector.detect(episode)
        assert len(detections) >= 1
        assert any("duration" in d.description.lower() for d in detections)

    def test_timeout_by_frame_count(self):
        """Frame count exceeds max -> detection."""
        frames = _make_frames(200)
        episode = _make_episode(frames)
        detector = TimeoutDetector(TimeoutConfig(max_frames=100))
        detections = detector.detect(episode)
        assert any("frame" in d.description.lower() for d in detections)

    def test_no_timeout(self):
        """Short episode under limits -> no detection."""
        frames = _make_frames(10)
        episode = _make_episode(frames)
        detector = TimeoutDetector()
        assert detector.detect(episode) == []

    def test_exact_limit_no_timeout(self):
        """Episode right at the time limit -> no detection."""
        frames = _make_frames(10)
        episode = _make_episode(frames)
        episode.end_time = episode.start_time + datetime.timedelta(seconds=60)
        detector = TimeoutDetector(TimeoutConfig(max_duration_seconds=60.0, max_frames=1000))
        assert detector.detect(episode) == []


# ── RewardThresholdDetector ───────────────────────────────────────


class TestRewardThresholdDetector:
    def test_low_total_reward(self):
        frames = _make_frames(10, reward_fn=lambda _: -1.0)
        episode = _make_episode(frames)
        detector = RewardThresholdDetector(
            RewardThresholdConfig(min_total_reward=0.0)
        )
        detections = detector.detect(episode)
        assert len(detections) >= 1
        assert detections[0].detector_name == "RewardThresholdDetector"

    def test_adequate_reward_no_detection(self):
        frames = _make_frames(10, reward_fn=lambda _: 1.0)
        episode = _make_episode(frames)
        detector = RewardThresholdDetector(
            RewardThresholdConfig(min_total_reward=0.0)
        )
        assert detector.detect(episode) == []

    def test_avg_reward_check(self):
        frames = _make_frames(10, reward_fn=lambda _: -0.5)
        episode = _make_episode(frames)
        detector = RewardThresholdDetector(
            RewardThresholdConfig(
                min_total_reward=-100.0,  # won't trigger
                min_avg_reward_per_frame=0.0,
            )
        )
        detections = detector.detect(episode)
        assert len(detections) == 1
        assert "average" in detections[0].description.lower()

    def test_none_rewards_treated_as_zero(self):
        """Frames with reward=None are excluded from total."""
        frames = _make_frames(10, reward_fn=lambda _: None)
        episode = _make_episode(frames)
        detector = RewardThresholdDetector(
            RewardThresholdConfig(min_total_reward=-1.0)
        )
        # total_reward is 0.0 (None excluded), which is >= -1.0
        assert detector.detect(episode) == []


# ── Normal episode (should NOT trigger detectors) ─────────────────


class TestNormalEpisode:
    def test_successful_pick_and_place(self):
        """A normal successful episode should not trigger any detector."""
        rng = np.random.default_rng(42)

        def joint_fn(i):
            # Smooth trajectory with movement
            return (rng.standard_normal(6) * 0.1 + np.array([0.5] * 6) + i * 0.01).tolist()

        frames = _make_frames(
            30,
            joint_fn=joint_fn,
            gripper_fn=lambda i: 0.1 if i < 25 else 0.1,  # stays closed
            reward_fn=lambda i: 0.1 if i < 25 else 1.0,
        )
        episode = _make_episode(frames)
        pipeline = DetectorPipeline(
            detectors=[
                GripperDropDetector(),
                StallDetector(),
                RewardThresholdDetector(RewardThresholdConfig(min_total_reward=0.0)),
                TimeoutDetector(),
            ]
        )
        result = pipeline.run(episode)
        assert not result.is_failure

    def test_at_time_limit(self):
        """Task completing right at the time limit -> no failure."""
        frames = _make_frames(50, reward_fn=lambda _: 0.5)
        episode = _make_episode(frames)
        episode.end_time = episode.start_time + datetime.timedelta(seconds=60)
        pipeline = DetectorPipeline(
            detectors=[
                TimeoutDetector(TimeoutConfig(max_duration_seconds=60.0, max_frames=1000)),
                RewardThresholdDetector(RewardThresholdConfig(min_total_reward=0.0)),
            ]
        )
        result = pipeline.run(episode)
        assert not result.is_failure


# ── DetectorPipeline ──────────────────────────────────────────────


class TestDetectorPipeline:
    def test_pipeline_aggregates_detections(self):
        """Pipeline collects detections from all detectors."""
        frames = _make_frames(
            30,
            joint_fn=lambda _: [0.5] * 6,
            gripper_fn=lambda _: 0.5,
            reward_fn=lambda _: -1.0,
        )
        episode = _make_episode(frames)
        pipeline = DetectorPipeline(
            detectors=[
                StallDetector(StallConfig(min_stall_frames=5)),
                RewardThresholdDetector(RewardThresholdConfig(min_total_reward=0.0)),
            ]
        )
        result = pipeline.run(episode)
        assert result.is_failure
        assert len(result.detections) >= 2
        assert result.failure_probability > 0.0
        assert 0.0 < result.failure_probability <= 1.0

    def test_empty_pipeline_no_failures(self):
        frames = _make_frames(10)
        episode = _make_episode(frames)
        pipeline = DetectorPipeline(detectors=[])
        result = pipeline.run(episode)
        assert not result.is_failure
        assert result.failure_probability == 0.0

    def test_run_batch(self):
        ep1 = _make_episode(_make_frames(10, reward_fn=lambda _: 1.0))
        ep2 = _make_episode(_make_frames(10, reward_fn=lambda _: -1.0))
        pipeline = DetectorPipeline(
            detectors=[
                RewardThresholdDetector(RewardThresholdConfig(min_total_reward=0.0)),
            ]
        )
        results = pipeline.run_batch([ep1, ep2])
        assert len(results) == 2
        assert not results[0].is_failure
        assert results[1].is_failure

    def test_default_pipeline_has_five_detectors(self):
        """DetectorPipeline() with no args includes all 5 detectors."""
        pipeline = DetectorPipeline()
        assert len(pipeline.detectors) == 5

    def test_add_detector(self):
        pipeline = DetectorPipeline(detectors=[])
        assert len(pipeline.detectors) == 0
        pipeline.add_detector(RewardThresholdDetector())
        assert len(pipeline.detectors) == 1

    def test_failure_probability_is_max_confidence(self):
        """failure_probability should be the max of individual confidences."""
        frames = _make_frames(
            30,
            joint_fn=lambda _: [0.5] * 6,
            reward_fn=lambda _: -1.0,
        )
        episode = _make_episode(frames)
        pipeline = DetectorPipeline(
            detectors=[
                StallDetector(StallConfig(min_stall_frames=5, confidence=0.80)),
                RewardThresholdDetector(
                    RewardThresholdConfig(min_total_reward=0.0, confidence=0.75)
                ),
            ]
        )
        result = pipeline.run(episode)
        assert result.failure_probability == 0.80


# ── BaseDetector explain ──────────────────────────────────────────


class TestBaseDetector:
    def test_explain_no_detections(self):
        detector = RewardThresholdDetector()
        explanation = detector.explain([])
        assert "No failures" in explanation

    def test_explain_with_detections(self):
        frames = _make_frames(10, reward_fn=lambda _: -1.0)
        episode = _make_episode(frames)
        detector = RewardThresholdDetector(
            RewardThresholdConfig(min_total_reward=0.0)
        )
        detections = detector.detect(episode)
        explanation = detector.explain(detections)
        assert "RewardThresholdDetector" in explanation
        assert "1 issue" in explanation


# ── YAML config loading ───────────────────────────────────────────


class TestYAMLConfig:
    def test_load_pipeline_from_yaml(self, tmp_path):
        config_yaml = tmp_path / "detectors.yaml"
        config_yaml.write_text(
            "detectors:\n"
            "  - name: gripper_drop\n"
            "    closed_threshold: 0.2\n"
            "    open_threshold: 0.8\n"
            "    min_closed_frames: 3\n"
            "  - name: stall\n"
            "    velocity_threshold: 0.005\n"
            "    min_stall_frames: 15\n"
            "  - name: reward_threshold\n"
            "    min_total_reward: -5.0\n"
        )
        pipeline = load_pipeline_from_yaml(config_yaml)
        assert len(pipeline.detectors) == 3
        assert isinstance(pipeline.detectors[0], GripperDropDetector)
        assert isinstance(pipeline.detectors[1], StallDetector)
        assert isinstance(pipeline.detectors[2], RewardThresholdDetector)
        # Verify config values propagated
        assert pipeline.detectors[0].config.closed_threshold == 0.2
        assert pipeline.detectors[1].config.min_stall_frames == 15
        assert pipeline.detectors[2].config.min_total_reward == -5.0

    def test_all_detector_types_in_yaml(self, tmp_path):
        config_yaml = tmp_path / "all.yaml"
        config_yaml.write_text(
            "detectors:\n"
            "  - name: gripper_drop\n"
            "  - name: stall\n"
            "  - name: out_of_bounds\n"
            "    joint_limits_lower: [-3.14, -1.57, -3.14, -3.14, -1.57, -3.14]\n"
            "    joint_limits_upper: [3.14, 1.57, 3.14, 3.14, 1.57, 3.14]\n"
            "  - name: timeout\n"
            "    max_duration_seconds: 120.0\n"
            "  - name: reward_threshold\n"
        )
        pipeline = load_pipeline_from_yaml(config_yaml)
        assert len(pipeline.detectors) == 5

    def test_unknown_detector_raises(self, tmp_path):
        config_yaml = tmp_path / "bad.yaml"
        config_yaml.write_text(
            "detectors:\n"
            "  - name: nonexistent\n"
        )
        with pytest.raises(ValueError, match="Unknown detector"):
            load_pipeline_from_yaml(config_yaml)

    def test_missing_detectors_key_raises(self, tmp_path):
        config_yaml = tmp_path / "bad2.yaml"
        config_yaml.write_text("some_other_key: true\n")
        with pytest.raises(ValueError, match="detectors"):
            load_pipeline_from_yaml(config_yaml)


# ── CLI smoke test ────────────────────────────────────────────────


class TestCLI:
    def test_cli_help(self):
        from click.testing import CliRunner

        from orbit.detector.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "session" in result.output.lower()

    def test_cli_with_session_file(self, tmp_path):
        """Create a minimal session file and run the CLI against it."""
        from orbit.logger.schemas import DeploymentSession
        from orbit.logger.storage import HDF5Storage

        config = LoggerConfig(storage_dir=str(tmp_path))
        storage = HDF5Storage(config)
        session = DeploymentSession()
        storage.create_session(session)

        frames = [
            EpisodeFrame(
                timestamp=float(i),
                joint_positions=[0.0] * 6,
                gripper_state=0.5,
                action=[0.0] * 6,
                reward=-1.0,
            )
            for i in range(10)
        ]
        ep = Episode(task_name="test", frames=frames, outcome=Outcome.UNKNOWN)
        ep.end_time = datetime.datetime.now()
        storage.save_episode(session.session_id, ep)
        storage.close()

        session_file = tmp_path / f"session_{session.session_id}.h5"

        from click.testing import CliRunner

        from orbit.detector.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--session", str(session_file)])
        assert result.exit_code == 0
        assert "1 episodes analyzed" in result.output

    def test_cli_json_output(self, tmp_path):
        """Test JSON output format."""
        import json as json_mod

        from orbit.logger.schemas import DeploymentSession
        from orbit.logger.storage import HDF5Storage

        config = LoggerConfig(storage_dir=str(tmp_path))
        storage = HDF5Storage(config)
        session = DeploymentSession()
        storage.create_session(session)

        frames = [
            EpisodeFrame(
                timestamp=float(i),
                joint_positions=[0.0] * 6,
                gripper_state=0.5,
                action=[0.0] * 6,
                reward=1.0,
            )
            for i in range(10)
        ]
        ep = Episode(task_name="test", frames=frames)
        ep.end_time = datetime.datetime.now()
        storage.save_episode(session.session_id, ep)
        storage.close()

        session_file = tmp_path / f"session_{session.session_id}.h5"

        from click.testing import CliRunner

        from orbit.detector.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--session", str(session_file), "--json-output"])
        assert result.exit_code == 0
        parsed = json_mod.loads(result.output)
        assert isinstance(parsed, list)
        assert len(parsed) == 1


# ── Post-episode hook integration ─────────────────────────────────


class TestPostEpisodeHook:
    def test_hook_fires_on_end_episode(self, tmp_path):
        """Hook is called after end_episode."""
        from orbit.logger.episode_logger import EpisodeLogger

        config = LoggerConfig(storage_dir=str(tmp_path), robot_dof=6)
        called = []

        def my_hook(episode):
            called.append(episode.episode_id)

        with EpisodeLogger(config) as logger:
            logger.add_post_episode_hook(my_hook)
            logger.start_episode()
            logger.log_frame(
                joint_positions=[0.0] * 6,
                gripper_state=0.5,
                action=[0.0] * 6,
            )
            ep = logger.end_episode(outcome=Outcome.SUCCESS)

        assert len(called) == 1
        assert called[0] == ep.episode_id

    def test_detector_hook_updates_outcome(self, tmp_path):
        """create_detector_hook marks UNKNOWN episodes as FAILURE."""
        from orbit.logger.episode_logger import EpisodeLogger, create_detector_hook

        config = LoggerConfig(storage_dir=str(tmp_path), robot_dof=6)
        pipeline = DetectorPipeline(
            detectors=[
                RewardThresholdDetector(RewardThresholdConfig(min_total_reward=0.0)),
            ]
        )
        hook = create_detector_hook(pipeline, update_outcome=True)

        with EpisodeLogger(config) as logger:
            logger.add_post_episode_hook(hook)
            logger.start_episode()
            for _ in range(5):
                logger.log_frame(
                    joint_positions=[0.0] * 6,
                    gripper_state=0.5,
                    action=[0.0] * 6,
                    reward=-1.0,
                )
            ep = logger.end_episode()  # outcome=UNKNOWN by default

        assert ep.outcome == Outcome.FAILURE
        assert "detector_results" in ep.metadata

    def test_hook_exception_does_not_propagate(self, tmp_path):
        """A failing hook should not crash end_episode."""
        from orbit.logger.episode_logger import EpisodeLogger

        config = LoggerConfig(storage_dir=str(tmp_path), robot_dof=6)

        def bad_hook(episode):
            raise RuntimeError("Hook exploded!")

        with EpisodeLogger(config) as logger:
            logger.add_post_episode_hook(bad_hook)
            logger.start_episode()
            logger.log_frame(
                joint_positions=[0.0] * 6,
                gripper_state=0.5,
                action=[0.0] * 6,
            )
            # Should not raise
            ep = logger.end_episode(outcome=Outcome.SUCCESS)
            assert ep.outcome == Outcome.SUCCESS

    def test_remove_hook(self, tmp_path):
        """Removed hooks should not fire."""
        from orbit.logger.episode_logger import EpisodeLogger

        config = LoggerConfig(storage_dir=str(tmp_path), robot_dof=6)
        called = []

        def my_hook(episode):
            called.append(True)

        with EpisodeLogger(config) as logger:
            logger.add_post_episode_hook(my_hook)
            logger.remove_post_episode_hook(my_hook)
            logger.start_episode()
            logger.log_frame(
                joint_positions=[0.0] * 6,
                gripper_state=0.5,
                action=[0.0] * 6,
            )
            logger.end_episode()

        assert called == []
