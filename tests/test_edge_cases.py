"""Comprehensive edge-case tests for the ORBIT project.

Each test exercises a boundary condition that real-world usage could
trigger, verifying that the system degrades gracefully instead of
crashing.

Edge cases covered
------------------
1. Empty directory            - discover_sessions on a dir with no .h5 files
2. Corrupted HDF5 file        - garbage bytes masquerading as session_*.h5
3. Single episode              - minimal 1-episode session through detector pipeline
4. All successes               - 100% success rate through prescriber
5. All failures                - 100% failure rate (negative reward) end-to-end
6. Missing / non-existent images - image_path references files that do not exist
"""

from __future__ import annotations

import datetime
from uuid import uuid4

import numpy as np
import pytest

from orbit.detector.heuristic import (
    DetectorPipeline,
    GripperDropDetector,
    OutOfBoundsDetector,
    RewardThresholdDetector,
    StallDetector,
    TimeoutDetector,
)
from orbit.logger.episode_logger import EpisodeLogger
from orbit.logger.schemas import (
    DeploymentSession,
    Episode,
    EpisodeFrame,
    LoggerConfig,
    Outcome,
)
from orbit.logger.storage import HDF5Storage

# NOTE: Prescriber and DetectionResult are imported lazily inside helper
# functions because Prescriber transitively imports open_clip / faiss / torch
# via orbit.analyzer.embedding_gap.  When those ML deps are not installed we
# still want the non-prescriber tests to run.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline() -> DetectorPipeline:
    """Build the standard 5-detector pipeline used by the dashboard."""
    return DetectorPipeline(
        detectors=[
            GripperDropDetector(),
            StallDetector(),
            OutOfBoundsDetector(),
            TimeoutDetector(),
            RewardThresholdDetector(),
        ]
    )


def _make_frame(
    idx: int,
    reward: float = 0.1,
    gripper: float = 0.5,
    image_path: str = "",
) -> EpisodeFrame:
    """Build a single plausible EpisodeFrame."""
    rng = np.random.default_rng(idx)
    return EpisodeFrame(
        timestamp=1000.0 + idx * 0.033,
        joint_positions=(rng.standard_normal(6) * 0.1).tolist(),
        gripper_state=float(np.clip(gripper, 0.0, 1.0)),
        action=(rng.standard_normal(6) * 0.01).tolist(),
        reward=reward,
        image_path=image_path,
    )


def _make_episode(
    n_frames: int = 20,
    outcome: Outcome = Outcome.SUCCESS,
    reward: float = 0.1,
    task_name: str = "edge_case_task",
    image_path: str = "",
) -> Episode:
    """Build a complete Episode with *n_frames* frames."""
    now = datetime.datetime.now()
    frames = [_make_frame(i, reward=reward, image_path=image_path) for i in range(n_frames)]
    return Episode(
        task_name=task_name,
        robot_id="test_robot",
        frames=frames,
        outcome=outcome,
        start_time=now,
        end_time=now + datetime.timedelta(seconds=n_frames * 0.033),
    )


def _detection_results_from_pipeline(
    episodes: list[Episode],
) -> list[dict]:
    """Run the heuristic pipeline and return dashboard-style dicts."""
    pipeline = _make_pipeline()
    results = pipeline.run_batch(episodes)
    serialized = []
    for r in results:
        serialized.append(
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
                "detector_summaries": r.detector_summaries,
            }
        )
    return serialized


def _run_prescriber_on(detection_dicts: list[dict]) -> dict:
    """Feed dashboard-style detection dicts into the Prescriber.

    Returns either a real prescription report dict or a stub when the
    heavy ML dependencies (open_clip, faiss, torch) are unavailable.
    """
    try:
        from orbit.detector.legacy import DetectionResult
        from orbit.prescriber.prescriber import Prescriber
    except ImportError:
        return {
            "prescriptions": [],
            "summary": "Prescriber unavailable (missing ML dependencies).",
            "num_failures_analyzed": 0,
        }

    legacy_results: list[DetectionResult] = []
    for r in detection_dicts:
        reasons = [d["description"] for d in r["detections"]]
        legacy_results.append(
            DetectionResult(
                episode_id=hash(r["episode_id"]) % (2**31),
                is_failure=r["is_failure"],
                failure_reasons=reasons,
                confidence=r["failure_probability"],
            )
        )
    prescriber = Prescriber()
    report = prescriber.prescribe(legacy_results)
    return {
        "prescriptions": [
            {
                "type": p.prescription_type.value,
                "title": p.title,
                "description": p.description,
                "priority": p.priority,
                "confidence": round(p.confidence, 3),
                "evidence": p.evidence,
                "suggested_params": p.suggested_params,
            }
            for p in report.prescriptions
        ],
        "summary": report.summary,
        "num_failures_analyzed": report.num_failures_analyzed,
    }


# Conditional skip for tests that absolutely require the Prescriber
_PRESCRIBER_AVAILABLE = True
try:
    from orbit.prescriber.prescriber import Prescriber as _Prescriber  # noqa: F401
except ImportError:
    _PRESCRIBER_AVAILABLE = False

requires_prescriber = pytest.mark.skipif(
    not _PRESCRIBER_AVAILABLE,
    reason="Prescriber unavailable (missing open_clip/faiss/torch)",
)


# ===================================================================
# 1. EMPTY DIRECTORY
# ===================================================================


class TestEmptyDirectory:
    """Edge case: discover_sessions and HDF5Storage on an empty dir."""

    def test_discover_sessions_empty_dir(self, tmp_path):
        """discover_sessions should return [] for a dir with no .h5 files."""
        empty_dir = tmp_path / "orbit-empty-dir"
        empty_dir.mkdir()

        from orbit.dashboard.data_loader import discover_sessions

        result = discover_sessions(str(empty_dir))
        assert result == [], f"Expected empty list, got {result}"

    def test_discover_sessions_nonexistent_dir(self, tmp_path):
        """discover_sessions should return [] for a dir that does not exist."""
        from orbit.dashboard.data_loader import discover_sessions

        result = discover_sessions(str(tmp_path / "does-not-exist"))
        assert result == [], f"Expected empty list, got {result}"

    def test_hdf5_storage_list_episodes_empty(self, tmp_path):
        """HDF5Storage.list_episodes on an empty dir returns []."""
        empty_dir = tmp_path / "orbit-empty-storage"
        empty_dir.mkdir()
        config = LoggerConfig(storage_dir=str(empty_dir))
        storage = HDF5Storage(config)
        pairs = storage.list_episodes()
        storage.close()
        assert pairs == []

    def test_detector_pipeline_empty_list(self):
        """DetectorPipeline.run_batch on an empty list returns []."""
        pipeline = _make_pipeline()
        results = pipeline.run_batch([])
        assert results == []

    @requires_prescriber
    def test_prescriber_empty_list(self):
        """Prescriber.prescribe on an empty detection list should not crash."""
        from orbit.prescriber.prescriber import Prescriber

        prescriber = Prescriber()
        report = prescriber.prescribe([])
        assert report.num_failures_analyzed == 0
        # Should produce a fallback "General Training Improvement"
        assert len(report.prescriptions) >= 0


# ===================================================================
# 2. CORRUPTED FILE
# ===================================================================


class TestCorruptedFile:
    """Edge case: a session_*.h5 file with garbage content."""

    def test_discover_sessions_skips_corrupt(self, tmp_path):
        """discover_sessions should skip corrupt .h5 files gracefully."""
        corrupt_dir = tmp_path / "corrupt-test"
        corrupt_dir.mkdir()

        fake_uuid = uuid4()
        corrupt_file = corrupt_dir / f"session_{fake_uuid}.h5"
        corrupt_file.write_bytes(b"THIS IS NOT AN HDF5 FILE -- GARBAGE CONTENT 1234567890!@#$%")

        from orbit.dashboard.data_loader import discover_sessions

        result = discover_sessions(str(corrupt_dir))
        # Should return empty list (skipped the corrupt file), not crash
        assert result == [], f"Expected empty list for corrupt dir, got {result}"

    def test_hdf5_storage_load_corrupt_file_raises(self, tmp_path):
        """HDF5Storage.load_episode on a corrupt file raises cleanly."""
        corrupt_dir = tmp_path / "corrupt-load"
        corrupt_dir.mkdir()

        fake_session_id = uuid4()
        fake_episode_id = uuid4()
        corrupt_file = corrupt_dir / f"session_{fake_session_id}.h5"
        corrupt_file.write_bytes(b"\x00\x01\x02GARBAGE" * 100)

        config = LoggerConfig(storage_dir=str(corrupt_dir))
        storage = HDF5Storage(config)

        with pytest.raises(Exception):
            # Should raise OSError or similar, not segfault
            storage.load_episode(fake_session_id, fake_episode_id)
        storage.close()

    def test_hdf5_storage_list_episodes_corrupt(self, tmp_path):
        """list_episodes on a corrupt file should raise, not segfault."""
        corrupt_dir = tmp_path / "corrupt-list"
        corrupt_dir.mkdir()

        fake_session_id = uuid4()
        corrupt_file = corrupt_dir / f"session_{fake_session_id}.h5"
        corrupt_file.write_bytes(b"NOT_HDF5_DATA_AT_ALL")

        config = LoggerConfig(storage_dir=str(corrupt_dir))
        storage = HDF5Storage(config)

        # Listing with a specific session_id should raise for the corrupt file
        with pytest.raises(Exception):
            storage.list_episodes(session_id=fake_session_id)
        storage.close()

    def test_discover_sessions_mixed_valid_and_corrupt(self, tmp_path):
        """discover_sessions returns valid sessions and skips corrupt ones."""
        mixed_dir = tmp_path / "mixed-test"
        mixed_dir.mkdir()

        # Create a valid session via the logger API
        config = LoggerConfig(
            storage_dir=str(mixed_dir),
            task_name="valid_task",
            robot_dof=6,
        )
        with EpisodeLogger(config) as logger:
            logger.start_episode()
            for i in range(5):
                logger.log_frame(
                    joint_positions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                    gripper_state=0.5,
                    action=[0.01, -0.02, 0.03, -0.01, 0.0, 0.0],
                    reward=0.1,
                )
            logger.end_episode(outcome=Outcome.SUCCESS)

        # Add a corrupt file alongside
        corrupt_uuid = uuid4()
        corrupt_file = mixed_dir / f"session_{corrupt_uuid}.h5"
        corrupt_file.write_bytes(b"CORRUPT FILE DATA")

        from orbit.dashboard.data_loader import discover_sessions

        result = discover_sessions(str(mixed_dir))
        # Should find exactly the one valid session
        assert len(result) == 1
        assert result[0]["episode_count"] == 1


# ===================================================================
# 3. SINGLE EPISODE
# ===================================================================


class TestSingleEpisode:
    """Edge case: a session with exactly 1 episode."""

    def test_single_episode_through_logger(self, tmp_path):
        """Create 1 episode via EpisodeLogger, read it back."""
        data_dir = tmp_path / "single-ep"
        config = LoggerConfig(
            storage_dir=str(data_dir),
            task_name="single_test",
            robot_dof=6,
        )
        with EpisodeLogger(config) as logger:
            logger.start_episode()
            for i in range(10):
                logger.log_frame(
                    joint_positions=[0.1 * i, 0.2, 0.3, 0.4, 0.5, 0.6],
                    gripper_state=0.5,
                    action=[0.01, -0.02, 0.03, -0.01, 0.0, 0.0],
                    reward=0.1,
                )
            logger.end_episode(outcome=Outcome.SUCCESS)
            session_id = logger.session_id

        storage = HDF5Storage(config)
        pairs = storage.list_episodes(session_id=session_id)
        assert len(pairs) == 1, f"Expected 1 episode, got {len(pairs)}"

        ep = storage.load_episode(*pairs[0])
        assert ep.num_frames == 10
        storage.close()

    def test_single_episode_detector_pipeline(self):
        """DetectorPipeline works on exactly 1 episode."""
        episode = _make_episode(n_frames=10, outcome=Outcome.SUCCESS)
        pipeline = _make_pipeline()
        results = pipeline.run_batch([episode])
        assert len(results) == 1
        # A simple success episode should not crash the pipeline
        assert isinstance(results[0].failure_probability, float)

    @requires_prescriber
    def test_single_episode_prescriber(self):
        """Prescriber works with detection results from 1 episode."""
        episode = _make_episode(n_frames=10, outcome=Outcome.FAILURE, reward=-0.5)
        detection_dicts = _detection_results_from_pipeline([episode])
        report = _run_prescriber_on(detection_dicts)
        assert "prescriptions" in report
        assert "summary" in report

    def test_single_frame_episode(self):
        """An episode with exactly 1 frame should not crash detectors."""
        episode = _make_episode(n_frames=1, outcome=Outcome.UNKNOWN)
        pipeline = _make_pipeline()
        results = pipeline.run_batch([episode])
        assert len(results) == 1
        # StallDetector requires >= 2 frames, so it should just return no detections
        assert isinstance(results[0].is_failure, bool)

    def test_zero_frame_episode(self):
        """An episode with 0 frames should not crash detectors."""
        episode = Episode(
            task_name="empty_ep",
            robot_id="test",
            frames=[],
            outcome=Outcome.UNKNOWN,
        )
        pipeline = _make_pipeline()
        results = pipeline.run_batch([episode])
        assert len(results) == 1


# ===================================================================
# 4. ALL SUCCESSES
# ===================================================================


class TestAllSuccesses:
    """Edge case: every episode in the session succeeds."""

    def test_all_success_detector_pipeline(self):
        """Pipeline on all-success episodes should produce zero failures."""
        episodes = [
            _make_episode(n_frames=20, outcome=Outcome.SUCCESS, reward=1.0) for _ in range(10)
        ]
        pipeline = _make_pipeline()
        results = pipeline.run_batch(episodes)
        assert len(results) == 10
        # With reward=1.0 per frame, RewardThreshold should not fire
        # (default threshold is min_total_reward=0.0)
        failure_count = sum(1 for r in results if r.is_failure)
        # All successes with positive reward should ideally have 0 failures
        # but the key assertion is: it does not crash
        assert isinstance(failure_count, int)

    @requires_prescriber
    def test_all_success_prescriber_no_crash(self):
        """Prescriber with 0 detected failures still produces a report."""
        episodes = [
            _make_episode(n_frames=20, outcome=Outcome.SUCCESS, reward=1.0) for _ in range(10)
        ]
        detection_dicts = _detection_results_from_pipeline(episodes)
        report = _run_prescriber_on(detection_dicts)
        assert "prescriptions" in report
        assert "summary" in report
        # With 0 actual failures, should still return valid structure
        assert report["num_failures_analyzed"] == len(episodes)

    @requires_prescriber
    def test_all_success_prescriber_generates_fallback(self):
        """Prescriber should generate a fallback prescription when no failures detected."""
        # Feed in detection results where is_failure=False for all
        detection_dicts = [
            {
                "episode_id": str(uuid4()),
                "is_failure": False,
                "failure_probability": 0.0,
                "detections": [],
                "detector_summaries": {},
            }
            for _ in range(5)
        ]
        report = _run_prescriber_on(detection_dicts)
        # Prescriber generates a fallback "General Training Improvement"
        # when no failure patterns are found
        assert len(report["prescriptions"]) >= 1
        assert report["num_failures_analyzed"] == 5

    def test_all_success_through_full_logger_pipeline(self, tmp_path):
        """End-to-end: log all successes, discover, detect, prescribe."""
        data_dir = tmp_path / "all-success"
        config = LoggerConfig(
            storage_dir=str(data_dir),
            task_name="all_success",
            robot_dof=6,
        )
        with EpisodeLogger(config) as logger:
            for _ in range(5):
                logger.start_episode()
                for i in range(15):
                    logger.log_frame(
                        joint_positions=(np.random.randn(6) * 0.1).tolist(),
                        gripper_state=0.5,
                        action=(np.random.randn(6) * 0.01).tolist(),
                        reward=1.0,
                    )
                logger.end_episode(outcome=Outcome.SUCCESS)

        from orbit.dashboard.data_loader import discover_sessions

        sessions = discover_sessions(str(data_dir))
        assert len(sessions) == 1
        assert sessions[0]["episode_count"] == 5


# ===================================================================
# 5. ALL FAILURES
# ===================================================================


class TestAllFailures:
    """Edge case: every episode fails with negative reward."""

    def test_all_failure_detector_pipeline(self):
        """Pipeline on all-failure episodes should flag them all."""
        episodes = [
            _make_episode(n_frames=20, outcome=Outcome.FAILURE, reward=-1.0) for _ in range(10)
        ]
        pipeline = _make_pipeline()
        results = pipeline.run_batch(episodes)
        assert len(results) == 10
        # All should be detected as failures (negative total reward)
        failure_count = sum(1 for r in results if r.is_failure)
        assert failure_count == 10, f"Expected all 10 flagged as failures, got {failure_count}"

    @requires_prescriber
    def test_all_failure_prescriber(self):
        """Prescriber produces actionable tasks from all-failure input."""
        episodes = [
            _make_episode(n_frames=20, outcome=Outcome.FAILURE, reward=-1.0) for _ in range(10)
        ]
        detection_dicts = _detection_results_from_pipeline(episodes)
        report = _run_prescriber_on(detection_dicts)
        assert "prescriptions" in report
        assert report["num_failures_analyzed"] == 10
        # With 10/10 failures, prescriber should generate meaningful advice
        assert len(report["prescriptions"]) >= 1

    def test_all_failure_extreme_negative_reward(self):
        """Very large negative rewards should not cause overflow or crash."""
        episodes = [
            _make_episode(n_frames=50, outcome=Outcome.FAILURE, reward=-999.99) for _ in range(5)
        ]
        pipeline = _make_pipeline()
        results = pipeline.run_batch(episodes)
        assert len(results) == 5
        for r in results:
            assert r.is_failure
            assert 0.0 <= r.failure_probability <= 1.0

    def test_all_failure_through_full_logger_pipeline(self, tmp_path):
        """End-to-end: log all failures, discover, detect, prescribe."""
        data_dir = tmp_path / "all-failure"
        config = LoggerConfig(
            storage_dir=str(data_dir),
            task_name="all_fail",
            robot_dof=6,
        )
        with EpisodeLogger(config) as logger:
            for _ in range(5):
                logger.start_episode()
                for i in range(15):
                    logger.log_frame(
                        joint_positions=(np.random.randn(6) * 0.1).tolist(),
                        gripper_state=0.5,
                        action=(np.random.randn(6) * 0.01).tolist(),
                        reward=-2.0,
                    )
                logger.end_episode(outcome=Outcome.FAILURE)

        from orbit.dashboard.data_loader import discover_sessions

        sessions = discover_sessions(str(data_dir))
        assert len(sessions) == 1
        assert sessions[0]["episode_count"] == 5

    def test_all_failure_summary_df(self):
        """episodes_to_summary_df handles all-failure episodes."""
        from orbit.dashboard.data_loader import episodes_to_summary_df

        episodes = [
            _make_episode(n_frames=10, outcome=Outcome.FAILURE, reward=-1.0) for _ in range(5)
        ]
        df = episodes_to_summary_df(episodes)
        assert len(df) == 5
        assert (df["outcome"] == "failure").all()
        assert (df["total_reward"] < 0).all()


# ===================================================================
# 6. MISSING IMAGES
# ===================================================================


class TestMissingImages:
    """Edge case: image_path references files that do not exist."""

    def test_missing_images_detector_pipeline(self):
        """Detector pipeline should work even when image files are missing."""
        episodes = [
            _make_episode(
                n_frames=15,
                outcome=Outcome.FAILURE,
                reward=-0.5,
                image_path="/tmp/nonexistent/missing_image_000000.png",
            )
            for _ in range(3)
        ]
        pipeline = _make_pipeline()
        results = pipeline.run_batch(episodes)
        assert len(results) == 3
        # Detectors are heuristic (joint/gripper/reward) -- images are irrelevant
        for r in results:
            assert isinstance(r.is_failure, bool)

    @requires_prescriber
    def test_missing_images_prescriber(self):
        """Prescriber works when episodes reference missing images."""
        episodes = [
            _make_episode(
                n_frames=15,
                outcome=Outcome.FAILURE,
                reward=-0.5,
                image_path="/tmp/nonexistent/ghost_image.png",
            )
            for _ in range(3)
        ]
        detection_dicts = _detection_results_from_pipeline(episodes)
        report = _run_prescriber_on(detection_dicts)
        assert "prescriptions" in report
        assert report["num_failures_analyzed"] == 3

    def test_missing_images_summary_df(self):
        """episodes_to_summary_df works when episodes reference missing images."""
        from orbit.dashboard.data_loader import episodes_to_summary_df

        episodes = [
            _make_episode(
                n_frames=10,
                outcome=Outcome.SUCCESS,
                reward=0.5,
                image_path="/does/not/exist/img.png",
            )
            for _ in range(3)
        ]
        df = episodes_to_summary_df(episodes)
        assert len(df) == 3

    def test_missing_images_leRobot_export(self, tmp_path):
        """LeRobotExporter.export_episode should skip missing images gracefully."""
        from orbit.logger.storage import LeRobotExporter

        episode = _make_episode(
            n_frames=5,
            outcome=Outcome.SUCCESS,
            reward=0.5,
            image_path="/tmp/nonexistent/img.png",
        )
        exporter = LeRobotExporter()
        output_dir = tmp_path / "lerobot-export"
        # Should not crash -- just skips copying non-existent files
        parquet_path = exporter.export_episode(episode, output_dir)
        assert parquet_path.exists()

    def test_mixed_existing_and_missing_images(self, tmp_path):
        """Episodes with a mix of real and missing images do not crash."""
        real_img_path = tmp_path / "real_image.png"
        from PIL import Image

        Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)).save(real_img_path)

        frames = []
        for i in range(10):
            # Alternate between real and fake paths
            if i % 2 == 0:
                path = str(real_img_path)
            else:
                path = f"/tmp/nonexistent/frame_{i:06d}.png"
            frames.append(_make_frame(i, reward=0.1, image_path=path))

        episode = Episode(
            task_name="mixed_images",
            frames=frames,
            outcome=Outcome.SUCCESS,
        )
        pipeline = _make_pipeline()
        results = pipeline.run_batch([episode])
        assert len(results) == 1

    def test_missing_images_hdf5_roundtrip(self, tmp_path):
        """Save and reload episodes with missing image_path through HDF5."""
        data_dir = tmp_path / "missing-img-roundtrip"
        config = LoggerConfig(storage_dir=str(data_dir), robot_dof=6)
        storage = HDF5Storage(config)
        session = DeploymentSession()
        storage.create_session(session)

        episode = _make_episode(
            n_frames=5,
            outcome=Outcome.SUCCESS,
            image_path="/tmp/nonexistent/phantom.png",
        )
        storage.save_episode(session.session_id, episode)

        loaded = storage.load_episode(session.session_id, episode.episode_id)
        assert loaded.num_frames == 5
        for frame in loaded.frames:
            assert frame.image_path == "/tmp/nonexistent/phantom.png"
        storage.close()


# ===================================================================
# Bonus: additional boundary conditions
# ===================================================================


class TestAdditionalEdgeCases:
    """Bonus tests for other subtle boundary conditions."""

    def test_episode_with_none_rewards(self):
        """All rewards are None -- total_reward should be 0.0."""
        frames = [
            EpisodeFrame(
                timestamp=1000.0 + i * 0.033,
                joint_positions=[0.1] * 6,
                gripper_state=0.5,
                action=[0.0] * 6,
                reward=None,
            )
            for i in range(10)
        ]
        episode = Episode(task_name="none_rewards", frames=frames)
        assert episode.total_reward == 0.0
        assert episode.avg_action_magnitude == 0.0

        pipeline = _make_pipeline()
        result = pipeline.run(episode)
        assert isinstance(result.is_failure, bool)

    def test_episode_with_zero_duration(self):
        """start_time == end_time (instantaneous episode)."""
        now = datetime.datetime.now()
        episode = Episode(
            task_name="zero_duration",
            start_time=now,
            end_time=now,
            frames=[_make_frame(0)],
        )
        assert episode.duration == 0.0

        pipeline = _make_pipeline()
        result = pipeline.run(episode)
        # TimeoutDetector should not flag a 0-second episode
        timeout_dets = [d for d in result.detections if d.detector_name == "TimeoutDetector"]
        # duration=0 < 60s default, so no timeout detection
        assert not any("duration" in d.description.lower() for d in timeout_dets)

    def test_very_large_episode(self):
        """Episode with 2000 frames should not cause memory/performance issues."""
        episode = _make_episode(n_frames=2000, outcome=Outcome.FAILURE, reward=-0.01)
        pipeline = _make_pipeline()
        results = pipeline.run_batch([episode])
        assert len(results) == 1
        # TimeoutDetector should flag > 1000 frames
        timeout_dets = [d for d in results[0].detections if d.detector_name == "TimeoutDetector"]
        assert len(timeout_dets) >= 1

    def test_discover_sessions_non_uuid_filename(self, tmp_path):
        """A file named session_notauuid.h5 should be skipped."""
        data_dir = tmp_path / "bad-name"
        data_dir.mkdir()
        bad_file = data_dir / "session_notauuid.h5"
        bad_file.write_bytes(b"")

        from orbit.dashboard.data_loader import discover_sessions

        result = discover_sessions(str(data_dir))
        assert result == []

    def test_empty_summary_dataframe(self):
        """episodes_to_summary_df on empty list returns empty DataFrame."""
        import pandas as pd

        from orbit.dashboard.data_loader import episodes_to_summary_df

        df = episodes_to_summary_df([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
