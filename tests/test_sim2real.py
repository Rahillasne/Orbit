"""Tests for the sim-to-real transfer readiness profiler."""

from __future__ import annotations

import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest

from orbit.sim2real_profiler import (
    Sim2RealProfiler,
    Sim2RealReport,
    _combine_sub_scores,
    compute_action_similarity,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_episodes(n_episodes: int, n_steps: int, action_dim: int, rng, offset: float = 0.0):
    """Create synthetic episode dicts."""
    episodes = []
    for i in range(n_episodes):
        episodes.append(
            {
                "episode_id": i,
                "states": rng.standard_normal((n_steps, 6)).astype(np.float32),
                "actions": (rng.standard_normal((n_steps, action_dim)) + offset).astype(
                    np.float32
                ),
            }
        )
    return episodes


def _write_hdf5_dataset(path, episodes, n_images: int = 5):
    """Write a minimal HDF5 dataset with images for testing."""
    import h5py
    from PIL import Image

    path.mkdir(parents=True, exist_ok=True)
    img_dir = path / "images"
    img_dir.mkdir(exist_ok=True)

    # Create some images
    rng = np.random.default_rng(42)
    image_paths = []
    for i in range(n_images):
        img_arr = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        img_path = img_dir / f"frame_{i:04d}.png"
        Image.fromarray(img_arr).save(img_path)
        image_paths.append(str(img_path))

    h5_path = path / "session_test.h5"
    with h5py.File(h5_path, "w") as f:
        eps_grp = f.create_group("episodes")
        for ep in episodes:
            grp = eps_grp.create_group(str(ep["episode_id"]))
            grp.create_dataset("states", data=ep["states"])
            grp.create_dataset("actions", data=ep["actions"])
            # Assign images round-robin
            ep_imgs = [image_paths[j % len(image_paths)] for j in range(len(ep["states"]))]
            dt = h5py.string_dtype()
            grp.create_dataset("image_paths", data=ep_imgs, dtype=dt)

    return h5_path


# ------------------------------------------------------------------
# Action similarity tests
# ------------------------------------------------------------------


class TestActionSimilarity:
    def test_identical_distributions(self):
        rng = np.random.default_rng(0)
        episodes = _make_episodes(5, 20, 4, rng)
        similarity = compute_action_similarity(episodes, episodes)
        assert similarity == pytest.approx(1.0, abs=0.01)

    def test_different_distributions(self):
        rng = np.random.default_rng(0)
        sim_eps = _make_episodes(5, 20, 4, rng, offset=0.0)
        rng2 = np.random.default_rng(1)
        real_eps = _make_episodes(5, 20, 4, rng2, offset=10.0)
        similarity = compute_action_similarity(sim_eps, real_eps)
        assert similarity < 0.5

    def test_empty_episodes(self):
        assert compute_action_similarity([], []) == 0.0

    def test_mismatched_action_dims(self):
        rng = np.random.default_rng(0)
        sim_eps = _make_episodes(3, 10, 4, rng)
        real_eps = _make_episodes(3, 10, 6, rng)
        similarity = compute_action_similarity(sim_eps, real_eps)
        # Should still work, using min dimension
        assert 0.0 <= similarity <= 1.0


# ------------------------------------------------------------------
# Sub-score combination
# ------------------------------------------------------------------


class TestCombineSubScores:
    def test_perfect_scores(self):
        sub = {
            "embedding_overlap": 1.0,
            "visual_gap": 0.0,
            "action_similarity": 1.0,
            "diversity": 1.0,
        }
        assert _combine_sub_scores(sub) == pytest.approx(1.0, abs=0.01)

    def test_worst_scores(self):
        sub = {
            "embedding_overlap": 0.0,
            "visual_gap": 1.0,
            "action_similarity": 0.0,
            "diversity": 0.0,
        }
        assert _combine_sub_scores(sub) == pytest.approx(0.0, abs=0.01)

    def test_weights_sum_to_one(self):
        from orbit.sim2real_profiler import _WEIGHTS

        assert sum(_WEIGHTS.values()) == pytest.approx(1.0)


# ------------------------------------------------------------------
# Report structure
# ------------------------------------------------------------------


class TestSim2RealReport:
    def test_to_dict_structure(self):
        report = Sim2RealReport(
            overall_transfer_score=0.65,
            per_task_scores={
                "pick": {"score": 0.7, "embedding_overlap": 0.8, "visual_gap": 0.2,
                         "action_similarity": 0.6, "diversity": 0.5},
                "place": {"score": 0.6, "embedding_overlap": 0.7, "visual_gap": 0.3,
                          "action_similarity": 0.5, "diversity": 0.4},
            },
            gap_analysis={"biggest_gaps": [], "recommendations": []},
            prescription=[],
        )
        d = report.to_dict()
        assert "overall_transfer_score" in d
        assert "per_task_scores" in d
        assert "gap_analysis" in d
        assert "prescription" in d
        assert isinstance(d["per_task_scores"]["pick"]["score"], float)

    def test_to_json_valid(self):
        report = Sim2RealReport(
            overall_transfer_score=0.5,
            per_task_scores={"task1": {"score": 0.5}},
            gap_analysis={},
            prescription=[],
        )
        parsed = json.loads(report.to_json())
        assert parsed["overall_transfer_score"] == 0.5


# ------------------------------------------------------------------
# Prescription generation
# ------------------------------------------------------------------


class TestPrescription:
    def test_low_score_generates_prescription(self):
        Sim2RealProfiler.__new__(Sim2RealProfiler)
        per_task = {
            "hard_task": {
                "score": 0.3,
                "embedding_overlap": 0.2,
                "visual_gap": 0.6,
                "action_similarity": 0.3,
                "diversity": 0.4,
            }
        }
        prescriptions = Sim2RealProfiler._build_prescription(per_task)
        assert len(prescriptions) == 1
        assert prescriptions[0]["task"] == "hard_task"
        assert prescriptions[0]["priority"] == 1
        assert prescriptions[0]["estimated_demos"] >= 5

    def test_high_score_no_prescription(self):
        per_task = {
            "easy_task": {
                "score": 0.85,
                "embedding_overlap": 0.9,
                "visual_gap": 0.1,
                "action_similarity": 0.8,
                "diversity": 0.7,
            }
        }
        prescriptions = Sim2RealProfiler._build_prescription(per_task)
        assert len(prescriptions) == 0


# ------------------------------------------------------------------
# Gap analysis
# ------------------------------------------------------------------


class TestGapAnalysis:
    def test_recommendations_for_low_overlap(self):
        per_task = {"t": {"score": 0.4}}
        sub = {
            "embedding_overlap": 0.3,
            "visual_gap": 0.6,
            "action_similarity": 0.3,
            "diversity": 0.3,
        }
        gap = Sim2RealProfiler._build_gap_analysis(per_task, sub)
        assert len(gap["recommendations"]) >= 3
        assert "domain_shift_summary" in gap
        assert "biggest_gaps" in gap

    def test_no_recommendations_for_good_scores(self):
        per_task = {"t": {"score": 0.9}}
        sub = {
            "embedding_overlap": 0.8,
            "visual_gap": 0.1,
            "action_similarity": 0.9,
            "diversity": 0.8,
        }
        gap = Sim2RealProfiler._build_gap_analysis(per_task, sub)
        assert len(gap["recommendations"]) == 0


# ------------------------------------------------------------------
# No-tasks fallback
# ------------------------------------------------------------------


class TestNoTasksFallback:
    def test_no_tasks_returns_general(self):
        profiler = Sim2RealProfiler.__new__(Sim2RealProfiler)
        profiler.embedding_model = "google/siglip-base-patch16-224"
        profiler.device = "cpu"
        sub = {
            "embedding_overlap": 0.5,
            "visual_gap": 0.3,
            "action_similarity": 0.6,
            "diversity": 0.4,
        }
        result = profiler._score_tasks(None, None, [], None, sub)
        assert "general" in result
        assert 0.0 <= result["general"]["score"] <= 1.0


# ------------------------------------------------------------------
# CLI integration
# ------------------------------------------------------------------


class TestCLI:
    def test_cli_help(self):
        from click.testing import CliRunner

        from orbit.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["sim2real", "--help"])
        assert result.exit_code == 0
        assert "--sim-data" in result.output
        assert "--real-data" in result.output

    def test_cli_missing_args(self):
        from click.testing import CliRunner

        from orbit.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["sim2real"])
        assert result.exit_code != 0


# ------------------------------------------------------------------
# Integration with synthetic HDF5 data
# ------------------------------------------------------------------


class TestIntegration:
    @pytest.fixture
    def sim_and_real_dirs(self, tmp_path):
        """Create two synthetic HDF5 datasets."""
        rng = np.random.default_rng(42)

        sim_dir = tmp_path / "sim"
        sim_eps = _make_episodes(3, 10, 4, rng, offset=0.0)
        _write_hdf5_dataset(sim_dir, sim_eps, n_images=8)

        real_dir = tmp_path / "real"
        rng2 = np.random.default_rng(99)
        real_eps = _make_episodes(3, 10, 4, rng2, offset=1.0)
        _write_hdf5_dataset(real_dir, real_eps, n_images=8)

        return sim_dir, real_dir

    def test_full_pipeline(self, sim_and_real_dirs):
        """End-to-end test with synthetic data (uses random projection fallback)."""
        sim_dir, real_dir = sim_and_real_dirs
        profiler = Sim2RealProfiler()
        report = profiler.analyze(str(sim_dir), str(real_dir))

        assert 0.0 <= report.overall_transfer_score <= 1.0
        assert "general" in report.per_task_scores
        assert isinstance(report.gap_analysis, dict)
        assert isinstance(report.prescription, list)

        # Verify JSON serialization round-trip
        parsed = json.loads(report.to_json())
        assert "overall_transfer_score" in parsed

    def test_full_pipeline_with_tasks(self, sim_and_real_dirs):
        """End-to-end test with task descriptions."""
        sim_dir, real_dir = sim_and_real_dirs
        profiler = Sim2RealProfiler()
        report = profiler.analyze(
            str(sim_dir), str(real_dir), task_descriptions=["pick object"]
        )

        assert "pick object" in report.per_task_scores
        assert "general" not in report.per_task_scores
