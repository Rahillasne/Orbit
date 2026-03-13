"""Tests for the benchmark validation pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from orbit.benchmarks.validate_profiler import (
    BenchmarkEntry,
    BenchmarkResult,
    BenchmarkValidator,
    BootstrapCI,
    CorrelationResult,
    ValidationReport,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GROUND_TRUTH_PATH = Path(__file__).parent.parent / "orbit" / "benchmarks" / "ground_truth.json"


def _make_entry(id: str = "test", rate: float = 0.8) -> BenchmarkEntry:
    return BenchmarkEntry(
        id=id,
        repo_id=f"lerobot/{id}",
        task_description=f"do {id}",
        reported_success_rate=rate,
        metric_type="success_rate",
        source="test",
        policy="test_policy",
        notes="",
        max_episodes=10,
    )


def _make_result(
    id: str = "test",
    orbit_score: float = 0.7,
    ground_truth: float = 0.8,
    error: str | None = None,
) -> BenchmarkResult:
    return BenchmarkResult(
        entry=_make_entry(id, ground_truth),
        orbit_score=orbit_score,
        orbit_confidence=0.5,
        ground_truth=ground_truth,
        residual=orbit_score - ground_truth,
        abs_error=abs(orbit_score - ground_truth),
        profiling_time_s=1.0,
        error=error,
    )


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------


class TestLoadGroundTruth:
    def test_load_ground_truth(self):
        """The shipped ground_truth.json loads correctly with all required fields."""
        validator = BenchmarkValidator()
        entries = validator._load_ground_truth()
        assert len(entries) >= 5
        for e in entries:
            assert e.id
            assert e.repo_id
            assert e.task_description
            assert 0.0 <= e.reported_success_rate <= 1.0
            assert e.source

    def test_ground_truth_json_valid(self):
        """The JSON file is valid and has expected structure."""
        with open(GROUND_TRUTH_PATH) as f:
            data = json.load(f)
        assert "version" in data
        assert "benchmarks" in data
        assert isinstance(data["benchmarks"], list)
        ids = [b["id"] for b in data["benchmarks"]]
        assert len(ids) == len(set(ids)), "Duplicate benchmark IDs"


# ---------------------------------------------------------------------------
# Correlation computation
# ---------------------------------------------------------------------------


class TestCorrelations:
    def test_perfect_positive(self):
        """Perfectly correlated data yields rho=1.0, r=1.0, rank_accuracy=1.0."""
        results = [
            _make_result("a", orbit_score=0.3, ground_truth=0.3),
            _make_result("b", orbit_score=0.5, ground_truth=0.5),
            _make_result("c", orbit_score=0.7, ground_truth=0.7),
            _make_result("d", orbit_score=0.9, ground_truth=0.9),
        ]
        validator = BenchmarkValidator()
        corr = validator._compute_correlations(results)
        assert corr is not None
        assert corr.spearman_rho == pytest.approx(1.0)
        assert corr.pearson_r == pytest.approx(1.0)
        assert corr.rank_accuracy == pytest.approx(1.0)
        assert corr.n_samples == 4

    def test_perfect_negative(self):
        """Inversely correlated data yields rho=-1.0."""
        results = [
            _make_result("a", orbit_score=0.9, ground_truth=0.3),
            _make_result("b", orbit_score=0.7, ground_truth=0.5),
            _make_result("c", orbit_score=0.5, ground_truth=0.7),
            _make_result("d", orbit_score=0.3, ground_truth=0.9),
        ]
        validator = BenchmarkValidator()
        corr = validator._compute_correlations(results)
        assert corr is not None
        assert corr.spearman_rho == pytest.approx(-1.0)
        assert corr.rank_accuracy == pytest.approx(0.0)

    def test_too_few_results(self):
        """With fewer than 3 results, returns None."""
        results = [
            _make_result("a", orbit_score=0.5, ground_truth=0.5),
            _make_result("b", orbit_score=0.7, ground_truth=0.7),
        ]
        validator = BenchmarkValidator()
        assert validator._compute_correlations(results) is None


# ---------------------------------------------------------------------------
# Rank accuracy
# ---------------------------------------------------------------------------


class TestRankAccuracy:
    def test_concordant(self):
        predicted = np.array([0.1, 0.5, 0.9])
        actual = np.array([0.2, 0.6, 0.8])
        assert BenchmarkValidator._rank_accuracy(predicted, actual) == pytest.approx(1.0)

    def test_discordant(self):
        predicted = np.array([0.9, 0.5, 0.1])
        actual = np.array([0.2, 0.6, 0.8])
        assert BenchmarkValidator._rank_accuracy(predicted, actual) == pytest.approx(0.0)

    def test_mixed(self):
        predicted = np.array([0.1, 0.9, 0.5])
        actual = np.array([0.2, 0.6, 0.8])
        # Pairs: (0,1) concordant, (0,2) concordant, (1,2) discordant → 2/3
        assert BenchmarkValidator._rank_accuracy(predicted, actual) == pytest.approx(2 / 3)

    def test_tied_actuals(self):
        """Tied actuals are excluded from comparison."""
        predicted = np.array([0.1, 0.5, 0.9])
        actual = np.array([0.5, 0.5, 0.8])
        # Only pairs involving index 2 count: (0,2) concordant, (1,2) concordant → 2/2
        assert BenchmarkValidator._rank_accuracy(predicted, actual) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


class TestBootstrap:
    def test_ci_contains_point_estimate(self):
        results = [
            _make_result("a", orbit_score=0.3, ground_truth=0.3),
            _make_result("b", orbit_score=0.5, ground_truth=0.5),
            _make_result("c", orbit_score=0.7, ground_truth=0.7),
            _make_result("d", orbit_score=0.9, ground_truth=0.9),
        ]
        validator = BenchmarkValidator(bootstrap_iterations=1000)
        bs_spearman, bs_pearson = validator._bootstrap_correlation(results)
        assert bs_spearman is not None
        assert bs_pearson is not None
        assert bs_spearman.ci_lower <= bs_spearman.point_estimate <= bs_spearman.ci_upper
        assert bs_pearson.ci_lower <= bs_pearson.point_estimate <= bs_pearson.ci_upper

    def test_too_few_for_bootstrap(self):
        results = [
            _make_result("a", orbit_score=0.5, ground_truth=0.5),
            _make_result("b", orbit_score=0.7, ground_truth=0.7),
            _make_result("c", orbit_score=0.9, ground_truth=0.9),
        ]
        validator = BenchmarkValidator()
        bs_s, bs_p = validator._bootstrap_correlation(results)
        assert bs_s is None
        assert bs_p is None


# ---------------------------------------------------------------------------
# Failure case identification
# ---------------------------------------------------------------------------


class TestFailureCases:
    def test_identifies_large_errors(self):
        results = [
            _make_result("good", orbit_score=0.79, ground_truth=0.8),  # abs_error=0.01
            _make_result("bad", orbit_score=0.4, ground_truth=0.9),  # abs_error=0.5
            _make_result("ok", orbit_score=0.55, ground_truth=0.65),  # abs_error=0.1
        ]
        validator = BenchmarkValidator(failure_threshold=0.25)
        failures = validator._identify_failure_cases(results)
        assert len(failures) == 1
        assert failures[0].entry.id == "bad"

    def test_no_failures(self):
        results = [
            _make_result("a", orbit_score=0.79, ground_truth=0.8),
            _make_result("b", orbit_score=0.60, ground_truth=0.65),
        ]
        validator = BenchmarkValidator(failure_threshold=0.25)
        assert validator._identify_failure_cases(results) == []


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


class TestReport:
    def _sample_report(self) -> ValidationReport:
        results = [
            _make_result("a", orbit_score=0.75, ground_truth=0.8),
            _make_result("b", orbit_score=0.60, ground_truth=0.65),
            _make_result("c", orbit_score=0.90, ground_truth=0.91),
            _make_result("err", orbit_score=0.0, ground_truth=0.5, error="download failed"),
        ]
        return ValidationReport(
            results=results,
            correlation=CorrelationResult(
                spearman_rho=0.95,
                spearman_p=0.01,
                pearson_r=0.93,
                pearson_p=0.02,
                rank_accuracy=0.9,
                n_samples=3,
            ),
            bootstrap_spearman=BootstrapCI(0.95, 0.80, 1.0, 0.95, 9500),
            bootstrap_pearson=BootstrapCI(0.93, 0.78, 0.99, 0.95, 9500),
            failure_cases=[],
            metadata={"total_time_s": 42.0},
        )

    def test_to_markdown_sections(self):
        md = self._sample_report().to_markdown()
        assert "# ORBIT Profiler Benchmark Validation" in md
        assert "## Summary" in md
        assert "## Per-Dataset Results" in md
        assert "## Statistical Notes" in md
        assert "Spearman rho" in md
        assert "Pearson r" in md
        assert len(md) > 500

    def test_to_json_valid(self):
        report = self._sample_report()
        text = report.to_json()
        data = json.loads(text)
        assert "results" in data
        assert "correlation" in data
        assert "bootstrap_spearman" in data
        assert "metadata" in data
        assert len(data["results"]) == 4

    def test_to_dict_roundtrip(self):
        report = self._sample_report()
        d = report.to_dict()
        # Should be JSON-serializable
        json.dumps(d)


# ---------------------------------------------------------------------------
# Profiler integration (mocked)
# ---------------------------------------------------------------------------


class TestProfileSingle:
    def test_graceful_failure(self):
        """When profiler raises, result has error field set."""
        validator = BenchmarkValidator(cache_dir="/tmp/orbit_test_bench_cache")
        entry = _make_entry("fail_test", 0.9)

        with patch(
            "orbit.profile.profiler.DatasetProfiler"
        ) as MockProfiler:
            MockProfiler.return_value.profile_from_hub.side_effect = RuntimeError("network error")
            result = validator._profile_single(entry, force=True)

        assert result.error is not None
        assert "network error" in result.error

    def test_successful_profiling(self, tmp_path):
        """Mock profiler returns a score and it's correctly captured."""
        validator = BenchmarkValidator(cache_dir=str(tmp_path / "cache"))
        entry = _make_entry("success_test", 0.85)

        mock_cap = MagicMock()
        mock_cap.score = 0.72
        mock_cap.confidence = 0.6

        mock_profile = MagicMock()
        mock_profile.capabilities = [mock_cap]

        with patch(
            "orbit.profile.profiler.DatasetProfiler"
        ) as MockProfiler:
            MockProfiler.return_value.profile_from_hub.return_value = mock_profile
            result = validator._profile_single(entry, force=True)

        assert result.error is None
        assert result.orbit_score == pytest.approx(0.72)
        assert result.orbit_confidence == pytest.approx(0.6)
        assert result.ground_truth == pytest.approx(0.85)
        assert result.residual == pytest.approx(0.72 - 0.85)

    def test_score_caching(self, tmp_path):
        """After profiling, the score is cached and reused on second call."""
        validator = BenchmarkValidator(cache_dir=str(tmp_path / "cache"))
        entry = _make_entry("cache_test", 0.75)

        mock_cap = MagicMock()
        mock_cap.score = 0.68
        mock_cap.confidence = 0.5

        mock_profile = MagicMock()
        mock_profile.capabilities = [mock_cap]

        with patch(
            "orbit.profile.profiler.DatasetProfiler"
        ) as MockProfiler:
            MockProfiler.return_value.profile_from_hub.return_value = mock_profile
            validator._profile_single(entry, force=True)
            result2 = validator._profile_single(entry, force=False)

            # Second call should NOT invoke the profiler again
            assert MockProfiler.return_value.profile_from_hub.call_count == 1

        assert result2.orbit_score == pytest.approx(0.68)
        assert result2.profiling_time_s == 0.0  # from cache


class TestRunWithMockProfiler:
    def test_full_pipeline(self, tmp_path):
        """Full validator.run() with mocked profiler produces valid report."""
        validator = BenchmarkValidator(
            cache_dir=str(tmp_path / "cache"),
            bootstrap_iterations=500,
        )

        # Map repo_id to (score, confidence) — must match repo_ids in ground_truth.json
        scores_by_repo = {
            "lerobot/pusht": (0.85, 0.7),
            "lerobot/aloha_sim_transfer_cube_human": (0.70, 0.6),
            "lerobot/aloha_sim_insertion_human": (0.75, 0.65),
            "lerobot/xarm_lift_medium_replay": (0.50, 0.4),
            "lerobot/pusht_keypoints": (0.72, 0.55),
            "lerobot/aloha_sim_transfer_cube_scripted": (0.88, 0.75),
            "lerobot/aloha_sim_insertion_scripted": (0.82, 0.7),
        }

        def fake_profile_from_hub(repo_id, task_descriptions, max_episodes, cache_dir):
            score, conf = scores_by_repo.get(repo_id, (0.5, 0.3))
            mock_cap = MagicMock()
            mock_cap.score = score
            mock_cap.confidence = conf
            mock_profile = MagicMock()
            mock_profile.capabilities = [mock_cap]
            return mock_profile

        with patch(
            "orbit.profile.profiler.DatasetProfiler"
        ) as MockProfiler:
            MockProfiler.return_value.profile_from_hub.side_effect = fake_profile_from_hub
            report = validator.run(force=True)

        assert len(report.results) >= 5
        assert all(r.error is None for r in report.results)
        assert report.correlation is not None
        assert report.correlation.n_samples >= 5

        # With correlated mock scores, expect positive correlation
        assert report.correlation.spearman_rho > 0
        assert report.correlation.pearson_r > 0

        # Bootstrap should exist
        assert report.bootstrap_spearman is not None
        assert report.bootstrap_pearson is not None

        # Reports render without error
        md = report.to_markdown()
        assert len(md) > 200
        j = json.loads(report.to_json())
        assert "results" in j
