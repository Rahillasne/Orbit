"""Tests for orbit.profile.quality (QualityEstimator)."""

from __future__ import annotations

import numpy as np

from orbit.profile.quality import QualityEstimator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_deterministic_episode(
    episode_id: int, T: int = 100, state_dim: int = 4, seed: int = 0
) -> dict:
    """Episode where actions = linear(states) + tiny noise."""
    rng = np.random.default_rng(seed)
    states = rng.standard_normal((T, state_dim))
    W = rng.standard_normal((state_dim, state_dim))
    actions = states @ W + rng.standard_normal((T, state_dim)) * 0.01
    return {"episode_id": episode_id, "states": states, "actions": actions}


def _make_random_episode(episode_id: int, T: int = 100, state_dim: int = 4, seed: int = 0) -> dict:
    """Episode where actions are independent of states."""
    rng = np.random.default_rng(seed)
    states = rng.standard_normal((T, state_dim))
    actions = rng.standard_normal((T, state_dim))
    return {"episode_id": episode_id, "states": states, "actions": actions}


# ---------------------------------------------------------------------------
# Tests — existing MI-based quality
# ---------------------------------------------------------------------------


class TestQualityEstimator:
    def test_deterministic_high_quality(self):
        """Deterministic actions should yield high MI."""
        eps = [_make_deterministic_episode(i, T=200, seed=i) for i in range(3)]
        est = QualityEstimator(k_neighbors=5)
        result = est.estimate_quality(eps)

        assert result.mutual_information_estimate > 0.5
        assert result.aggregate_score > 0.0

    def test_random_actions_low_quality(self):
        """Independent actions should yield MI ≈ 0."""
        eps = [_make_random_episode(i, T=200, seed=i) for i in range(3)]
        est = QualityEstimator(k_neighbors=5)
        result = est.estimate_quality(eps)

        assert result.mutual_information_estimate < 0.5

    def test_mixed_quality(self):
        """Bad episodes should be identified as low quality."""
        good = [_make_deterministic_episode(i, T=150, seed=i) for i in range(3)]
        # Bad episode: constant actions (zero variance → no MI contribution)
        rng = np.random.default_rng(99)
        bad = {
            "episode_id": 99,
            "states": rng.standard_normal((150, 4)),
            "actions": np.zeros((150, 4)),
        }
        eps = good + [bad]
        est = QualityEstimator(k_neighbors=5)
        result = est.estimate_quality(eps)

        # The bad episode should have the lowest score
        bad_score = result.episode_scores[99]
        good_scores = [result.episode_scores[i] for i in range(3)]
        assert bad_score <= min(good_scores)

    def test_ksg_estimator(self):
        """For bivariate Gaussian with ρ=0.8, verify MI within 20% of truth."""
        rng = np.random.default_rng(42)
        N = 2000
        rho = 0.8

        # Generate correlated bivariate Gaussian
        mean = [0, 0]
        cov = [[1, rho], [rho, 1]]
        data = rng.multivariate_normal(mean, cov, size=N)
        X = data[:, :1]
        Y = data[:, 1:]

        # Analytical MI for bivariate Gaussian
        true_mi = -0.5 * np.log(1 - rho**2)

        est = QualityEstimator(k_neighbors=5)
        estimated_mi = est._knn_mutual_information(X, Y, k=5)

        # Within 20% of true value
        assert abs(estimated_mi - true_mi) / true_mi < 0.20, (
            f"KSG estimate {estimated_mi:.4f} not within 20% of true MI {true_mi:.4f}"
        )

    def test_edge_case_single_episode(self):
        """Single episode should work without crash."""
        ep = _make_deterministic_episode(0, T=50)
        est = QualityEstimator(k_neighbors=5)
        result = est.estimate_quality([ep])

        assert result.mutual_information_estimate >= 0.0
        assert 0 in result.episode_scores

    def test_edge_case_short_episode(self):
        """Episode with 3 timesteps (< k) should be handled gracefully."""
        short = {
            "episode_id": 0,
            "states": np.random.randn(3, 4),
            "actions": np.random.randn(3, 4),
        }
        long = _make_deterministic_episode(1, T=100, seed=1)
        est = QualityEstimator(k_neighbors=5)
        result = est.estimate_quality([short, long])

        assert 0 in result.episode_scores
        assert 1 in result.episode_scores


# ---------------------------------------------------------------------------
# Tests — new quality signals
# ---------------------------------------------------------------------------


class TestActionSmoothness:
    def test_smooth_vs_jerky(self):
        """Smooth sinusoidal actions should score higher than random jerk."""
        est = QualityEstimator()

        # Smooth: sinusoidal trajectory
        t = np.linspace(0, 2 * np.pi, 200)
        smooth_actions = np.column_stack([np.sin(t), np.cos(t), np.sin(2 * t), np.cos(2 * t)])
        smooth_score = est._action_smoothness(smooth_actions)

        # Jerky: random independent samples
        rng = np.random.default_rng(42)
        jerky_actions = rng.standard_normal((200, 4))
        jerky_score = est._action_smoothness(jerky_actions)

        assert smooth_score > jerky_score, (
            f"Smooth ({smooth_score:.3f}) should be > jerky ({jerky_score:.3f})"
        )
        assert 0.0 <= smooth_score <= 1.0
        assert 0.0 <= jerky_score <= 1.0

    def test_short_trajectory(self):
        """Trajectory with < 4 steps returns default 0.5."""
        est = QualityEstimator()
        actions = np.array([[1, 2], [3, 4], [5, 6]])
        assert est._action_smoothness(actions) == 0.5


class TestEpisodeCompletion:
    def test_converging_vs_diverging(self):
        """Converging states should score higher than diverging ones."""
        est = QualityEstimator()

        # Converging: state variance decreases over time
        rng = np.random.default_rng(42)
        T = 100
        converging = np.zeros((T, 4))
        for i in range(T):
            scale = 1.0 - 0.9 * (i / T)  # shrinks over time
            converging[i] = rng.standard_normal(4) * scale
        conv_score = est._episode_completion(converging)

        # Diverging: state variance increases over time
        diverging = np.zeros((T, 4))
        for i in range(T):
            scale = 0.1 + 2.0 * (i / T)  # grows over time
            diverging[i] = rng.standard_normal(4) * scale
        div_score = est._episode_completion(diverging)

        assert conv_score > div_score, (
            f"Converging ({conv_score:.3f}) should be > diverging ({div_score:.3f})"
        )

    def test_short_trajectory(self):
        """Short trajectory (< 10 steps) returns default 0.5."""
        est = QualityEstimator()
        states = np.random.randn(5, 4)
        assert est._episode_completion(states) == 0.5


class TestObservationConsistency:
    def test_clean_vs_corrupted(self):
        """Clean data should score higher than data with NaN injected."""
        est = QualityEstimator()

        rng = np.random.default_rng(42)
        clean = rng.standard_normal((100, 4))
        clean_score = est._observation_consistency(clean)

        corrupted = clean.copy()
        corrupted[10] = np.nan  # inject NaN
        corrupted[50] = np.inf  # inject inf
        corrupted_score = est._observation_consistency(corrupted)

        assert clean_score > corrupted_score, (
            f"Clean ({clean_score:.3f}) should be > corrupted ({corrupted_score:.3f})"
        )
        assert clean_score <= 1.0

    def test_sudden_jumps(self):
        """States with sudden jumps should score lower."""
        est = QualityEstimator()

        rng = np.random.default_rng(42)
        smooth = rng.standard_normal((100, 4)) * 0.01
        smooth_score = est._observation_consistency(smooth)

        # Insert large jumps
        with_jumps = smooth.copy()
        with_jumps[30] = [100, 100, 100, 100]
        with_jumps[60] = [-100, -100, -100, -100]
        jump_score = est._observation_consistency(with_jumps)

        assert smooth_score >= jump_score


class TestDemonstrationQuality:
    def test_expert_vs_random(self):
        """Moderate-variance consistent actions should beat high-variance random."""
        est = QualityEstimator()

        rng = np.random.default_rng(42)

        # Expert-like: moderate, consistent variance across dimensions
        expert = rng.standard_normal((200, 4)) * 0.5
        expert_score = est._demonstration_quality(expert)

        # Random/noisy: very high variance
        noisy = rng.standard_normal((200, 4)) * 100.0
        noisy_score = est._demonstration_quality(noisy)

        assert expert_score > noisy_score, (
            f"Expert ({expert_score:.3f}) should be > noisy ({noisy_score:.3f})"
        )

    def test_constant_actions(self):
        """Constant actions (degenerate) should score very low."""
        est = QualityEstimator()
        constant = np.ones((100, 4)) * 0.5
        score = est._demonstration_quality(constant)
        assert score <= 0.2, f"Constant actions should score very low, got {score:.3f}"


class TestQualitySignalBreakdown:
    def test_breakdown_populated(self):
        """Signal breakdown should be populated with all values in [0, 1]."""
        eps = [_make_deterministic_episode(i, T=100, seed=i) for i in range(3)]
        est = QualityEstimator(k_neighbors=5)
        result = est.estimate_quality(eps)

        assert result.signal_breakdown is not None
        sb = result.signal_breakdown
        for field_name in [
            "mutual_information",
            "action_smoothness",
            "episode_completion",
            "observation_consistency",
            "demonstration_quality",
        ]:
            val = getattr(sb, field_name)
            assert 0.0 <= val <= 1.0, f"{field_name} = {val} not in [0, 1]"

    def test_backward_compat_aggregate_score(self):
        """Aggregate score should still work and be in [0, 1]."""
        eps = [_make_deterministic_episode(i, T=100, seed=i) for i in range(3)]
        est = QualityEstimator(k_neighbors=5)
        result = est.estimate_quality(eps)

        assert 0.0 <= result.aggregate_score <= 1.0
        assert len(result.episode_scores) == 3
