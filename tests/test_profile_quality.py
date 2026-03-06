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
# Tests
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
