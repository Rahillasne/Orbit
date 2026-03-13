"""Data quality metrics (mutual information + action/observation signals)."""

from __future__ import annotations

import logging
import warnings

import numpy as np
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from orbit.profile.types import QualityMetrics, QualitySignalBreakdown

logger = logging.getLogger(__name__)


class QualityEstimator:
    """Score episode data quality using multiple signals.

    Combines:
      - KSG mutual information between states and actions (DemInf-inspired)
      - Action smoothness (jerk of action trajectories)
      - Episode completion (state convergence toward goal)
      - Observation consistency (sensor corruption detection)
      - Demonstration quality (expert vs noisy, via action variance)
    """

    def __init__(self, k_neighbors: int = 5) -> None:
        self.k = k_neighbors

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate_quality(self, episodes: list[dict]) -> QualityMetrics:
        """Score each episode's data quality.

        Each episode dict must have:
          - ``'states'``: ``np.ndarray`` of shape ``(T, state_dim)``
          - ``'actions'``: ``np.ndarray`` of shape ``(T, action_dim)``
          - ``'episode_id'``: ``int``
        """
        if not episodes:
            raise ValueError("No episodes provided")

        # Collect all states/actions and build episode boundaries
        all_states_raw: list[np.ndarray] = []
        all_actions_raw: list[np.ndarray] = []
        episode_masks: list[np.ndarray] = []
        episode_ids: list[int] = []
        valid_episodes: list[dict] = []

        offset = 0
        for ep in episodes:
            states = np.asarray(ep["states"], dtype=np.float64)
            actions = np.asarray(ep["actions"], dtype=np.float64)
            T = len(states)
            if T < 2:
                logger.warning(
                    "Episode %s has %d timesteps (< 2), assigning score 0",
                    ep["episode_id"],
                    T,
                )
                episode_ids.append(ep["episode_id"])
                episode_masks.append(np.zeros(0, dtype=bool))
                valid_episodes.append(ep)
                continue

            all_states_raw.append(states)
            all_actions_raw.append(actions)

            mask = np.zeros(offset + T, dtype=bool)
            mask[offset : offset + T] = True
            episode_masks.append(mask)
            episode_ids.append(ep["episode_id"])
            valid_episodes.append(ep)
            offset += T

        if offset == 0:
            # All episodes too short
            scores = {eid: 0.0 for eid in episode_ids}
            return QualityMetrics(
                episode_scores=scores,
                aggregate_score=0.0,
                low_quality_episodes=list(episode_ids),
                mutual_information_estimate=0.0,
                signal_breakdown=QualitySignalBreakdown(
                    mutual_information=0.0,
                    action_smoothness=0.0,
                    episode_completion=0.0,
                    observation_consistency=0.0,
                    demonstration_quality=0.0,
                ),
            )

        all_states = np.vstack(all_states_raw)
        all_actions = np.vstack(all_actions_raw)

        # Normalize
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s_scaler = StandardScaler()
            a_scaler = StandardScaler()
            all_states_n = s_scaler.fit_transform(all_states)
            all_actions_n = a_scaler.fit_transform(all_actions)

        # Replace NaN from constant columns with 0
        all_states_n = np.nan_to_num(all_states_n, nan=0.0)
        all_actions_n = np.nan_to_num(all_actions_n, nan=0.0)

        # Global MI
        global_mi = self._knn_mutual_information(all_states_n, all_actions_n, self.k)

        # Pad masks to final length
        n_total = len(all_states)
        for i in range(len(episode_masks)):
            if len(episode_masks[i]) < n_total:
                episode_masks[i] = np.pad(
                    episode_masks[i],
                    (0, n_total - len(episode_masks[i])),
                    constant_values=False,
                )
            elif len(episode_masks[i]) > n_total:
                episode_masks[i] = episode_masks[i][:n_total]

        # --- Pass 1: MI contributions (existing logic) ---
        mi_raw: dict[int, float] = {}
        for idx, eid in enumerate(episode_ids):
            mask = episode_masks[idx]
            if mask.sum() == 0:
                mi_raw[eid] = 0.0
                continue
            contribution = self._per_episode_contribution(all_states_n, all_actions_n, mask)
            mi_raw[eid] = contribution

        # Normalize MI to [0, 1]
        mi_vals = list(mi_raw.values())
        mi_min, mi_max = min(mi_vals), max(mi_vals)
        if mi_max > mi_min:
            mi_norm = {eid: (v - mi_min) / (mi_max - mi_min) for eid, v in mi_raw.items()}
        elif len(episodes) == 1:
            mi_norm = {eid: 1.0 for eid in mi_raw}
        else:
            mi_norm = {eid: 0.5 for eid in mi_raw}

        # --- Pass 2: Compute new quality signals per episode ---
        smoothness_scores: dict[int, float] = {}
        completion_scores: dict[int, float] = {}
        consistency_scores: dict[int, float] = {}
        demo_quality_scores: dict[int, float] = {}

        for ep in valid_episodes:
            eid = ep["episode_id"]
            states = np.asarray(ep["states"], dtype=np.float64)
            actions = np.asarray(ep["actions"], dtype=np.float64)

            if len(states) < 2:
                smoothness_scores[eid] = 0.0
                completion_scores[eid] = 0.0
                consistency_scores[eid] = 0.0
                demo_quality_scores[eid] = 0.0
                continue

            smoothness_scores[eid] = self._action_smoothness(actions)
            completion_scores[eid] = self._episode_completion(states)
            consistency_scores[eid] = self._observation_consistency(states)
            demo_quality_scores[eid] = self._demonstration_quality(actions)

        # --- Blend into composite per-episode score ---
        composite_scores: dict[int, float] = {}
        for eid in episode_ids:
            composite = (
                0.30 * mi_norm.get(eid, 0.0)
                + 0.25 * smoothness_scores.get(eid, 0.0)
                + 0.15 * completion_scores.get(eid, 0.0)
                + 0.15 * consistency_scores.get(eid, 0.0)
                + 0.15 * demo_quality_scores.get(eid, 0.0)
            )
            composite_scores[eid] = float(np.clip(composite, 0.0, 1.0))

        # Identify low quality
        score_vals = np.array(list(composite_scores.values()))
        if len(score_vals) > 1:
            q1, q3 = np.percentile(score_vals, [25, 75])
            iqr = q3 - q1
            threshold = max(0.3, float(np.median(score_vals) - 1.5 * iqr))
        else:
            threshold = 0.3
        low_quality = [eid for eid, s in composite_scores.items() if s < threshold]

        agg = float(np.mean(score_vals))

        # Dataset-wide signal averages for breakdown
        signal_breakdown = QualitySignalBreakdown(
            mutual_information=float(np.mean(list(mi_norm.values()))) if mi_norm else 0.0,
            action_smoothness=(
                float(np.mean(list(smoothness_scores.values()))) if smoothness_scores else 0.0
            ),
            episode_completion=(
                float(np.mean(list(completion_scores.values()))) if completion_scores else 0.0
            ),
            observation_consistency=(
                float(np.mean(list(consistency_scores.values()))) if consistency_scores else 0.0
            ),
            demonstration_quality=(
                float(np.mean(list(demo_quality_scores.values()))) if demo_quality_scores else 0.0
            ),
        )

        return QualityMetrics(
            episode_scores=composite_scores,
            aggregate_score=agg,
            low_quality_episodes=low_quality,
            mutual_information_estimate=global_mi,
            signal_breakdown=signal_breakdown,
        )

    # ------------------------------------------------------------------
    # Quality signal: Action smoothness
    # ------------------------------------------------------------------

    @staticmethod
    def _action_smoothness(actions: np.ndarray) -> float:
        """Score action trajectory smoothness via jerk (3rd derivative).

        Smooth expert demonstrations have low jerk. Returns 1.0 for smooth,
        0.0 for very jerky trajectories.
        """
        if len(actions) < 4:
            return 0.5  # not enough data to compute jerk

        # Compute derivatives along time axis
        velocity = np.diff(actions, axis=0)
        acceleration = np.diff(velocity, axis=0)
        jerk = np.diff(acceleration, axis=0)

        # Mean absolute jerk, normalized by action scale
        action_scale = np.std(actions) + 1e-8
        normalized_jerk = np.mean(np.abs(jerk)) / action_scale

        # Map to [0, 1]: jerk of 2.0 or above → 0.0, jerk of 0 → 1.0
        return float(np.clip(1.0 - normalized_jerk / 2.0, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Quality signal: Episode completion
    # ------------------------------------------------------------------

    @staticmethod
    def _episode_completion(states: np.ndarray) -> float:
        """Score whether an episode converges toward a goal state.

        Compares state variance in the last 20% vs first 20% of the
        trajectory. Decreasing variance suggests goal convergence.
        """
        T = len(states)
        if T < 10:
            return 0.5  # not enough data

        split = max(2, T // 5)  # 20% of trajectory
        early_var = np.mean(np.var(states[:split], axis=0)) + 1e-8
        late_var = np.mean(np.var(states[-split:], axis=0)) + 1e-8

        # Ratio < 1 means converging (good), > 1 means diverging (bad)
        ratio = late_var / early_var

        # Sigmoid-like mapping: ratio=0.5 → ~0.73, ratio=1.0 → 0.5, ratio=2.0 → ~0.27
        score = 1.0 / (1.0 + ratio)
        return float(np.clip(score, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Quality signal: Observation consistency
    # ------------------------------------------------------------------

    @staticmethod
    def _observation_consistency(states: np.ndarray) -> float:
        """Score observation/sensor data quality.

        Detects NaN/inf values, sudden jumps (>3 std from mean), and
        constant runs (stuck sensors). Returns fraction of clean timesteps.
        """
        T = len(states)
        if T < 2:
            return 0.5

        clean = np.ones(T, dtype=bool)

        # Check NaN/inf
        nan_mask = ~np.isfinite(states).all(axis=1)
        clean[nan_mask] = False

        # Check sudden jumps (> 3 std deviation)
        diffs = np.diff(states, axis=0)
        diff_std = np.std(diffs, axis=0) + 1e-8
        jumps = np.any(np.abs(diffs) > 3 * diff_std, axis=1)
        # Mark both the frame before and after the jump
        clean[:-1][jumps] = False
        clean[1:][jumps] = False

        return float(np.mean(clean))

    # ------------------------------------------------------------------
    # Quality signal: Demonstration quality
    # ------------------------------------------------------------------

    @staticmethod
    def _demonstration_quality(actions: np.ndarray) -> float:
        """Score whether actions look like expert demonstrations.

        Expert demos have moderate, consistent per-dimension action variance.
        Random/noisy demos have very high variance.
        Constant/degenerate demos have near-zero variance.
        Uses an inverted-U: moderate variance scores highest.
        """
        if len(actions) < 3:
            return 0.5

        per_dim_var = np.var(actions, axis=0)
        mean_var = np.mean(per_dim_var)

        if mean_var < 1e-10:
            return 0.1  # constant actions — degenerate

        # Coefficient of variation of per-dimension variances
        # Low CV = consistent variance across dims (expert-like)
        cv = np.std(per_dim_var) / (mean_var + 1e-8)
        consistency = float(np.clip(1.0 - cv / 3.0, 0.0, 1.0))

        # Inverted-U for variance magnitude:
        # Very low variance (< 0.01) or very high (> 10) → bad
        # Moderate variance (around 0.1 - 2.0) → good
        log_var = np.log10(mean_var + 1e-10)
        # Peak at log_var ≈ -0.5 (variance ≈ 0.3), width ~2.0
        magnitude = float(np.exp(-0.5 * ((log_var + 0.5) / 1.0) ** 2))

        return float(np.clip(0.5 * consistency + 0.5 * magnitude, 0.0, 1.0))

    # ------------------------------------------------------------------
    # KSG Mutual Information Estimator
    # ------------------------------------------------------------------

    def _knn_mutual_information(self, X: np.ndarray, Y: np.ndarray, k: int = 5) -> float:
        """Estimate MI(X; Y) using the KSG estimator.

        Uses Chebyshev (max-norm) distance as in the original KSG paper.
        """
        N = len(X)
        if N < 2 * k + 1:
            logger.warning("Too few samples (%d) for k=%d; returning MI=0", N, k)
            return 0.0

        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        Z = np.hstack([X, Y])

        # Find k-th neighbor distance in joint space
        nn_z = NearestNeighbors(n_neighbors=k + 1, metric="chebyshev", algorithm="ball_tree")
        nn_z.fit(Z)
        dists_z, _ = nn_z.kneighbors(Z)
        eps = dists_z[:, k]  # distance to k-th neighbor (0-indexed: k-th col)

        # Count neighbors within eps in marginal spaces
        nn_x = NearestNeighbors(metric="chebyshev", algorithm="ball_tree")
        nn_x.fit(X)
        nn_y = NearestNeighbors(metric="chebyshev", algorithm="ball_tree")
        nn_y.fit(Y)

        n_x = np.zeros(N)
        n_y = np.zeros(N)
        for i in range(N):
            # radius_neighbors with radius=eps[i], excluding self
            r = eps[i]
            if r < 1e-15:
                r = 1e-15
            idx_x = nn_x.radius_neighbors(X[i : i + 1], radius=r, return_distance=False)[0]
            idx_y = nn_y.radius_neighbors(Y[i : i + 1], radius=r, return_distance=False)[0]
            n_x[i] = len(idx_x) - 1  # exclude self
            n_y[i] = len(idx_y) - 1

        # KSG formula
        mi = digamma(k) - np.mean(digamma(n_x + 1) + digamma(n_y + 1)) + digamma(N)
        return max(0.0, float(mi))

    # ------------------------------------------------------------------
    # Per-episode contribution
    # ------------------------------------------------------------------

    def _per_episode_contribution(
        self,
        all_states: np.ndarray,
        all_actions: np.ndarray,
        episode_mask: np.ndarray,
    ) -> float:
        """Compute how much one episode contributes to overall MI.

        Leave-one-out: MI(full) - MI(full minus this episode).
        """
        remaining = ~episode_mask
        if remaining.sum() < 2 * self.k + 1:
            # Not enough data to estimate MI without this episode
            return self._knn_mutual_information(all_states, all_actions, self.k)

        mi_full = self._knn_mutual_information(all_states, all_actions, self.k)
        mi_without = self._knn_mutual_information(
            all_states[remaining], all_actions[remaining], self.k
        )
        return mi_full - mi_without
