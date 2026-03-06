"""Data quality metrics (mutual information)."""

from __future__ import annotations

import logging
import warnings

import numpy as np
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from orbit.profile.types import QualityMetrics

logger = logging.getLogger(__name__)


class QualityEstimator:
    """Score episode data quality using k-NN mutual information estimation.

    Implements the KSG (Kraskov-Stögbauer-Grassberger) estimator for
    mutual information between states and actions, inspired by the DemInf
    approach (Hejna et al., RSS 2025).
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

        # Per-episode contribution
        raw_scores: dict[int, float] = {}
        for idx, eid in enumerate(episode_ids):
            mask = episode_masks[idx]
            if mask.sum() == 0:
                raw_scores[eid] = 0.0
                continue
            contribution = self._per_episode_contribution(all_states_n, all_actions_n, mask)
            raw_scores[eid] = contribution

        # Normalize to [0, 1]
        vals = list(raw_scores.values())
        vmin, vmax = min(vals), max(vals)
        if vmax > vmin:
            scores = {eid: (v - vmin) / (vmax - vmin) for eid, v in raw_scores.items()}
        elif len(episodes) == 1:
            scores = {eid: 1.0 for eid in raw_scores}
        else:
            scores = {eid: 0.5 for eid in raw_scores}

        # Identify low quality
        score_vals = np.array(list(scores.values()))
        if len(score_vals) > 1:
            q1, q3 = np.percentile(score_vals, [25, 75])
            iqr = q3 - q1
            threshold = max(0.3, float(np.median(score_vals) - 1.5 * iqr))
        else:
            threshold = 0.3
        low_quality = [eid for eid, s in scores.items() if s < threshold]

        agg = float(np.mean(score_vals))

        return QualityMetrics(
            episode_scores=scores,
            aggregate_score=agg,
            low_quality_episodes=low_quality,
            mutual_information_estimate=global_mi,
        )

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
