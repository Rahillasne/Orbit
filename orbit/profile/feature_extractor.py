"""Fixed-size feature extraction from profiled datasets.

Converts a :class:`DatasetProfile` (and optional :class:`DatasetReportCard`)
into a 64-dimensional float32 feature vector suitable for training a
downstream predictor model.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import numpy as np

from orbit.profile.types import (
    ActionStats,
    DatasetProfile,
    DatasetReportCard,
    EmbeddingStats,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature names (human-readable, one per dimension)
# ---------------------------------------------------------------------------

_EMBEDDING_NAMES = [
    "emb_mean_norm",
    "emb_std_norm",
    "emb_mean_pairwise_cosine",
    "emb_std_pairwise_cosine",
    "emb_min_pairwise_cosine",
    "emb_max_pairwise_cosine",
    "emb_num_clusters",
    "emb_noise_ratio",
    "emb_silhouette_score",
    "emb_calinski_harabasz",
    "emb_convex_hull_volume",
    "emb_effective_dimensionality",
    "emb_mean_sequential_distance",
    "emb_std_sequential_distance",
    "emb_mean_episode_trajectory_length",
    "emb_std_episode_trajectory_length",
    "emb_skewness_pc1",
    "emb_kurtosis_pc1",
    "emb_entropy_estimate",
    "emb_uniformity_score",
]

_ACTION_NAMES = [
    "act_dimensionality",
    "act_mean_magnitude",
    "act_std_magnitude",
    "act_smoothness",
    "act_range_utilization",
    "act_num_modes",
    "act_mean_episode_entropy",
    "act_cross_episode_consistency",
    "act_effective_dimensionality",
    "act_mean_autocorrelation",
    "act_zero_fraction",
    "act_boundary_fraction",
]

_QUALITY_NAMES = [
    "qual_aggregate_score",
    "qual_action_smoothness",
    "qual_episode_completion",
    "qual_observation_consistency",
    "qual_demonstration_quality",
    "qual_frame_brightness_mean",
    "qual_frame_brightness_std",
    "qual_frame_blur_score",
    "qual_temporal_consistency",
    "qual_reward_signal_present",
    "qual_language_annotation_present",
    "qual_multi_camera",
]

_SCALE_NAMES = [
    "scale_log_episodes",
    "scale_log_frames",
    "scale_avg_episode_length",
    "scale_std_episode_length",
    "scale_fps",
    "scale_image_resolution_pixels",
    "scale_observation_dims",
    "scale_dataset_size_mb",
]

_TASK_NAMES = [
    "task_primary_score",
    "task_primary_visual_relevance",
    "task_primary_coverage_diversity",
    "task_primary_data_quality",
    "task_coverage_score",
    "task_quality_score",
    "task_diversity_score",
    "task_volume_score",
    "task_interaction_emb_x_act",
    "task_interaction_clusters_per_ep",
    "task_interaction_hull_x_quality",
    "task_interaction_uniform_x_diversity",
]

# ---------------------------------------------------------------------------
# NEW feature groups for breaking the rho=0.75 ceiling (Phase 7)
# ---------------------------------------------------------------------------

_TEMPORAL_NAMES = [
    "temp_state_autocorrelation",       # do states follow smooth trajectories?
    "temp_coverage_rate",               # how quickly does dataset explore new regions?
    "temp_action_temporal_entropy",      # is action sequence predictable or chaotic over time?
]

_CROSS_EPISODE_NAMES = [
    "cross_inter_episode_overlap",      # do episodes visit the same states?
    "cross_episode_diversity_index",    # Shannon entropy over episode embedding centroids
]

_EMB_GEOMETRY_NAMES = [
    "geom_intrinsic_dimensionality",    # intrinsic dim of embedding manifold (MLE)
    "geom_isotropy",                    # are embeddings spread uniformly or clustered in cone?
    "geom_hub_score",                   # fraction of hub points (NN of many others)
]

_ADVANCED_INTERACTION_NAMES = [
    "adv_coverage_action_ratio",        # embedding coverage / action coverage ratio
]

# Reduced feature set: embedding (20) + quality (12) + scale (8) + task (12) = 52 dims
# Ablation showed removing action features improves rho from 0.666 to 0.704
REDUCED_FEATURE_NAMES: list[str] = (
    _EMBEDDING_NAMES + _QUALITY_NAMES + _SCALE_NAMES + _TASK_NAMES
)

# Extended feature set: 64 base + 9 new = 73 dims
EXTENDED_FEATURE_NAMES: list[str] = (
    _EMBEDDING_NAMES + _ACTION_NAMES + _QUALITY_NAMES + _SCALE_NAMES + _TASK_NAMES
    + _TEMPORAL_NAMES + _CROSS_EPISODE_NAMES + _EMB_GEOMETRY_NAMES
    + _ADVANCED_INTERACTION_NAMES
)

# Max embeddings to reconstruct for expensive stats
_MAX_EMBEDDINGS = 2000
# Max pairwise cosine pairs to sample
_MAX_PAIRS = 500
# Max actions for GMM mode detection
_MAX_ACTIONS_GMM = 1000


# ===================================================================
# FeatureScaler — StandardScaler with JSON persistence
# ===================================================================


class FeatureScaler:
    """StandardScaler wrapper with JSON-based save/load.

    Wraps :class:`sklearn.preprocessing.StandardScaler` and serialises
    fitted parameters (mean and scale) to a JSON file for portability.
    """

    def __init__(self) -> None:
        from sklearn.preprocessing import StandardScaler

        self._scaler = StandardScaler()
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, features: np.ndarray) -> FeatureScaler:
        """Fit the scaler on a (N, D) feature matrix."""
        self._scaler.fit(features)
        self._fitted = True
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Apply standardisation.  Raises if not yet fitted."""
        if not self._fitted:
            raise ValueError("FeatureScaler has not been fitted yet. Call fit() first.")
        return self._scaler.transform(features).astype(np.float32)

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit then transform in one step."""
        self.fit(features)
        return self.transform(features)

    def save(self, path: str | Path) -> None:
        """Save fitted parameters to a JSON file."""
        if not self._fitted:
            raise ValueError("Cannot save an unfitted scaler.")
        data = {
            "mean": self._scaler.mean_.tolist(),
            "scale": self._scaler.scale_.tolist(),
            "var": self._scaler.var_.tolist(),
            "n_features_in": int(self._scaler.n_features_in_),
            "n_samples_seen": int(self._scaler.n_samples_seen_),
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> FeatureScaler:
        """Load a fitted scaler from a JSON file."""
        data = json.loads(Path(path).read_text())
        scaler = cls()
        scaler._scaler.mean_ = np.array(data["mean"], dtype=np.float64)
        scaler._scaler.scale_ = np.array(data["scale"], dtype=np.float64)
        scaler._scaler.var_ = np.array(data["var"], dtype=np.float64)
        scaler._scaler.n_features_in_ = data["n_features_in"]
        scaler._scaler.n_samples_seen_ = data["n_samples_seen"]
        scaler._fitted = True
        return scaler


# ===================================================================
# DatasetFeatureExtractor
# ===================================================================


class DatasetFeatureExtractor:
    """Extract a fixed-size feature vector from a :class:`DatasetProfile`.

    Feature groups (base 64 dims):
        * Embedding distribution (20 dims)
        * Action space (12 dims)
        * Quality (12 dims)
        * Scale (8 dims)
        * Task relevance (12 dims)

    Extended features (+9 dims = 73 total):
        * Temporal (3 dims)
        * Cross-episode (2 dims)
        * Embedding geometry (3 dims)
        * Advanced interactions (1 dim)

    Parameters
    ----------
    scaler:
        Optional feature scaler.
    extended:
        If True, extract extended features (73 dims) in addition to
        the base 64.  Default False for backward compatibility.
    """

    FEATURE_DIM = 64
    EXTENDED_FEATURE_DIM = 73
    FEATURE_NAMES: list[str] = (
        _EMBEDDING_NAMES + _ACTION_NAMES + _QUALITY_NAMES + _SCALE_NAMES + _TASK_NAMES
    )

    def __init__(
        self,
        scaler: FeatureScaler | None = None,
        extended: bool = False,
    ) -> None:
        self._scaler = scaler
        self.extended = extended

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        profile: DatasetProfile,
        report_card: DatasetReportCard | None = None,
        episodes: list[dict] | None = None,
        metadata: dict | None = None,
    ) -> np.ndarray:
        """Return a ``(64,)`` float32 feature vector.

        Parameters
        ----------
        profile:
            A fully-populated :class:`DatasetProfile`.
        report_card:
            Optional :class:`DatasetReportCard` for task-relevance features.
        episodes:
            Optional raw episode dicts (with ``states``, ``actions``,
            ``episode_id`` keys) for action-space features.
        metadata:
            Optional dict with keys like ``fps``, ``image_resolution_pixels``,
            ``observation_dims``, ``dataset_size_mb``,
            ``frame_brightness_mean``, etc.
        """
        metadata = metadata or {}

        emb_feats = self._embedding_features(profile)
        act_feats = self._action_features(profile, episodes)
        qual_feats = self._quality_features(profile, metadata)
        scale_feats = self._scale_features(profile, metadata)
        task_feats = self._task_features(profile, report_card, emb_feats, act_feats, qual_feats)

        base_features = np.concatenate([emb_feats, act_feats, qual_feats, scale_feats, task_feats])
        assert len(base_features) == self.FEATURE_DIM, (
            f"Expected {self.FEATURE_DIM} features, got {len(base_features)}"
        )

        if self.extended:
            ext_feats = self._extended_features(profile, episodes, emb_feats, act_feats)
            features = np.concatenate([base_features, ext_feats])
        else:
            features = base_features

        # Replace any non-finite values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32
        )

        if self._scaler is not None:
            features = self._scaler.transform(features.reshape(1, -1))[0]

        return features

    def extract_from_metadata(self, estimated_features: dict) -> np.ndarray:
        """Build a 64-dim vector from estimated metadata (no real profile needed).

        Used for non-downloadable datasets in ground truth where only high-level
        metadata is available (num_episodes, action_dims, quality/diversity
        estimates, etc.).

        Parameters
        ----------
        estimated_features:
            Dict with optional keys: ``num_episodes``, ``avg_episode_length``,
            ``action_dims``, ``image_resolution``, ``diversity_estimate``,
            ``quality_estimate``, ``task_complexity``, ``num_objects``,
            ``bimanual``, ``demo_type``, ``horizon_length``,
            ``observation_type``.
        """
        _QUALITY_MAP = {"high": 0.8, "medium": 0.5, "low": 0.2}
        _COMPLEXITY_MAP = {"simple": 0.2, "medium": 0.5, "hard": 0.9}
        _HORIZON_MAP = {"short": 0.3, "medium": 0.6, "long": 1.0}
        _DEMO_TYPE_MAP = {
            "expert": 0.95, "proficient_human": 0.85, "scripted": 0.9,
            "multi_human": 0.55, "medium": 0.45, "medium_replay": 0.35,
            "random": 0.1,
        }

        num_episodes = estimated_features.get("num_episodes", 0)
        avg_ep_len = estimated_features.get("avg_episode_length", 0)
        action_dims = estimated_features.get("action_dims", 0)
        img_res = estimated_features.get("image_resolution")
        diversity_str = estimated_features.get("diversity_estimate", "medium")
        quality_str = estimated_features.get("quality_estimate", "medium")
        complexity_str = estimated_features.get("task_complexity", "medium")
        bimanual = estimated_features.get("bimanual", False)
        demo_type = estimated_features.get("demo_type", "proficient_human")
        obs_type = estimated_features.get("observation_type", "image")

        diversity_val = _QUALITY_MAP.get(diversity_str, 0.5)
        quality_val = _QUALITY_MAP.get(quality_str, 0.5)
        complexity_val = _COMPLEXITY_MAP.get(complexity_str, 0.5)
        demo_quality = _DEMO_TYPE_MAP.get(demo_type, 0.5)

        img_pixels = 0.0
        if img_res and isinstance(img_res, (list, tuple)) and len(img_res) >= 2:
            img_pixels = float(img_res[0] * img_res[1])

        num_frames = num_episodes * avg_ep_len
        has_images = obs_type in ("image", "both")

        vec = np.zeros(self.FEATURE_DIM, dtype=np.float32)

        # Embedding features (0-19): synthetic estimates based on metadata
        # These won't match real embeddings but provide differentiation
        vec[0] = 0.5 * demo_quality  # emb_mean_norm: higher for expert data
        vec[1] = 0.3 * (1.0 - demo_quality)  # emb_std_norm: more variance for noisy data
        vec[2] = 0.6 * demo_quality  # emb_mean_pairwise_cosine: expert=more similar
        vec[3] = 0.2 * diversity_val  # emb_std_pairwise_cosine
        vec[6] = 3.0 * diversity_val  # emb_num_clusters
        vec[7] = 0.1 * (1.0 - demo_quality)  # emb_noise_ratio
        vec[8] = 0.4 * demo_quality  # emb_silhouette_score
        vec[10] = 0.3 * diversity_val  # emb_convex_hull_volume
        vec[11] = min(action_dims, 20) / 20.0  # emb_effective_dimensionality
        vec[16] = 0.5 * complexity_val  # emb_skewness_pc1
        vec[18] = 3.0 + 2.0 * diversity_val  # emb_entropy_estimate
        vec[19] = 0.3 * diversity_val  # emb_uniformity_score

        # Action features (20-31)
        vec[20] = float(action_dims)  # dimensionality
        vec[21] = 0.5 * (1.0 - complexity_val)  # act_mean_magnitude
        vec[22] = 0.3 * diversity_val  # act_std_magnitude
        vec[23] = demo_quality  # act_smoothness: expert demos are smoother
        vec[24] = 0.5 + 0.3 * demo_quality  # act_range_utilization
        vec[25] = 1.0 + 2.0 * diversity_val  # act_num_modes
        vec[26] = 2.0 + complexity_val  # act_mean_episode_entropy
        vec[27] = 0.7 * demo_quality  # act_cross_episode_consistency
        vec[28] = min(action_dims, 10)  # act_dimensionality_effective
        vec[29] = 0.5 * demo_quality  # act_mean_autocorrelation
        vec[30] = 0.05 * (1.0 - demo_quality)  # act_zero_fraction
        vec[31] = 0.02 * complexity_val  # act_boundary_fraction

        # Quality features (32-43)
        vec[32] = demo_quality  # qual_aggregate_score
        vec[33] = demo_quality  # qual_action_smoothness
        vec[34] = 0.7 + 0.3 * demo_quality  # qual_episode_completion
        vec[35] = 0.8 + 0.2 * demo_quality  # qual_observation_consistency
        vec[36] = demo_quality  # qual_demonstration_quality
        vec[37] = 0.5 if has_images else 0.0  # qual_frame_brightness_mean
        vec[38] = 0.1 if has_images else 0.0  # qual_frame_brightness_std
        vec[39] = 0.3 if has_images else 0.0  # qual_frame_blur_score
        vec[40] = 0.7 * demo_quality  # qual_temporal_consistency
        vec[42] = 1.0 if obs_type == "both" else 0.0  # qual_language_annotation
        vec[43] = 1.0 if bimanual else 0.0  # qual_multi_camera (proxy)

        # Scale features (44-51)
        vec[44] = np.log1p(num_episodes)
        vec[45] = np.log1p(num_frames)
        vec[46] = float(avg_ep_len)
        vec[47] = float(avg_ep_len) * 0.2  # std_episode_length estimate
        vec[49] = img_pixels
        vec[50] = float(action_dims)  # observation_dims proxy

        # Task features (52-63)
        vec[52] = demo_quality * (1.0 - complexity_val * 0.3)  # task_primary_score
        vec[53] = 0.5 if has_images else 0.0  # task_primary_visual_relevance
        vec[54] = diversity_val  # task_primary_coverage_diversity
        vec[55] = quality_val  # task_primary_data_quality
        vec[56] = diversity_val  # coverage_score
        vec[57] = quality_val  # quality_score
        vec[58] = diversity_val  # diversity_score
        vec[59] = min(np.log1p(num_episodes) / 10.0, 1.0)  # volume_score
        # Interaction features
        vec[60] = vec[0] * vec[23]  # emb_norm * action_smoothness
        vec[61] = vec[6] / max(vec[32], 0.1)  # clusters / quality
        vec[62] = vec[10] * quality_val  # hull_volume * quality
        vec[63] = vec[19] * diversity_val  # uniformity * diversity

        return vec

    def extract_batch(
        self,
        profiles: list[DatasetProfile],
        report_cards: list[DatasetReportCard | None] | None = None,
        episodes_list: list[list[dict] | None] | None = None,
        metadata_list: list[dict | None] | None = None,
    ) -> np.ndarray:
        """Return a ``(N, 64)`` float32 feature matrix."""
        n = len(profiles)
        rcs = report_cards or [None] * n
        eps = episodes_list or [None] * n
        mds = metadata_list or [None] * n
        rows = [self.extract(p, rc, ep, md) for p, rc, ep, md in zip(profiles, rcs, eps, mds)]
        return np.stack(rows)

    # ------------------------------------------------------------------
    # Group 1: Embedding distribution features (20 dims)
    # ------------------------------------------------------------------

    def _embedding_features(self, profile: DatasetProfile) -> np.ndarray:
        stats = self._ensure_embedding_stats(profile)
        return np.array(
            [
                stats.mean_norm,
                stats.std_norm,
                stats.mean_pairwise_cosine,
                stats.std_pairwise_cosine,
                stats.min_pairwise_cosine,
                stats.max_pairwise_cosine,
                stats.num_clusters,
                stats.noise_ratio,
                stats.silhouette_score,
                stats.calinski_harabasz,
                stats.convex_hull_volume,
                stats.effective_dimensionality,
                stats.mean_sequential_distance,
                stats.std_sequential_distance,
                stats.mean_episode_trajectory_length,
                stats.std_episode_trajectory_length,
                stats.skewness_pc1,
                stats.kurtosis_pc1,
                stats.entropy_estimate,
                stats.uniformity_score,
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Group 2: Action space features (12 dims)
    # ------------------------------------------------------------------

    def _action_features(
        self, profile: DatasetProfile, episodes: list[dict] | None
    ) -> np.ndarray:
        stats = self._ensure_action_stats(profile, episodes)
        return np.array(
            [
                stats.dimensionality,
                stats.mean_magnitude,
                stats.std_magnitude,
                stats.smoothness,
                stats.action_range_utilization,
                stats.num_action_modes,
                stats.mean_episode_action_entropy,
                stats.cross_episode_action_consistency,
                stats.action_dimensionality_effective,
                stats.mean_action_autocorrelation,
                stats.zero_action_fraction,
                stats.action_boundary_fraction,
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Group 3: Quality features (12 dims)
    # ------------------------------------------------------------------

    def _quality_features(self, profile: DatasetProfile, metadata: dict) -> np.ndarray:
        q = profile.quality
        sb = q.signal_breakdown
        return np.array(
            [
                q.aggregate_score,
                sb.action_smoothness if sb else 0.0,
                sb.episode_completion if sb else 0.0,
                sb.observation_consistency if sb else 0.0,
                sb.demonstration_quality if sb else 0.0,
                metadata.get("frame_brightness_mean", 0.0),
                metadata.get("frame_brightness_std", 0.0),
                metadata.get("frame_blur_score", 0.0),
                metadata.get("temporal_consistency", 0.0),
                float(metadata.get("reward_signal_present", 0)),
                float(metadata.get("language_annotation_present", 0)),
                float(metadata.get("multi_camera", 0)),
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Group 4: Scale features (8 dims)
    # ------------------------------------------------------------------

    def _scale_features(self, profile: DatasetProfile, metadata: dict) -> np.ndarray:
        avg_ep_len = profile.num_frames / max(profile.num_episodes, 1)
        std_ep_len = 0.0
        if profile.action_stats is not None:
            # Use episode length variability from action stats computation
            std_ep_len = 0.0  # will be overwritten below if episodes were given
        # Compute std from embedding index episode_ids if possible
        if profile.embedding_index and profile.embedding_index.episode_ids:
            ep_ids = np.array(profile.embedding_index.episode_ids)
            unique_eps = np.unique(ep_ids)
            if len(unique_eps) > 1:
                counts = np.array([np.sum(ep_ids == eid) for eid in unique_eps])
                std_ep_len = float(np.std(counts))

        return np.array(
            [
                np.log1p(profile.num_episodes),
                np.log1p(profile.num_frames),
                avg_ep_len,
                std_ep_len,
                float(metadata.get("fps", 0.0)),
                float(metadata.get("image_resolution_pixels", 0.0)),
                float(metadata.get("observation_dims", 0.0)),
                float(metadata.get("dataset_size_mb", 0.0)),
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Group 5: Task relevance features (12 dims)
    # ------------------------------------------------------------------

    def _task_features(
        self,
        profile: DatasetProfile,
        report_card: DatasetReportCard | None,
        emb_feats: np.ndarray,
        act_feats: np.ndarray,
        qual_feats: np.ndarray,
    ) -> np.ndarray:
        # Primary task assessment (first 4 dims)
        primary = np.zeros(4, dtype=np.float32)
        if report_card and report_card.task_assessments:
            ta = report_card.task_assessments[0]
            primary[0] = ta.score
            # Map from CapabilityScore.score_breakdown if available
            cap = _find_capability(profile, ta.task)
            if cap and cap.score_breakdown:
                primary[1] = cap.score_breakdown.visual_relevance
                primary[2] = cap.score_breakdown.coverage_diversity
                primary[3] = cap.score_breakdown.data_quality
        elif profile.capabilities:
            cap = profile.capabilities[0]
            primary[0] = cap.score
            if cap.score_breakdown:
                primary[1] = cap.score_breakdown.visual_relevance
                primary[2] = cap.score_breakdown.coverage_diversity
                primary[3] = cap.score_breakdown.data_quality

        # Report card scores (next 4 dims)
        rc_scores = np.zeros(4, dtype=np.float32)
        if report_card:
            rc_scores[0] = report_card.coverage_score
            rc_scores[1] = report_card.quality_score
            rc_scores[2] = report_card.diversity_score
            rc_scores[3] = report_card.volume_score
        else:
            # Fallback to profile-level
            rc_scores[0] = profile.coverage.overall_coverage_score
            rc_scores[1] = profile.quality.aggregate_score

        # Interaction features (last 4 dims)
        interactions = np.array(
            [
                emb_feats[0] * act_feats[3],  # emb_mean_norm * act_smoothness
                emb_feats[6] / max(qual_feats[0], 1e-8),  # num_clusters / aggregate_quality
                emb_feats[10] * qual_feats[0],  # hull_volume * quality
                emb_feats[19] * rc_scores[2],  # uniformity * diversity
            ],
            dtype=np.float32,
        )

        return np.concatenate([primary, rc_scores, interactions])

    # ------------------------------------------------------------------
    # Extended features (Phase 7 — breaking the 0.75 ceiling)
    # ------------------------------------------------------------------

    def _extended_features(
        self,
        profile: DatasetProfile,
        episodes: list[dict] | None,
        emb_feats: np.ndarray,
        act_feats: np.ndarray,
    ) -> np.ndarray:
        """Compute 9 extended features: temporal, cross-episode, geometry, interactions."""
        temporal = self._temporal_features(profile, episodes)
        cross_ep = self._cross_episode_features(profile)
        geometry = self._embedding_geometry_features(profile)
        adv_interaction = self._advanced_interaction_features(
            profile, emb_feats, act_feats
        )
        ext = np.concatenate([temporal, cross_ep, geometry, adv_interaction])
        assert len(ext) == 9, f"Expected 9 extended features, got {len(ext)}"
        return ext

    def _temporal_features(
        self, profile: DatasetProfile, episodes: list[dict] | None
    ) -> np.ndarray:
        """3 temporal features: state autocorrelation, coverage rate, action temporal entropy."""
        state_autocorr = 0.0
        coverage_rate = 0.0
        action_temporal_entropy = 0.0

        idx = profile.embedding_index
        n = idx.num_embeddings
        ep_ids = idx.episode_ids
        frame_ids = idx.frame_indices

        # State autocorrelation from embeddings
        if ep_ids and frame_ids and n >= 4:
            try:
                embeddings = _reconstruct_embeddings(profile, max_n=min(n, _MAX_EMBEDDINGS))
                ep_arr = np.array(ep_ids[: len(embeddings)])
                frame_arr = np.array(frame_ids[: len(embeddings)])
                unique_eps = np.unique(ep_arr)

                autocorrs: list[float] = []
                seen_centroids: list[np.ndarray] = []

                for eid in unique_eps:
                    mask = ep_arr == eid
                    if mask.sum() < 4:
                        continue
                    ep_embs = embeddings[mask]
                    ep_frames = frame_arr[mask]
                    order = np.argsort(ep_frames)
                    ep_embs = ep_embs[order]

                    # Autocorrelation of embedding norms
                    norms = np.linalg.norm(ep_embs, axis=1)
                    if np.std(norms) > 1e-8:
                        centered = norms - np.mean(norms)
                        var = np.sum(centered**2)
                        if var > 1e-10:
                            ac = float(np.sum(centered[:-1] * centered[1:]) / var)
                            autocorrs.append(ac)

                    # Coverage rate: track cumulative unique centroid count
                    centroid = np.mean(ep_embs, axis=0)
                    seen_centroids.append(centroid)

                # Coverage rate: how quickly new clusters appear
                if len(seen_centroids) > 2:
                    centroids_arr = np.array(seen_centroids)
                    # Count distinct clusters among episode centroids
                    from sklearn.cluster import KMeans

                    n_check = min(8, len(centroids_arr))
                    if n_check >= 2:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            km = KMeans(n_clusters=n_check, n_init="auto", random_state=42)
                            labels = km.fit_predict(centroids_arr)
                        # Rate = fraction of labels seen in first half vs second half
                        half = len(labels) // 2
                        first_unique = len(set(labels[:half]))
                        total_unique = len(set(labels))
                        coverage_rate = float(first_unique / max(total_unique, 1))

                if autocorrs:
                    state_autocorr = float(np.mean(autocorrs))
            except Exception:
                pass

        # Action temporal entropy
        if episodes:
            ep_entropies: list[float] = []
            for ep in episodes:
                actions = np.asarray(ep.get("actions", []), dtype=np.float64)
                if actions.ndim < 2 or len(actions) < 4:
                    continue
                # Temporal entropy: entropy of sequential action differences
                diffs = np.diff(actions, axis=0)
                diff_mags = np.linalg.norm(diffs, axis=1)
                if np.std(diff_mags) < 1e-8:
                    continue
                hist, _ = np.histogram(
                    diff_mags, bins=min(15, len(diff_mags) // 2 + 1), density=True
                )
                hist = hist / (hist.sum() + 1e-10)
                ent = float(-np.sum(hist * np.log(hist + 1e-10)))
                ep_entropies.append(ent)
            if ep_entropies:
                action_temporal_entropy = float(np.mean(ep_entropies))

        return np.array(
            [state_autocorr, coverage_rate, action_temporal_entropy],
            dtype=np.float32,
        )

    def _cross_episode_features(self, profile: DatasetProfile) -> np.ndarray:
        """2 cross-episode features: inter-episode overlap, episode diversity index."""
        overlap = 0.0
        diversity_index = 0.0

        idx = profile.embedding_index
        n = idx.num_embeddings
        ep_ids = idx.episode_ids

        if ep_ids and n >= 4:
            try:
                embeddings = _reconstruct_embeddings(profile, max_n=min(n, _MAX_EMBEDDINGS))
                ep_arr = np.array(ep_ids[: len(embeddings)])
                unique_eps = np.unique(ep_arr)

                if len(unique_eps) >= 2:
                    # Episode centroids
                    centroids = []
                    for eid in unique_eps:
                        mask = ep_arr == eid
                        if mask.sum() > 0:
                            centroids.append(np.mean(embeddings[mask], axis=0))

                    centroids_arr = np.array(centroids)

                    # Inter-episode overlap: mean pairwise cosine similarity of centroids
                    norms = np.linalg.norm(centroids_arr, axis=1, keepdims=True)
                    normed = centroids_arr / np.maximum(norms, 1e-8)
                    cosines = normed @ normed.T
                    triu = np.triu_indices(len(centroids), k=1)
                    if len(triu[0]) > 0:
                        overlap = float(np.mean(cosines[triu]))

                    # Episode diversity index: Shannon entropy over centroid cluster assignments
                    n_clusters = min(max(2, len(centroids) // 3), 8)
                    if len(centroids) > n_clusters:
                        from sklearn.cluster import KMeans

                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            km = KMeans(
                                n_clusters=n_clusters, n_init="auto", random_state=42
                            )
                            labels = km.fit_predict(centroids_arr)
                        counts = np.bincount(labels, minlength=n_clusters).astype(
                            np.float64
                        )
                        probs = counts / counts.sum()
                        probs = probs[probs > 0]
                        entropy = float(-np.sum(probs * np.log(probs)))
                        max_entropy = np.log(n_clusters)
                        diversity_index = float(entropy / max(max_entropy, 1e-10))
            except Exception:
                pass

        return np.array([overlap, diversity_index], dtype=np.float32)

    def _embedding_geometry_features(self, profile: DatasetProfile) -> np.ndarray:
        """3 embedding geometry features: intrinsic dim, isotropy, hub score."""
        intrinsic_dim = 0.0
        isotropy = 0.0
        hub_score = 0.0

        idx = profile.embedding_index
        n = idx.num_embeddings

        if n >= 10:
            try:
                embeddings = _reconstruct_embeddings(profile, max_n=min(n, _MAX_EMBEDDINGS))

                # Intrinsic dimensionality via MLE (Levina & Bickel, 2004)
                k_nn = min(10, len(embeddings) - 1)
                if k_nn >= 2:
                    from sklearn.neighbors import NearestNeighbors

                    nn = NearestNeighbors(n_neighbors=k_nn + 1, metric="euclidean")
                    nn.fit(embeddings)
                    dists, _ = nn.kneighbors(embeddings)
                    # Skip self-distance (column 0)
                    dists = dists[:, 1:]
                    # MLE estimator
                    log_ratios = np.log(
                        dists[:, -1:] / np.maximum(dists[:, :-1], 1e-10)
                    )
                    mle_dims = (k_nn - 1) / np.sum(log_ratios, axis=1)
                    intrinsic_dim = float(np.median(mle_dims))
                    intrinsic_dim = min(intrinsic_dim, 100.0)  # cap outliers

                # Isotropy: variance of singular values (uniform = isotropic)
                centered = embeddings - embeddings.mean(axis=0)
                n_sv = min(50, *centered.shape)
                if n_sv >= 2:
                    try:
                        from scipy.linalg import svdvals

                        svs = svdvals(centered)[:n_sv]
                    except Exception:
                        _, svs, _ = np.linalg.svd(centered, full_matrices=False)
                        svs = svs[:n_sv]
                    svs_norm = svs / (svs.sum() + 1e-10)
                    # Isotropy = 1 - normalized std of singular values
                    isotropy = float(1.0 - np.std(svs_norm) / (np.mean(svs_norm) + 1e-10))
                    isotropy = float(np.clip(isotropy, 0.0, 1.0))

                # Hub score: fraction of points that are NN of many others
                if k_nn >= 2:
                    _, indices = nn.kneighbors(embeddings)
                    nn_indices = indices[:, 1:]  # skip self
                    # Count how often each point appears as a neighbor
                    nn_counts = np.bincount(nn_indices.ravel(), minlength=len(embeddings))
                    # Hub = points appearing as NN more than 2x the expected rate
                    expected = k_nn  # each point should appear ~k_nn times
                    hub_threshold = 2.0 * expected
                    hub_score = float(np.mean(nn_counts > hub_threshold))
            except Exception:
                pass

        return np.array([intrinsic_dim, isotropy, hub_score], dtype=np.float32)

    def _advanced_interaction_features(
        self,
        profile: DatasetProfile,
        emb_feats: np.ndarray,
        act_feats: np.ndarray,
    ) -> np.ndarray:
        """1 advanced interaction feature: coverage/action coverage ratio."""
        # Ratio of embedding coverage to action coverage
        emb_coverage = emb_feats[10]  # convex hull volume
        act_coverage = act_feats[4] if len(act_feats) > 4 else 0.0  # range utilization
        ratio = float(emb_coverage / max(act_coverage, 1e-8))
        ratio = min(ratio, 10.0)  # cap
        return np.array([ratio], dtype=np.float32)

    # ==================================================================
    # Stat computation (cached on profile)
    # ==================================================================

    def _ensure_embedding_stats(self, profile: DatasetProfile) -> EmbeddingStats:
        """Return cached stats or compute them from the FAISS index."""
        if profile.embedding_stats is not None:
            return profile.embedding_stats

        stats = _compute_embedding_stats(profile)
        profile.embedding_stats = stats
        return stats

    def _ensure_action_stats(
        self, profile: DatasetProfile, episodes: list[dict] | None
    ) -> ActionStats:
        """Return cached stats or compute them from episode data."""
        if profile.action_stats is not None:
            return profile.action_stats

        if episodes:
            stats = _compute_action_stats(episodes)
        else:
            # Fallback: use what we can from quality signal breakdown
            stats = _action_stats_from_quality(profile)

        profile.action_stats = stats
        return stats


# ======================================================================
# Internal stat computation helpers
# ======================================================================


def _find_capability(profile: DatasetProfile, task: str):
    """Find a CapabilityScore matching *task* by description."""
    for cap in profile.capabilities:
        if cap.task_description == task:
            return cap
    return profile.capabilities[0] if profile.capabilities else None


def _reconstruct_embeddings(profile: DatasetProfile, max_n: int = _MAX_EMBEDDINGS) -> np.ndarray:
    """Reconstruct embeddings from FAISS index, subsampled if large."""
    idx = profile.embedding_index
    n = idx.num_embeddings
    if n == 0:
        return np.zeros((0, idx.dimension), dtype=np.float32)
    n_use = min(n, max_n)
    try:
        embeddings = idx.index.reconstruct_n(0, n_use)
    except RuntimeError:
        # Some index types don't support reconstruct_n
        embeddings = np.array(
            [idx.index.reconstruct(i) for i in range(n_use)], dtype=np.float32
        )
    return embeddings


def _compute_embedding_stats(profile: DatasetProfile) -> EmbeddingStats:
    """Compute embedding distribution statistics from the FAISS index."""
    from scipy import stats as sp_stats

    embeddings = _reconstruct_embeddings(profile)
    n, dim = embeddings.shape if embeddings.ndim == 2 else (0, 0)

    if n < 2:
        return EmbeddingStats()

    # --- Norms ---
    norms = np.linalg.norm(embeddings, axis=1)
    mean_norm = float(np.mean(norms))
    std_norm = float(np.std(norms))

    # --- Pairwise cosine (sampled) ---
    rng = np.random.default_rng(42)
    n_pairs = min(_MAX_PAIRS, n * (n - 1) // 2)
    i_idx = rng.integers(0, n, size=n_pairs)
    j_idx = rng.integers(0, n, size=n_pairs)
    # Avoid self-pairs
    mask = i_idx != j_idx
    i_idx, j_idx = i_idx[mask], j_idx[mask]
    if len(i_idx) > 0:
        # Embeddings are L2-normalized in FAISS IndexFlatIP, so dot = cosine
        cosines = np.sum(embeddings[i_idx] * embeddings[j_idx], axis=1)
        mean_pw_cos = float(np.mean(cosines))
        std_pw_cos = float(np.std(cosines))
        min_pw_cos = float(np.min(cosines))
        max_pw_cos = float(np.max(cosines))
    else:
        mean_pw_cos = std_pw_cos = min_pw_cos = max_pw_cos = 0.0

    # --- PCA ---
    from sklearn.decomposition import PCA

    n_components = min(10, dim, n)
    pca = PCA(n_components=n_components, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pca.fit(embeddings)
    explained = pca.explained_variance_ratio_

    # Effective dimensionality: dims needed for 95% variance
    cumvar = np.cumsum(explained)
    effective_dim = float(np.searchsorted(cumvar, 0.95) + 1)

    # PC1 shape
    pc1 = pca.transform(embeddings)[:, 0]
    skew_pc1 = float(sp_stats.skew(pc1))
    kurt_pc1 = float(sp_stats.kurtosis(pc1))

    # --- Convex hull volume (in PCA space, low-dim) ---
    hull_volume = 0.0
    n_hull_dims = min(3, n_components)
    if n > n_hull_dims + 1:
        try:
            from scipy.spatial import ConvexHull

            hull = ConvexHull(pca.transform(embeddings)[:, :n_hull_dims])
            hull_volume = float(hull.volume)
        except Exception:
            pass

    # --- HDBSCAN clustering ---
    num_clusters = 0
    noise_ratio = 0.0
    sil_score = 0.0
    cal_har = 0.0
    try:
        from sklearn.cluster import HDBSCAN as SkHDBSCAN

        min_cs = min(5, max(2, n // 2))
        clusterer = SkHDBSCAN(min_cluster_size=min_cs, store_centers="centroid")
        labels = clusterer.fit_predict(embeddings)
        unique_labels = set(labels) - {-1}
        num_clusters = len(unique_labels)
        noise_ratio = float(np.mean(labels == -1))

        if num_clusters >= 2:
            from sklearn.metrics import calinski_harabasz_score, silhouette_score

            non_noise = labels != -1
            if non_noise.sum() > num_clusters:
                sil_score = float(
                    silhouette_score(embeddings[non_noise], labels[non_noise])
                )
                cal_har = float(
                    np.log1p(
                        calinski_harabasz_score(embeddings[non_noise], labels[non_noise])
                    )
                )
    except ImportError:
        logger.debug("HDBSCAN not available; cluster features set to 0")

    # --- Temporal features ---
    mean_seq_dist = 0.0
    std_seq_dist = 0.0
    mean_traj_len = 0.0
    std_traj_len = 0.0

    ep_ids = profile.embedding_index.episode_ids
    frame_ids = profile.embedding_index.frame_indices
    if ep_ids and frame_ids and len(ep_ids) == n:
        ep_arr = np.array(ep_ids[:n])
        frame_arr = np.array(frame_ids[:n])
        unique_eps = np.unique(ep_arr)

        all_seq_dists: list[float] = []
        traj_lengths: list[float] = []

        for eid in unique_eps:
            mask = ep_arr == eid
            if mask.sum() < 2:
                continue
            ep_embs = embeddings[mask]
            ep_frames = frame_arr[mask]
            order = np.argsort(ep_frames)
            ep_embs = ep_embs[order]
            diffs = np.linalg.norm(np.diff(ep_embs, axis=0), axis=1)
            all_seq_dists.extend(diffs.tolist())
            traj_lengths.append(float(np.sum(diffs)))

        if all_seq_dists:
            mean_seq_dist = float(np.mean(all_seq_dists))
            std_seq_dist = float(np.std(all_seq_dists))
        if traj_lengths:
            mean_traj_len = float(np.mean(traj_lengths))
            std_traj_len = float(np.std(traj_lengths))

    # --- Entropy estimate (KDE on norms) ---
    entropy_est = 0.0
    if n > 5:
        try:
            from scipy.stats import gaussian_kde

            kde = gaussian_kde(norms)
            # Evaluate on grid
            grid = np.linspace(norms.min(), norms.max(), 200)
            pdf = kde(grid)
            pdf = pdf / (pdf.sum() + 1e-10)
            entropy_est = float(-np.sum(pdf * np.log(pdf + 1e-10)))
        except Exception:
            pass

    # --- Uniformity score ---
    uniformity = 0.0
    if n > 5 and n_components >= 1:
        # Test uniformity of PC1 distribution via KS test against uniform
        pc1_sorted = np.sort((pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-10))
        try:
            ks_stat, _ = sp_stats.kstest(pc1_sorted, "uniform")
            uniformity = float(1.0 - ks_stat)  # 1.0 = perfectly uniform
        except Exception:
            pass

    return EmbeddingStats(
        mean_norm=mean_norm,
        std_norm=std_norm,
        mean_pairwise_cosine=mean_pw_cos,
        std_pairwise_cosine=std_pw_cos,
        min_pairwise_cosine=min_pw_cos,
        max_pairwise_cosine=max_pw_cos,
        num_clusters=num_clusters,
        noise_ratio=noise_ratio,
        silhouette_score=sil_score,
        calinski_harabasz=cal_har,
        convex_hull_volume=hull_volume,
        effective_dimensionality=effective_dim,
        mean_sequential_distance=mean_seq_dist,
        std_sequential_distance=std_seq_dist,
        mean_episode_trajectory_length=mean_traj_len,
        std_episode_trajectory_length=std_traj_len,
        skewness_pc1=skew_pc1,
        kurtosis_pc1=kurt_pc1,
        entropy_estimate=entropy_est,
        uniformity_score=uniformity,
    )


def _compute_action_stats(episodes: list[dict]) -> ActionStats:
    """Compute action distribution statistics from raw episodes."""
    if not episodes:
        return ActionStats()

    all_actions: list[np.ndarray] = []
    ep_lengths: list[int] = []

    for ep in episodes:
        actions = np.asarray(ep.get("actions", []), dtype=np.float64)
        if actions.ndim < 2 or len(actions) < 2:
            continue
        all_actions.append(actions)
        ep_lengths.append(len(actions))

    if not all_actions:
        return ActionStats()

    stacked = np.vstack(all_actions)
    n_total, action_dim = stacked.shape

    # --- Basic statistics ---
    magnitudes = np.linalg.norm(stacked, axis=1)
    mean_mag = float(np.mean(magnitudes))
    std_mag = float(np.std(magnitudes))

    # --- Smoothness (jerk) ---
    smoothness_scores: list[float] = []
    for actions in all_actions:
        if len(actions) < 4:
            continue
        vel = np.diff(actions, axis=0)
        acc = np.diff(vel, axis=0)
        jerk = np.diff(acc, axis=0)
        action_scale = np.std(actions) + 1e-8
        normalized_jerk = np.mean(np.abs(jerk)) / action_scale
        smoothness_scores.append(float(np.clip(1.0 - normalized_jerk / 2.0, 0.0, 1.0)))
    smoothness = float(np.mean(smoothness_scores)) if smoothness_scores else 0.5

    # --- Action range utilization ---
    per_dim_range = np.ptp(stacked, axis=0)
    # Normalise by max observed range across dims
    max_range = np.max(per_dim_range) + 1e-8
    range_util = float(np.mean(per_dim_range / max_range))

    # --- Number of action modes (GMM with BIC) ---
    num_modes = 1.0
    try:
        from sklearn.mixture import GaussianMixture

        sub = stacked
        if len(sub) > _MAX_ACTIONS_GMM:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(sub), _MAX_ACTIONS_GMM, replace=False)
            sub = sub[idx]
        best_bic = np.inf
        best_k = 1
        for k in range(1, min(8, len(sub) // 5 + 1)):
            gmm = GaussianMixture(n_components=k, random_state=42, max_iter=50)
            gmm.fit(sub)
            bic = gmm.bic(sub)
            if bic < best_bic:
                best_bic = bic
                best_k = k
        num_modes = float(best_k)
    except Exception:
        pass

    # --- Per-episode action entropy ---
    ep_entropies: list[float] = []
    for actions in all_actions:
        # Discretise via histogram
        mags = np.linalg.norm(actions, axis=1)
        hist, _ = np.histogram(mags, bins=min(20, len(actions) // 2 + 1), density=True)
        hist = hist / (hist.sum() + 1e-10)
        ent = float(-np.sum(hist * np.log(hist + 1e-10)))
        ep_entropies.append(ent)
    mean_ep_entropy = float(np.mean(ep_entropies)) if ep_entropies else 0.0

    # --- Cross-episode action consistency ---
    # Mean pairwise cosine similarity between episode mean action vectors
    ep_means = []
    for actions in all_actions:
        m = np.mean(actions, axis=0)
        norm = np.linalg.norm(m) + 1e-8
        ep_means.append(m / norm)
    cross_consistency = 0.0
    if len(ep_means) >= 2:
        ep_means_arr = np.array(ep_means)
        cosines = ep_means_arr @ ep_means_arr.T
        # Upper triangle excluding diagonal
        triu_idx = np.triu_indices(len(ep_means), k=1)
        cross_consistency = float(np.mean(cosines[triu_idx]))

    # --- Effective action dimensionality ---
    from sklearn.decomposition import PCA

    eff_dim = float(action_dim)
    if n_total > action_dim and action_dim > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pca = PCA(random_state=42)
            pca.fit(stacked)
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            eff_dim = float(np.searchsorted(cumvar, 0.95) + 1)

    # --- Autocorrelation ---
    autocorrs: list[float] = []
    for actions in all_actions:
        if len(actions) < 3:
            continue
        mags = np.linalg.norm(actions, axis=1)
        if np.std(mags) < 1e-8:
            autocorrs.append(1.0)
            continue
        mags_centered = mags - np.mean(mags)
        var = np.sum(mags_centered**2)
        if var < 1e-10:
            autocorrs.append(0.0)
            continue
        autocorr = float(np.sum(mags_centered[:-1] * mags_centered[1:]) / var)
        autocorrs.append(autocorr)
    mean_autocorr = float(np.mean(autocorrs)) if autocorrs else 0.0

    # --- Zero-action fraction ---
    median_mag = np.median(magnitudes) + 1e-8
    zero_frac = float(np.mean(magnitudes < 1e-3 * median_mag))

    # --- Boundary fraction ---
    per_dim_min = np.min(stacked, axis=0)
    per_dim_max = np.max(stacked, axis=0)
    per_dim_range_full = per_dim_max - per_dim_min + 1e-8
    margin = 0.05 * per_dim_range_full
    at_lower = stacked <= (per_dim_min + margin)
    at_upper = stacked >= (per_dim_max - margin)
    boundary_frac = float(np.mean(at_lower | at_upper))

    return ActionStats(
        dimensionality=float(action_dim),
        mean_magnitude=mean_mag,
        std_magnitude=std_mag,
        smoothness=smoothness,
        action_range_utilization=range_util,
        num_action_modes=num_modes,
        mean_episode_action_entropy=mean_ep_entropy,
        cross_episode_action_consistency=cross_consistency,
        action_dimensionality_effective=eff_dim,
        mean_action_autocorrelation=mean_autocorr,
        zero_action_fraction=zero_frac,
        action_boundary_fraction=boundary_frac,
    )


def _action_stats_from_quality(profile: DatasetProfile) -> ActionStats:
    """Build partial ActionStats from quality signal breakdown only."""
    sb = profile.quality.signal_breakdown
    smoothness = sb.action_smoothness if sb else 0.0
    return ActionStats(smoothness=smoothness)
