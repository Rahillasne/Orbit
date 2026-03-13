"""Main DatasetProfiler class for analyzing robot datasets."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from orbit.profile.types import DatasetProfile, ScoringWeights

logger = logging.getLogger(__name__)

# Models where image embeddings live in a different space than text encoders
_VISION_ONLY_MODELS = ("r3m", "voltron")


class DatasetProfiler:
    """Analyzes a robot dataset and predicts what tasks it can support.

    Builds a complete profile including embedding coverage, task capability
    scores, data quality metrics, and prescriptions for what to collect next.
    """

    def __init__(
        self,
        embedding_model: str = "auto",
        device: str = "cpu",
        fast_mode: bool = False,
        scoring_weights: ScoringWeights | None = None,
    ) -> None:
        """Initialize the profiler.

        Parameters
        ----------
        embedding_model:
            Embedding backend: ``"auto"``, ``"r3m"``, ``"siglip"``,
            ``"openclip"``, or a HuggingFace model ID for backward compat.
        device:
            Torch device to run the model on (``"cpu"`` or ``"cuda"``).
        fast_mode:
            Use lightweight OpenCLIP ViT-B/32 for faster CPU embedding.
        scoring_weights:
            Configurable weights for the capability scoring formula.
        """
        self.embedding_model = embedding_model
        self.device = device
        self.fast_mode = fast_mode
        self.scoring_weights = scoring_weights

    def _is_vision_only_model(self) -> bool:
        """Check if the resolved embedding model lacks a text encoder."""
        return self.embedding_model in _VISION_ONLY_MODELS

    def profile(
        self,
        data_dir: str,
        task_descriptions: list[str] | None = None,
    ) -> DatasetProfile:
        """Run the full profiling pipeline.

        Steps:
            1. Extract embeddings from all frames
            2. Build FAISS index and compute coverage
            3. Score capabilities for each task
            4. Compute quality metrics
            5. Generate prescriptions for gaps
            6. Return complete DatasetProfile

        Parameters
        ----------
        data_dir:
            Path to the dataset directory.
        task_descriptions:
            Optional list of task descriptions to score capabilities against.

        Returns
        -------
        DatasetProfile
            Complete profile of the dataset's capabilities.
        """
        from orbit.profile.capability import CapabilityScorer
        from orbit.profile.coverage import CoverageAnalyzer
        from orbit.profile.embedding import EmbeddingExtractor
        from orbit.profile.quality import QualityEstimator
        from orbit.profile.report import ProfileReporter

        # 1. Extract embeddings
        extractor = EmbeddingExtractor(
            model_name=self.embedding_model
            if "/" in self.embedding_model
            else "google/siglip-base-patch16-224",
            device=self.device,
            fast_mode=self.fast_mode,
            embedding_model=self.embedding_model
            if self.embedding_model in EmbeddingExtractor.KNOWN_MODELS
            else "siglip",
        )
        embedding_index = extractor.extract_from_directory(data_dir)

        # 2. Analyze coverage
        coverage_analyzer = CoverageAnalyzer()
        coverage = coverage_analyzer.analyze(embedding_index)

        # 3. Load episodes and estimate quality
        episodes = self._load_episodes(data_dir)
        quality_estimator = QualityEstimator()
        quality = quality_estimator.estimate_quality(episodes)

        # 4. Score capabilities
        capabilities = []
        relevance_index = None
        if task_descriptions:
            # For vision-only models (R3M, Voltron), build a secondary
            # SigLIP/OpenCLIP index for text-image relevance scoring.
            resolved_model = extractor.embedding_model
            if resolved_model in _VISION_ONLY_MODELS:
                relevance_index = self._build_relevance_index(data_dir)

            scorer = CapabilityScorer(
                model_name="google/siglip-base-patch16-224",
                device=self.device,
                scoring_weights=self.scoring_weights,
                fast_mode=self.fast_mode,
            )
            capabilities = scorer.score_tasks(
                embedding_index,
                coverage,
                quality,
                task_descriptions,
                relevance_index=relevance_index,
            )

        # 5. Build profile
        profile = DatasetProfile(
            dataset_name=Path(data_dir).name,
            num_episodes=len(set(embedding_index.episode_ids)),
            num_frames=embedding_index.num_embeddings,
            embedding_index=embedding_index,
            coverage=coverage,
            capabilities=capabilities,
            quality=quality,
            timestamp=datetime.now().isoformat(),
        )

        # 6. Generate prescriptions
        reporter = ProfileReporter()
        profile.prescriptions = reporter.generate_prescriptions(profile)

        # 7. Run V2 quality prediction if model exists
        self._run_quality_prediction(profile, episodes)

        return profile

    def _run_quality_prediction(
        self, profile: DatasetProfile, episodes: list[dict]
    ) -> None:
        """Attempt to run the V2 quality predictor, skip if unavailable."""
        model_path = Path(__file__).parent / "pretrained" / "quality_model_v2.pkl"
        if not model_path.exists():
            logger.debug("V2 quality model not found at %s, skipping prediction", model_path)
            return

        try:
            from orbit.profile.feature_extractor import DatasetFeatureExtractor
            from orbit.profile.predictor_v2 import DatasetQualityModelV2
            from orbit.profile.report_card import ReportCardGenerator

            extractor = DatasetFeatureExtractor()
            generator = ReportCardGenerator()
            card = generator.generate(profile)

            features = extractor.extract(
                profile=profile,
                report_card=card,
                episodes=episodes,
                metadata={},
            )

            model = DatasetQualityModelV2(model_path=str(model_path))
            prediction = model.predict(features)

            profile.predicted_success_rate = prediction.predicted_success_rate
            profile.prediction_ci_low = prediction.confidence_interval_low
            profile.prediction_ci_high = prediction.confidence_interval_high
            profile.prediction_confidence = prediction.confidence_level

            if prediction.warning:
                logger.warning("Quality prediction warning: %s", prediction.warning)

        except Exception:
            logger.debug("V2 quality prediction failed", exc_info=True)

    def _build_relevance_index(self, data_dir: str):
        """Build a lightweight SigLIP/OpenCLIP index for text-image relevance.

        Sub-samples frames (every 5th) to keep this fast.
        """
        from orbit.profile.embedding import EmbeddingExtractor

        logger.info("Building secondary relevance index (SigLIP/OpenCLIP) for text matching")
        relevance_extractor = EmbeddingExtractor(
            model_name="google/siglip-base-patch16-224",
            device=self.device,
            fast_mode=self.fast_mode,
            embedding_model="siglip" if not self.fast_mode else "openclip",
        )
        return relevance_extractor.extract_from_directory(data_dir, sample_rate=5)

    def profile_from_hub(
        self,
        repo_id: str,
        task_descriptions: list[str] | None = None,
        max_episodes: int | None = None,
        cache_dir: str | None = None,
    ) -> DatasetProfile:
        """Profile a dataset directly from HuggingFace Hub.

        Downloads and converts a LeRobot dataset to ORBIT's HDF5 format,
        then runs the full profiling pipeline.

        Parameters
        ----------
        repo_id:
            HuggingFace dataset repository ID (e.g. ``"lerobot/aloha_sim"``).
        task_descriptions:
            Optional list of task descriptions to score capabilities against.
        max_episodes:
            Maximum number of episodes to convert.  ``None`` for all.
        cache_dir:
            Directory to cache converted data.  Uses a temp dir if ``None``.

        Returns
        -------
        DatasetProfile
            Complete profile of the dataset's capabilities.
        """
        import tempfile

        from orbit.profile.loaders import DatasetLoader

        output_dir = Path(cache_dir or tempfile.mkdtemp(prefix="orbit_lerobot_"))
        DatasetLoader.from_lerobot(repo_id, output_dir, max_episodes=max_episodes)
        return self.profile(str(output_dir), task_descriptions)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_episodes(self, data_dir: str) -> list[dict]:
        """Load episode state/action data from HDF5 files.

        Returns a list of dicts with keys ``episode_id``, ``states``,
        ``actions`` suitable for :class:`QualityEstimator`.
        """
        data_path = Path(data_dir)
        h5_files = sorted(data_path.glob("session_*.h5")) or sorted(data_path.glob("*.h5"))

        episodes: list[dict] = []
        if h5_files:
            episodes = self._load_episodes_from_hdf5(h5_files)

        if not episodes:
            # Synthesize placeholder episodes from image count
            logger.warning(
                "No state/action data found in %s; synthesizing placeholder episodes",
                data_dir,
            )
            episodes = self._synthesize_placeholder_episodes(data_dir)

        return episodes

    @staticmethod
    def _load_episodes_from_hdf5(h5_files: list[Path]) -> list[dict]:
        """Extract states and actions from HDF5 episode groups."""
        import h5py

        episodes: list[dict] = []
        for h5_path in h5_files:
            with h5py.File(h5_path, "r") as f:
                if "episodes" not in f:
                    continue
                for ep_key in f["episodes"]:
                    grp = f["episodes"][ep_key]
                    # Try dedicated state/action arrays first
                    states = None
                    actions = None
                    if "states" in grp:
                        states = grp["states"][:]
                    elif "joint_positions" in grp:
                        states = grp["joint_positions"][:]
                    if "actions" in grp:
                        actions = grp["actions"][:]

                    if states is not None and actions is not None:
                        min_len = min(len(states), len(actions))
                        if min_len >= 2:
                            episodes.append(
                                {
                                    "episode_id": int(ep_key),
                                    "states": states[:min_len],
                                    "actions": actions[:min_len],
                                }
                            )
        return episodes

    @staticmethod
    def _synthesize_placeholder_episodes(data_dir: str) -> list[dict]:
        """Create minimal placeholder episodes for quality estimation."""
        rng = np.random.default_rng(0)
        state_dim = 6
        action_dim = 6
        T = 20
        return [
            {
                "episode_id": 0,
                "states": rng.standard_normal((T, state_dim)),
                "actions": rng.standard_normal((T, action_dim)),
            }
        ]
