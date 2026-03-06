"""Main DatasetProfiler class for analyzing robot datasets."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from orbit.profile.types import DatasetProfile

logger = logging.getLogger(__name__)


class DatasetProfiler:
    """Analyzes a robot dataset and predicts what tasks it can support.

    Builds a complete profile including embedding coverage, task capability
    scores, data quality metrics, and prescriptions for what to collect next.
    """

    def __init__(
        self,
        embedding_model: str = "google/siglip-base-patch16-224",
        device: str = "cpu",
    ) -> None:
        """Initialize with SigLIP model for embeddings.

        Parameters
        ----------
        embedding_model:
            HuggingFace model ID for visual embedding extraction.
        device:
            Torch device to run the model on (``"cpu"`` or ``"cuda"``).
        """
        self.embedding_model = embedding_model
        self.device = device

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
        extractor = EmbeddingExtractor(model_name=self.embedding_model, device=self.device)
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
        if task_descriptions:
            scorer = CapabilityScorer(model_name=self.embedding_model, device=self.device)
            capabilities = scorer.score_tasks(embedding_index, coverage, quality, task_descriptions)

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

        return profile

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
