"""Embedding-based gap analysis between successful and failed episodes."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from orbit.logger.schemas import EpisodeRecord


@dataclass
class GapAnalysisResult:
    """Result of embedding gap analysis."""

    mean_gap_distance: float
    nearest_success_distances: list[float]
    failure_cluster_ids: list[int] = field(default_factory=list)
    embeddings_2d: np.ndarray | None = None
    labels: np.ndarray | None = None
    raw_embeddings: np.ndarray | None = None


@dataclass
class AnalyzerConfig:
    """Configuration for embedding gap analysis."""

    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    device: str = "cpu"
    image_key: str = "front"
    sample_steps: int = 10
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_n_components: int = 2
    faiss_nprobe: int = 10


class EmbeddingGapAnalyzer:
    """Analyzes the embedding-space gap between successful and failed episodes.

    Workflow:

    1. Extract representative images from episodes (evenly sampled steps)
    2. Compute CLIP embeddings for all images
    3. Build FAISS index of success embeddings
    4. Query failed episode embeddings against the index
    5. Compute gap metrics (distances, cluster separation)
    6. Generate UMAP projections for visualization
    """

    def __init__(self, config: AnalyzerConfig | None = None) -> None:
        self.config = config or AnalyzerConfig()
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    def _load_clip_model(self) -> None:
        if self._model is not None:
            return
        import open_clip

        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.config.clip_model_name, pretrained=self.config.clip_pretrained
        )
        self._model.eval().to(self.config.device)
        self._tokenizer = open_clip.get_tokenizer(self.config.clip_model_name)

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        """Compute CLIP embeddings for a list of images.

        Returns an array of shape ``(N, D)`` with L2-normalized embeddings.
        """
        self._load_clip_model()
        import torch

        preprocessed = torch.stack([self._preprocess(img) for img in images])
        preprocessed = preprocessed.to(self.config.device)

        with torch.no_grad():
            embeddings = self._model.encode_image(preprocessed)

        # L2 normalize
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().numpy()

    def _sample_images_from_episode(self, episode: EpisodeRecord) -> list[Image.Image]:
        """Sample evenly-spaced images from an episode."""
        if not episode.steps:
            return []

        n_steps = len(episode.steps)
        n_samples = min(self.config.sample_steps, n_steps)
        indices = np.linspace(0, n_steps - 1, n_samples, dtype=int)

        images = []
        for idx in indices:
            step = episode.steps[idx]
            if self.config.image_key in step.images:
                img = step.images[self.config.image_key]
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                images.append(img)

        return images

    def embed_episodes(self, episodes: list[EpisodeRecord]) -> np.ndarray:
        """Compute mean CLIP embeddings for each episode.

        Returns an array of shape ``(N_episodes, D)``.
        """
        all_embeddings = []
        for ep in episodes:
            images = self._sample_images_from_episode(ep)
            if not images:
                raise ValueError(
                    f"Episode {ep.episode_id} has no images for key '{self.config.image_key}'"
                )
            embs = self.embed_images(images)
            mean_emb = embs.mean(axis=0)
            mean_emb = mean_emb / np.linalg.norm(mean_emb)
            all_embeddings.append(mean_emb)

        return np.array(all_embeddings, dtype=np.float32)

    def _build_faiss_index(self, embeddings: np.ndarray):
        """Build a FAISS L2 index from embeddings."""
        import faiss

        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings.astype(np.float32))
        return index

    def compute_umap(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce embeddings to 2D via UMAP."""
        import umap

        reducer = umap.UMAP(
            n_neighbors=min(self.config.umap_n_neighbors, len(embeddings) - 1),
            min_dist=self.config.umap_min_dist,
            n_components=self.config.umap_n_components,
        )
        return reducer.fit_transform(embeddings)

    def analyze(
        self,
        success_episodes: list[EpisodeRecord],
        failure_episodes: list[EpisodeRecord],
    ) -> GapAnalysisResult:
        """Run the full gap analysis pipeline."""
        success_embs = self.embed_episodes(success_episodes)
        failure_embs = self.embed_episodes(failure_episodes)

        # Build FAISS index from success embeddings
        index = self._build_faiss_index(success_embs)

        # Query each failure embedding
        distances, _ = index.search(failure_embs, 1)
        nearest_distances = distances[:, 0].tolist()

        # Centroid gap
        success_centroid = success_embs.mean(axis=0)
        failure_centroid = failure_embs.mean(axis=0)
        mean_gap = float(np.linalg.norm(success_centroid - failure_centroid))

        # UMAP projection
        all_embs = np.vstack([success_embs, failure_embs])
        labels = np.array([0] * len(success_embs) + [1] * len(failure_embs), dtype=np.int32)

        embeddings_2d = None
        if len(all_embs) > 2:
            embeddings_2d = self.compute_umap(all_embs)

        return GapAnalysisResult(
            mean_gap_distance=mean_gap,
            nearest_success_distances=nearest_distances,
            failure_cluster_ids=[1] * len(failure_embs),
            embeddings_2d=embeddings_2d,
            labels=labels,
            raw_embeddings=all_embs,
        )
