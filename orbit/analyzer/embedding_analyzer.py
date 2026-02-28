"""Frame-level embedding analyzer using SigLIP + FAISS + HDBSCAN.

Computes per-frame distribution gap scores between deployment episodes
and training data, clusters failure frames, and generates interactive
visualizations.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from uuid import UUID

import faiss
import numpy as np
from PIL import Image
from tqdm import tqdm

from orbit.analyzer.models import (
    AnalysisReport,
    EmbeddingAnalyzerConfig,
    EpisodeGapSummary,
    FailureCluster,
    FailureClusterReport,
    FrameGapResult,
)
from orbit.logger.schemas import Episode, Outcome

logger = logging.getLogger(__name__)


class EmbeddingAnalyzer:
    """Frame-level embedding-space analyzer for distribution gap detection.

    Workflow:
      1. Index training data (from LeRobot dataset, image directory, or PIL list)
      2. Embed deployment episode frames
      3. Compute per-frame gap scores via FAISS KNN
      4. Aggregate per-episode statistics
      5. Cluster failure frames via HDBSCAN
      6. Generate UMAP + Plotly visualizations
    """

    def __init__(self, config: EmbeddingAnalyzerConfig | None = None) -> None:
        self.config = config or EmbeddingAnalyzerConfig()
        self._model = None
        self._processor = None
        self._faiss_index: faiss.IndexFlatIP | None = None
        self._training_embeddings: np.ndarray | None = None
        self._umap_reducer = None

    # ------------------------------------------------------------------
    # Model loading (lazy)
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load SigLIP model and processor from HuggingFace."""
        if self._model is not None:
            return
        import transformers

        logger.info("Loading SigLIP model: %s", self.config.model_name)
        self._processor = transformers.AutoProcessor.from_pretrained(self.config.model_name)
        self._model = transformers.AutoModel.from_pretrained(self.config.model_name).eval()

        import torch  # noqa: F811

        if self.config.device != "cpu" and torch.cuda.is_available():
            self._model = self._model.to(self.config.device)
        elif self.config.device != "cpu":
            logger.warning(
                "Requested device %s not available, falling back to CPU",
                self.config.device,
            )
            self.config.device = "cpu"

    # ------------------------------------------------------------------
    # Embedding computation
    # ------------------------------------------------------------------

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        """Compute SigLIP embeddings for images in batches.

        Returns ``(N, D)`` float32 array with L2-normalized embeddings.
        """
        self._load_model()
        import torch

        all_embeddings: list[np.ndarray] = []
        for i in tqdm(
            range(0, len(images), self.config.batch_size),
            desc="Embedding images",
            disable=len(images) <= self.config.batch_size,
        ):
            batch = images[i : i + self.config.batch_size]
            inputs = self._processor(images=batch, return_tensors="pt")
            inputs = {
                k: v.to(self.config.device)
                for k, v in inputs.items()
                if isinstance(v, torch.Tensor)
            }
            with torch.no_grad():
                embs = self._model.get_image_features(**inputs)
            embs_np = embs.cpu().numpy()
            norms = np.linalg.norm(embs_np, axis=1, keepdims=True)
            embs_np = embs_np / np.maximum(norms, 1e-8)
            all_embeddings.append(embs_np)

        return np.vstack(all_embeddings).astype(np.float32)

    # ------------------------------------------------------------------
    # Training data indexing
    # ------------------------------------------------------------------

    def index_training_data(
        self,
        source: Path | str | list[Image.Image],
        *,
        image_key: str = "observation.images.front",
    ) -> int:
        """Build FAISS index from training data.

        Args:
            source: Path to a LeRobot dataset directory, path to a flat
                image directory, or a list of PIL Image objects.
            image_key: Column/key for images when loading LeRobot datasets.

        Returns:
            Number of training frames indexed.
        """
        if isinstance(source, list):
            images = source
            cache_key = None
        else:
            source = Path(source)
            cache_key = self._cache_key(str(source))
            cached = self._load_cached_embeddings(cache_key)
            if cached is not None:
                logger.info("Loaded %d cached training embeddings", len(cached))
                self._training_embeddings = cached
                self._build_faiss_index(cached)
                return len(cached)

            if (source / "meta_data").exists() or (source / "meta").exists():
                images = self._load_lerobot_images(source, image_key)
            else:
                images = self._load_image_directory(source)

        if not images:
            raise ValueError(f"No images found in training source: {source}")

        embeddings = self.embed_images(images)
        self._training_embeddings = embeddings
        self._build_faiss_index(embeddings)

        if cache_key is not None:
            self._save_cached_embeddings(cache_key, embeddings)

        logger.info("Indexed %d training frames", len(embeddings))
        return len(embeddings)

    def _load_lerobot_images(self, dataset_path: Path, image_key: str) -> list[Image.Image]:
        """Load images from a LeRobot dataset."""
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(str(dataset_path))
        images: list[Image.Image] = []
        for i in tqdm(range(len(dataset)), desc="Loading LeRobot images"):
            sample = dataset[i]
            if image_key in sample:
                img_data = sample[image_key]
                if isinstance(img_data, np.ndarray):
                    images.append(Image.fromarray(img_data))
                elif isinstance(img_data, Image.Image):
                    images.append(img_data)
                else:
                    import torch

                    if isinstance(img_data, torch.Tensor):
                        arr = img_data.permute(1, 2, 0).numpy()
                        if arr.max() <= 1.0:
                            arr = (arr * 255).astype(np.uint8)
                        else:
                            arr = arr.astype(np.uint8)
                        images.append(Image.fromarray(arr))
        return images

    def _load_image_directory(self, dir_path: Path) -> list[Image.Image]:
        """Load all images from a directory."""
        extensions = {"*.png", "*.jpg", "*.jpeg", "*.bmp"}
        paths: list[Path] = []
        for ext in extensions:
            paths.extend(sorted(dir_path.glob(ext)))
        images = []
        for p in tqdm(paths, desc="Loading images from directory"):
            images.append(Image.open(p).convert("RGB"))
        return images

    # ------------------------------------------------------------------
    # Embedding caching
    # ------------------------------------------------------------------

    def _cache_key(self, source_path: str) -> str:
        """Generate a deterministic cache key from path + model name."""
        payload = f"{source_path}:{self.config.model_name}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _load_cached_embeddings(self, cache_key: str) -> np.ndarray | None:
        """Load .npy cached embeddings if they exist."""
        cache_path = Path(self.config.cache_dir) / f"{cache_key}.npy"
        if cache_path.exists():
            return np.load(cache_path)
        return None

    def _save_cached_embeddings(self, cache_key: str, embeddings: np.ndarray) -> None:
        """Save embeddings as .npy to cache dir."""
        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_dir / f"{cache_key}.npy", embeddings)
        logger.info("Cached embeddings to %s/%s.npy", cache_dir, cache_key)

    # ------------------------------------------------------------------
    # FAISS index
    # ------------------------------------------------------------------

    def _build_faiss_index(self, embeddings: np.ndarray) -> None:
        """Build IndexFlatIP (cosine similarity via dot product on normalized vectors)."""
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)

        if self.config.use_gpu_faiss:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                logger.info("Using GPU FAISS index")
            except Exception:
                logger.warning("GPU FAISS not available, falling back to CPU")

        index.add(embeddings.astype(np.float32))
        self._faiss_index = index

    # ------------------------------------------------------------------
    # Gap score computation
    # ------------------------------------------------------------------

    def compute_gap_scores(
        self, episodes: list[Episode]
    ) -> tuple[list[FrameGapResult], list[EpisodeGapSummary]]:
        """Compute per-frame gap scores for deployment episodes.

        For each frame, finds K nearest training frames and computes
        ``gap_score = 1.0 - mean(cosine_similarities)``.

        Returns:
            frame_results: Per-frame gap results across all episodes.
            episode_summaries: Aggregated stats per episode.
        """
        if self._faiss_index is None:
            raise RuntimeError("Training data not indexed. Call index_training_data() first.")

        all_frame_results: list[FrameGapResult] = []
        episode_summaries: list[EpisodeGapSummary] = []

        for ep in tqdm(episodes, desc="Computing gap scores"):
            images = self._load_episode_images(ep)
            if not images:
                logger.warning(
                    "Episode %s has no loadable images, skipping",
                    ep.episode_id,
                )
                continue

            embeddings = self.embed_images(images)
            k = min(self.config.num_neighbors, self._faiss_index.ntotal)
            similarities, indices = self._faiss_index.search(embeddings, k)

            ep_frame_results: list[FrameGapResult] = []
            for frame_idx in range(len(images)):
                sims = similarities[frame_idx]
                idxs = indices[frame_idx]
                gap_score = float(1.0 - np.mean(sims))
                fr = FrameGapResult(
                    frame_idx=frame_idx,
                    episode_id=ep.episode_id,
                    gap_score=gap_score,
                    nearest_distances=[float(1.0 - s) for s in sims],
                    nearest_indices=idxs.tolist(),
                )
                ep_frame_results.append(fr)

            all_frame_results.extend(ep_frame_results)
            summary = self._aggregate_episode(ep.episode_id, ep.outcome, ep_frame_results)
            episode_summaries.append(summary)

        return all_frame_results, episode_summaries

    def _load_episode_images(self, episode: Episode) -> list[Image.Image]:
        """Load images from episode frames on disk."""
        images: list[Image.Image] = []
        for frame in episode.frames:
            if not frame.image_path:
                continue
            try:
                images.append(Image.open(frame.image_path).convert("RGB"))
            except Exception as exc:
                logger.warning("Failed to load image %s: %s", frame.image_path, exc)
        return images

    def _aggregate_episode(
        self,
        episode_id: UUID,
        outcome: Outcome,
        frame_results: list[FrameGapResult],
    ) -> EpisodeGapSummary:
        """Aggregate frame-level results into episode summary."""
        gaps = [fr.gap_score for fr in frame_results]
        return EpisodeGapSummary(
            episode_id=episode_id,
            outcome=outcome.value,
            mean_gap=float(np.mean(gaps)),
            max_gap=float(np.max(gaps)),
            gap_percentile_95=float(np.percentile(gaps, 95)),
            gap_trajectory=gaps,
            num_frames=len(gaps),
        )

    # ------------------------------------------------------------------
    # Failure clustering
    # ------------------------------------------------------------------

    def cluster_failures(
        self,
        episodes: list[Episode],
        frame_results: list[FrameGapResult] | None = None,
    ) -> FailureClusterReport:
        """Cluster failure episode frames using HDBSCAN.

        Args:
            episodes: Deployment episodes (only FAILURE outcomes are used).
            frame_results: Pre-computed frame results. If *None* they are
                recomputed via :meth:`compute_gap_scores`.

        Returns:
            :class:`FailureClusterReport` with clusters ranked by average
            gap score (highest first).
        """
        try:
            import hdbscan
        except ImportError:
            raise ImportError(
                "hdbscan is required for failure clustering. Install with: pip install hdbscan"
            ) from None

        failure_episodes = [ep for ep in episodes if ep.outcome == Outcome.FAILURE]
        if not failure_episodes:
            return FailureClusterReport()

        # Compute gap scores for failures if not provided
        if frame_results is None:
            frame_results, _ = self.compute_gap_scores(failure_episodes)

        # Build gap-score lookup
        gap_lookup: dict[tuple[str, int], float] = {}
        for fr in frame_results:
            gap_lookup[(str(fr.episode_id), fr.frame_idx)] = fr.gap_score

        # Embed failure frames
        failure_images: list[Image.Image] = []
        frame_meta: list[tuple[str, int]] = []  # (episode_id_str, frame_idx)
        for ep in failure_episodes:
            for i, frame in enumerate(ep.frames):
                if not frame.image_path:
                    continue
                try:
                    failure_images.append(Image.open(frame.image_path).convert("RGB"))
                    frame_meta.append((str(ep.episode_id), i))
                except Exception:
                    continue

        if len(failure_images) < self.config.hdbscan_min_cluster_size:
            return FailureClusterReport(
                num_failure_frames=len(failure_images),
            )

        failure_embs = self.embed_images(failure_images)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.config.hdbscan_min_cluster_size,
            min_samples=self.config.hdbscan_min_samples,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(failure_embs)
        unique_labels = set(labels)
        noise_count = int(np.sum(labels == -1))

        clusters: list[FailureCluster] = []
        for label in sorted(unique_labels):
            if label == -1:
                continue
            mask = labels == label
            cluster_embs = failure_embs[mask]
            cluster_indices = np.where(mask)[0].tolist()
            cluster_meta = [frame_meta[idx] for idx in cluster_indices]

            centroid = cluster_embs.mean(axis=0)
            centroid = centroid / max(np.linalg.norm(centroid), 1e-8)

            # Representative frames: closest to centroid
            dists = np.linalg.norm(cluster_embs - centroid, axis=1)
            rep_order = np.argsort(dists)[:5]
            rep_indices = [cluster_indices[j] for j in rep_order]
            rep_episode_ids = list(dict.fromkeys(frame_meta[idx][0] for idx in rep_indices))

            # Average gap score
            gap_scores = [gap_lookup.get(meta, 0.0) for meta in cluster_meta]
            avg_gap = float(np.mean(gap_scores)) if gap_scores else 0.0

            # Temporal distribution
            temporal: dict[str, int] = {"early": 0, "mid": 0, "late": 0}
            for ep_id, fidx in cluster_meta:
                ep = next(
                    (e for e in failure_episodes if str(e.episode_id) == ep_id),
                    None,
                )
                if ep and ep.num_frames > 0:
                    ratio = fidx / ep.num_frames
                    if ratio < 1 / 3:
                        temporal["early"] += 1
                    elif ratio < 2 / 3:
                        temporal["mid"] += 1
                    else:
                        temporal["late"] += 1

            clusters.append(
                FailureCluster(
                    cluster_id=int(label),
                    size=int(mask.sum()),
                    avg_gap_score=avg_gap,
                    representative_frame_indices=rep_indices,
                    representative_episode_ids=rep_episode_ids,
                    temporal_distribution=temporal,
                    centroid=centroid.tolist(),
                )
            )

        clusters.sort(key=lambda c: c.avg_gap_score, reverse=True)

        return FailureClusterReport(
            clusters=clusters,
            num_failure_frames=len(failure_images),
            num_noise_frames=noise_count,
            num_clusters=len(clusters),
        )

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def generate_visualization(
        self,
        episodes: list[Episode],
        frame_results: list[FrameGapResult] | None = None,
        episode_summaries: list[EpisodeGapSummary] | None = None,
        cluster_report: FailureClusterReport | None = None,
        output_dir: Path | str = "./orbit_viz",
    ) -> dict[str, str]:
        """Generate interactive visualizations and save as HTML files.

        Generates:
          1. UMAP scatter plot (training=gray, success=green, failure=red).
          2. Gap heatmap over episode timelines.

        Returns:
            Mapping of visualization name to HTML file path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, str] = {}

        # Compute results if not provided
        if frame_results is None or episode_summaries is None:
            frame_results, episode_summaries = self.compute_gap_scores(episodes)

        # ----- UMAP scatter -----
        if self._training_embeddings is not None:
            deploy_images: list[Image.Image] = []
            deploy_meta: list[tuple[str, int, str]] = []
            for ep in episodes:
                for i, frame in enumerate(ep.frames):
                    if not frame.image_path:
                        continue
                    try:
                        deploy_images.append(Image.open(frame.image_path).convert("RGB"))
                        deploy_meta.append((str(ep.episode_id), i, ep.outcome.value))
                    except Exception:
                        continue

            if deploy_images:
                deploy_embs = self.embed_images(deploy_images)
                self._fit_umap(self._training_embeddings)

                train_2d = self._umap_reducer.transform(self._training_embeddings)
                deploy_2d = self._umap_reducer.transform(deploy_embs)

                gap_lookup: dict[tuple[str, int], float] = {}
                cluster_lookup: dict[tuple[str, int], int] = {}
                for fr in frame_results:
                    gap_lookup[(str(fr.episode_id), fr.frame_idx)] = fr.gap_score
                if cluster_report:
                    for cl in cluster_report.clusters:
                        for idx in cl.representative_frame_indices:
                            # Mark representative frames with cluster id
                            cluster_lookup[idx] = cl.cluster_id

                ep_ids = [m[0][:8] for m in deploy_meta]
                frame_indices = [m[1] for m in deploy_meta]
                outcomes = [m[2] for m in deploy_meta]
                gap_scores = [gap_lookup.get((m[0], m[1]), 0.0) for m in deploy_meta]

                scatter_path = output_dir / "embedding_space.html"
                self._generate_umap_scatter(
                    train_2d,
                    deploy_2d,
                    ep_ids,
                    frame_indices,
                    gap_scores,
                    outcomes,
                    output_path=scatter_path,
                )
                paths["embedding_space"] = str(scatter_path)

        # ----- Gap heatmap -----
        if episode_summaries:
            heatmap_path = output_dir / "gap_heatmap.html"
            self._generate_gap_heatmap(episode_summaries, heatmap_path)
            paths["gap_heatmap"] = str(heatmap_path)

        return paths

    def _fit_umap(self, training_embeddings: np.ndarray) -> None:
        """Fit UMAP on training embeddings (stores reducer for transform)."""
        import umap

        self._umap_reducer = umap.UMAP(
            n_neighbors=min(
                self.config.umap_n_neighbors,
                len(training_embeddings) - 1,
            ),
            min_dist=self.config.umap_min_dist,
            metric=self.config.umap_metric,
            n_components=2,
        )
        self._umap_reducer.fit(training_embeddings)

    def _generate_umap_scatter(
        self,
        training_2d: np.ndarray,
        deployment_2d: np.ndarray,
        episode_ids: list[str],
        frame_indices: list[int],
        gap_scores: list[float],
        outcomes: list[str],
        output_path: Path,
    ) -> None:
        """Build Plotly UMAP scatter and write to HTML."""
        import plotly.graph_objects as go

        fig = go.Figure()

        # Training data (gray background)
        fig.add_trace(
            go.Scatter(
                x=training_2d[:, 0],
                y=training_2d[:, 1],
                mode="markers",
                marker=dict(size=3, color="#cccccc", opacity=0.4),
                name="Training",
                hoverinfo="skip",
            )
        )

        # Deployment data colored by outcome
        color_map = {"success": "#2ecc71", "failure": "#e74c3c", "unknown": "#95a5a6"}
        for outcome_val, color in color_map.items():
            mask = [i for i, o in enumerate(outcomes) if o == outcome_val]
            if not mask:
                continue
            fig.add_trace(
                go.Scatter(
                    x=deployment_2d[mask, 0],
                    y=deployment_2d[mask, 1],
                    mode="markers",
                    marker=dict(size=6, color=color, opacity=0.7),
                    name=outcome_val.capitalize(),
                    text=[
                        f"ep={episode_ids[i]} frame={frame_indices[i]} gap={gap_scores[i]:.3f}"
                        for i in mask
                    ],
                    hoverinfo="text",
                )
            )

        fig.update_layout(
            title="Embedding Space (UMAP)",
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            template="plotly_white",
            width=900,
            height=700,
        )
        fig.write_html(str(output_path))
        logger.info("UMAP scatter saved to %s", output_path)

    def _generate_gap_heatmap(
        self,
        episode_summaries: list[EpisodeGapSummary],
        output_path: Path,
    ) -> None:
        """Build Plotly gap heatmap and write to HTML."""
        import plotly.graph_objects as go

        sorted_summaries = sorted(episode_summaries, key=lambda s: s.mean_gap, reverse=True)

        max_frames = max(s.num_frames for s in sorted_summaries)
        z: list[list[float | None]] = []
        y_labels: list[str] = []

        for s in sorted_summaries:
            row = list(s.gap_trajectory)
            row.extend([None] * (max_frames - len(row)))
            z.append(row)
            y_labels.append(f"{str(s.episode_id)[:8]} ({s.outcome})")

        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=list(range(max_frames)),
                y=y_labels,
                colorscale="RdYlBu_r",
                colorbar=dict(title="Gap Score"),
                hoverongaps=False,
            )
        )
        fig.update_layout(
            title="Gap Score Heatmap (episodes sorted by mean gap)",
            xaxis_title="Frame Index",
            yaxis_title="Episode",
            template="plotly_white",
            width=1000,
            height=max(400, 30 * len(sorted_summaries)),
        )
        fig.write_html(str(output_path))
        logger.info("Gap heatmap saved to %s", output_path)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def analyze(
        self,
        training_source: Path | str | list[Image.Image],
        deployment_episodes: list[Episode],
        *,
        image_key: str = "observation.images.front",
        output_dir: Path | str = "./orbit_viz",
        generate_viz: bool = True,
    ) -> AnalysisReport:
        """Run the full analysis pipeline.

        1. Index training data.
        2. Compute gap scores for all deployment episodes.
        3. Cluster failure frames.
        4. Generate visualizations (optional).
        """
        if isinstance(training_source, str):
            training_source = Path(training_source)

        n_training = self.index_training_data(training_source, image_key=image_key)

        frame_results, episode_summaries = self.compute_gap_scores(deployment_episodes)

        failure_episodes = [ep for ep in deployment_episodes if ep.outcome == Outcome.FAILURE]
        cluster_report = None
        if failure_episodes:
            failure_frame_results = [
                fr
                for fr in frame_results
                if any(str(fr.episode_id) == str(ep.episode_id) for ep in failure_episodes)
            ]
            cluster_report = self.cluster_failures(failure_episodes, failure_frame_results)

        viz_paths: dict[str, str] = {}
        if generate_viz:
            viz_paths = self.generate_visualization(
                deployment_episodes,
                frame_results,
                episode_summaries,
                cluster_report,
                output_dir,
            )

        n_deployment = sum(s.num_frames for s in episode_summaries)

        return AnalysisReport(
            episode_summaries=episode_summaries,
            cluster_report=cluster_report,
            training_embedding_count=n_training,
            deployment_embedding_count=n_deployment,
            visualization_paths=viz_paths,
        )
