"""Coverage density analysis of embedding space."""

from __future__ import annotations

import logging
import os
import warnings

import numpy as np

# Prevent OpenMP threading conflict between faiss/torch and HDBSCAN/UMAP on macOS
if os.environ.get("OMP_NUM_THREADS") is None:
    os.environ["OMP_NUM_THREADS"] = "1"

from orbit.profile.types import CoverageMap, EmbeddingIndex

logger = logging.getLogger(__name__)


def _reconstruct_embeddings(embedding_index: EmbeddingIndex) -> np.ndarray:
    """Reconstruct all embeddings from a FAISS index."""
    return np.vstack(
        [embedding_index.index.reconstruct(i) for i in range(embedding_index.num_embeddings)]
    ).astype(np.float32)


class CoverageAnalyzer:
    """Analyse coverage density across embedding space.

    Uses HDBSCAN (with k-means fallback) for clustering, k-NN density
    estimation for coverage scoring, and optional UMAP for 2-D projection.
    """

    def __init__(self, n_clusters: int = 20, min_cluster_size: int = 5) -> None:
        self.n_clusters = n_clusters
        self.min_cluster_size = min_cluster_size

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------

    def analyze(self, embedding_index: EmbeddingIndex) -> CoverageMap:
        """Compute coverage density across embedding space."""
        embeddings = _reconstruct_embeddings(embedding_index)
        n = len(embeddings)

        if n == 0:
            return CoverageMap(
                dense_regions=[],
                sparse_regions=[],
                overall_coverage_score=0.0,
                umap_projection=None,
            )

        # --- Clustering ---
        labels = self._cluster(embeddings)
        unique_labels = set(labels)
        unique_labels.discard(-1)

        # --- Per-cluster stats ---
        clusters: list[dict] = []
        for label in sorted(unique_labels):
            mask = labels == label
            pts = embeddings[mask]
            centroid = pts.mean(axis=0)
            centroid /= max(np.linalg.norm(centroid), 1e-8)
            dists = np.linalg.norm(pts - centroid, axis=1)
            mean_dist = float(dists.mean()) if len(dists) > 0 else 0.0
            density = len(pts) / max(mean_dist, 1e-8)
            clusters.append({"center": centroid, "density": density, "size": int(mask.sum())})

        # Split into dense / sparse by median density
        if clusters:
            densities = [c["density"] for c in clusters]
            med = float(np.median(densities))
            dense_regions = [c for c in clusters if c["density"] >= med]
            sparse_regions = [c for c in clusters if c["density"] < med]
        else:
            dense_regions, sparse_regions = [], []

        # --- k-NN coverage score ---
        k = min(10, n - 1) if n > 1 else 1
        import faiss

        search_index = faiss.IndexFlatIP(embeddings.shape[1])
        search_index.add(embeddings)
        sims, _ = search_index.search(embeddings, k + 1)
        # Exclude self-match (first column), convert similarity to distance
        if n > 1:
            neighbor_sims = sims[:, 1:]
            mean_dist_per_point = 1.0 - neighbor_sims.mean(axis=1)
        else:
            mean_dist_per_point = np.array([0.0])
        coverage_per_point = 1.0 / (1.0 + mean_dist_per_point)
        overall_score = float(coverage_per_point.mean())
        overall_score = float(np.clip(overall_score, 0.0, 1.0))

        # --- Optional UMAP ---
        umap_proj = self._try_umap(embeddings)

        return CoverageMap(
            dense_regions=dense_regions,
            sparse_regions=sparse_regions,
            overall_coverage_score=overall_score,
            umap_projection=umap_proj,
        )

    # ------------------------------------------------------------------
    # Gap detection
    # ------------------------------------------------------------------

    def find_gaps(
        self,
        index: EmbeddingIndex,
        reference_index: EmbeddingIndex | None = None,
    ) -> list[dict]:
        """Find regions where coverage is weak.

        If *reference_index* is provided, finds regions in the reference
        that are poorly covered by *index*.  Otherwise finds sparse regions
        within the dataset itself.
        """
        import faiss

        if reference_index is not None:
            ref_embs = _reconstruct_embeddings(reference_index)
            sims, _ = index.index.search(ref_embs, 1)
            gaps: list[dict] = []
            for i in range(len(ref_embs)):
                gap_score = float(1.0 - sims[i, 0])
                if gap_score > 0.3:
                    gaps.append(
                        {
                            "center": ref_embs[i],
                            "gap_score": gap_score,
                            "nearest_covered_distance": gap_score,
                        }
                    )
            gaps.sort(key=lambda g: g["gap_score"], reverse=True)
            return gaps

        # Self-gap analysis: find points farthest from cluster centroids
        embeddings = _reconstruct_embeddings(index)
        labels = self._cluster(embeddings)
        unique_labels = set(labels)
        unique_labels.discard(-1)

        if not unique_labels:
            # No clusters found; every point is a gap candidate
            return [
                {"center": embeddings[i], "gap_score": 1.0, "nearest_covered_distance": 1.0}
                for i in range(min(5, len(embeddings)))
            ]

        centroids = []
        for label in sorted(unique_labels):
            mask = labels == label
            c = embeddings[mask].mean(axis=0)
            c /= max(np.linalg.norm(c), 1e-8)
            centroids.append(c)
        centroids_arr = np.vstack(centroids).astype(np.float32)

        # For each embedding, distance to nearest centroid
        cent_index = faiss.IndexFlatIP(centroids_arr.shape[1])
        cent_index.add(centroids_arr)
        sims, _ = cent_index.search(embeddings, 1)
        distances = 1.0 - sims[:, 0]

        # Top sparse points
        order = np.argsort(distances)[::-1]
        gaps = []
        for idx in order[: min(10, len(order))]:
            gaps.append(
                {
                    "center": embeddings[idx],
                    "gap_score": float(distances[idx]),
                    "nearest_covered_distance": float(distances[idx]),
                }
            )
        return gaps

    # ------------------------------------------------------------------
    # Overlap
    # ------------------------------------------------------------------

    def compute_overlap(self, index_a: EmbeddingIndex, index_b: EmbeddingIndex) -> float:
        """Compute distribution overlap between two datasets (0-1)."""
        import faiss

        embs_a = _reconstruct_embeddings(index_a)
        embs_b = _reconstruct_embeddings(index_b)

        if len(embs_a) == 0 or len(embs_b) == 0:
            return 0.0

        # A→B: for each point in A, find NN in B
        idx_b = faiss.IndexFlatIP(embs_b.shape[1])
        idx_b.add(embs_b)
        sims_ab, _ = idx_b.search(embs_a, 1)

        # Calibrate threshold from intra-dataset distances in A
        idx_a = faiss.IndexFlatIP(embs_a.shape[1])
        idx_a.add(embs_a)
        k_intra = min(2, len(embs_a))
        intra_sims, _ = idx_a.search(embs_a, k_intra)
        if k_intra > 1:
            threshold = float(np.percentile(intra_sims[:, 1], 10))
        else:
            threshold = 0.5

        overlap = float(np.mean(sims_ab[:, 0] >= threshold))
        return float(np.clip(overlap, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """Cluster embeddings with HDBSCAN, falling back to KMeans."""
        n = len(embeddings)
        if n < self.min_cluster_size:
            return np.zeros(n, dtype=int)

        try:
            import hdbscan

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=max(1, self.min_cluster_size // 2),
                metric="euclidean",
            )
            labels = clusterer.fit_predict(embeddings)
            n_clusters = len(set(labels) - {-1})
            if n_clusters > 0:
                return labels
            logger.info("HDBSCAN found 0 clusters, falling back to KMeans")
        except ImportError:
            logger.info("HDBSCAN not available, using KMeans")

        from sklearn.cluster import KMeans

        k = min(self.n_clusters, n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            km = KMeans(n_clusters=k, n_init="auto", random_state=42)
            return km.fit_predict(embeddings)

    def _try_umap(self, embeddings: np.ndarray) -> np.ndarray | None:
        """Try to compute UMAP 2-D projection; return None on failure."""
        try:
            import umap

            n_neighbors = min(15, len(embeddings) - 1)
            if n_neighbors < 2:
                return None
            reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, n_components=2)
            return reducer.fit_transform(embeddings).astype(np.float32)
        except Exception:
            return None
