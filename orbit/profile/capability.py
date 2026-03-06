"""Task capability scoring."""

from __future__ import annotations

import logging
import warnings

import numpy as np

from orbit.profile.types import (
    CapabilityScore,
    CoverageMap,
    DatasetProfile,
    EmbeddingIndex,
    QualityMetrics,
)

logger = logging.getLogger(__name__)


class CapabilityScorer:
    """Score a dataset's capability for specific target tasks.

    Uses SigLIP's text encoder to match task descriptions against the
    visual embedding index, then combines relevance, coverage breadth,
    and quality weighting into a single capability score.
    """

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        device: str = "cpu",
        top_k: int = 50,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.top_k = top_k

        self._text_encoder_mode: str | None = None
        self._model = None
        self._processor = None
        self._st_model = None
        self._random_proj = None
        self._tfidf = None
        self._tfidf_dim: int | None = None
        self._text_embedding_cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------

    def _load_text_encoder(self, target_dim: int = 768) -> None:
        """Load a text encoder with graceful fallback."""
        if self._text_encoder_mode is not None:
            return

        # Try SigLIP
        try:
            from transformers import AutoModel, AutoProcessor

            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval().to(self.device)
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._text_encoder_mode = "siglip"
            logger.info("Using SigLIP text encoder")
            return
        except Exception:
            logger.info("SigLIP unavailable, trying sentence-transformers")

        # Try sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
            st_dim = self._st_model.get_sentence_embedding_dimension()
            rng = np.random.default_rng(42)
            self._random_proj = rng.standard_normal((st_dim, target_dim)).astype(
                np.float32
            ) / np.sqrt(target_dim)
            self._text_encoder_mode = "sentence_transformers"
            logger.info("Using sentence-transformers text encoder (degraded)")
            return
        except Exception:
            logger.info("sentence-transformers unavailable, using TF-IDF")

        self._tfidf_dim = target_dim
        self._text_encoder_mode = "tfidf"
        logger.info("Using TF-IDF text encoder (degraded)")

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode task descriptions to embedding vectors.

        Returns L2-normalised float32 array of shape ``(N, dim)``.
        """
        # Check cache for single-text calls
        uncached = [(i, t) for i, t in enumerate(texts) if t not in self._text_embedding_cache]
        if not uncached and texts:
            result = np.vstack([self._text_embedding_cache[t] for t in texts])
            return result

        self._load_text_encoder()

        if self._text_encoder_mode == "siglip":
            embs = self._encode_siglip([t for _, t in uncached])
        elif self._text_encoder_mode == "sentence_transformers":
            embs = self._encode_st([t for _, t in uncached])
        else:
            embs = self._encode_tfidf([t for _, t in uncached])

        # Cache
        for idx, (_, t) in enumerate(uncached):
            self._text_embedding_cache[t] = embs[idx : idx + 1]

        return np.vstack([self._text_embedding_cache[t] for t in texts])

    def _encode_siglip(self, texts: list[str]) -> np.ndarray:
        import torch

        inputs = self._processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            text_features = self._model.get_text_features(**inputs)
            if not isinstance(text_features, torch.Tensor):
                text_features = text_features.pooler_output
        embs = text_features.cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return (embs / np.maximum(norms, 1e-8)).astype(np.float32)

    def _encode_st(self, texts: list[str]) -> np.ndarray:
        raw = self._st_model.encode(texts, convert_to_numpy=True).astype(np.float32)
        projected = raw @ self._random_proj
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        return (projected / np.maximum(norms, 1e-8)).astype(np.float32)

    def _encode_tfidf(self, texts: list[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer

        dim = self._tfidf_dim or 768
        if self._tfidf is None:
            self._tfidf = TfidfVectorizer(max_features=dim)
            self._tfidf.fit(texts)

        sparse = self._tfidf.transform(texts).toarray().astype(np.float32)
        # Pad or truncate to target dim
        if sparse.shape[1] < dim:
            sparse = np.pad(sparse, ((0, 0), (0, dim - sparse.shape[1])))
        else:
            sparse = sparse[:, :dim]
        norms = np.linalg.norm(sparse, axis=1, keepdims=True)
        return (sparse / np.maximum(norms, 1e-8)).astype(np.float32)

    # ------------------------------------------------------------------
    # Main scoring
    # ------------------------------------------------------------------

    def score_tasks(
        self,
        embedding_index: EmbeddingIndex,
        coverage: CoverageMap,
        quality: QualityMetrics,
        task_descriptions: list[str],
        task_reference_embeddings: dict[str, np.ndarray] | None = None,
    ) -> list[CapabilityScore]:
        """Score dataset capability for each target task."""
        if not task_descriptions:
            return []

        k = min(self.top_k, embedding_index.num_embeddings)
        if k == 0:
            return [
                CapabilityScore(
                    task_description=desc,
                    score=0.0,
                    confidence=0.0,
                    supporting_episodes=0,
                    action_diversity=0.0,
                    environment_diversity=0.0,
                    gap_description="Empty embedding index — no data available.",
                )
                for desc in task_descriptions
            ]

        # Encode all task descriptions at once
        if task_reference_embeddings:
            text_embs = np.vstack(
                [
                    task_reference_embeddings.get(d, self._encode_texts([d]))
                    for d in task_descriptions
                ]
            )
        else:
            text_embs = self._encode_texts(task_descriptions)

        # Collect raw top-k similarities for all tasks to enable relative scoring
        all_sims: list[np.ndarray] = []
        all_indices: list[np.ndarray] = []
        for i in range(len(task_descriptions)):
            query = text_embs[i : i + 1].astype(np.float32)
            sims, indices = embedding_index.index.search(query, k)
            all_sims.append(sims[0])
            all_indices.append(indices[0])

        # Compute relative relevance across tasks.
        # SigLIP text-image scores for robotics data are small and often
        # negative; we use a combination of absolute and relative scoring.
        max_sims = np.array([s.max() for s in all_sims])

        if len(task_descriptions) >= 2:
            # Relative scoring: softmax over max similarities amplifies
            # even small differences into a meaningful distribution.
            temperature = 0.01
            shifted = (max_sims - max_sims.max()) / temperature
            exp_sims = np.exp(shifted)
            normalized_relevances = exp_sims / exp_sims.sum()
        else:
            # Single task: fall back to absolute similarity scoring
            normalized_relevances = None  # use default logic in _compute_score

        results: list[CapabilityScore] = []
        for i, desc in enumerate(task_descriptions):
            nr = float(normalized_relevances[i]) if normalized_relevances is not None else None
            score_obj = self._compute_score(
                desc,
                all_sims[i],
                all_indices[i],
                embedding_index,
                coverage,
                quality,
                normalized_relevance=nr,
            )
            results.append(score_obj)

        return results

    def _compute_score(
        self,
        task_description: str,
        sims: np.ndarray,
        indices: np.ndarray,
        embedding_index: EmbeddingIndex,
        coverage: CoverageMap,
        quality: QualityMetrics,
        normalized_relevance: float | None = None,
    ) -> CapabilityScore:
        """Compute capability score for a single task."""
        # 1. Relevance — use relative normalization across tasks when available,
        #    fall back to absolute positive similarities
        if normalized_relevance is not None:
            relevance = normalized_relevance
        else:
            positive_sims = sims[sims > 0]
            if len(positive_sims) > 0:
                relevance = float(np.mean(positive_sims))
            else:
                relevance = 0.0
        relevance = float(np.clip(relevance, 0.0, 1.0))

        # 2. Coverage breadth
        matched_episode_ids = set()
        for idx in indices:
            if 0 <= idx < len(embedding_index.episode_ids):
                matched_episode_ids.add(embedding_index.episode_ids[idx])
        supporting_episodes = len(matched_episode_ids)

        coverage_breadth = self._compute_coverage_breadth(indices, embedding_index, coverage)

        # 3. Quality weight
        quality_weight = self._compute_quality_weight(matched_episode_ids, quality)

        # 4. Combine — gate coverage/quality by relevance so irrelevant
        #    tasks don't get inflated scores from good data quality alone
        score = float(
            np.clip(
                relevance * (0.4 + 0.3 * coverage_breadth + 0.3 * quality_weight),
                0.0,
                1.0,
            )
        )

        # 5. Confidence
        denom = max(embedding_index.num_embeddings / 10.0, 1.0)
        confidence = float(np.clip(supporting_episodes / denom, 0.0, 1.0))

        # 6. Diversity metrics
        action_diversity = self._compute_action_diversity(matched_episode_ids, embedding_index)
        env_diversity = float(np.std(sims)) if len(sims) > 1 else 0.0

        # 7. Gap description
        gap_description = None
        if score < 0.5:
            gap_description = self._generate_gap_description(
                task_description, relevance, coverage_breadth, quality_weight
            )

        return CapabilityScore(
            task_description=task_description,
            score=score,
            confidence=confidence,
            supporting_episodes=supporting_episodes,
            action_diversity=action_diversity,
            environment_diversity=env_diversity,
            gap_description=gap_description,
        )

    def _compute_coverage_breadth(
        self,
        indices: np.ndarray,
        embedding_index: EmbeddingIndex,
        coverage: CoverageMap,
    ) -> float:
        """Fraction of coverage regions spanned by matched embeddings."""
        total_regions = len(coverage.dense_regions) + len(coverage.sparse_regions)
        if total_regions == 0:
            eps = set()
            for idx in indices:
                if 0 <= idx < len(embedding_index.episode_ids):
                    eps.add(embedding_index.episode_ids[idx])
            return float(np.clip(len(eps) / 10.0, 0.0, 1.0))

        all_regions = coverage.dense_regions + coverage.sparse_regions
        region_centers = []
        for r in all_regions:
            c = r["center"]
            norm = np.linalg.norm(c)
            region_centers.append(c / max(norm, 1e-8))
        region_centers_arr = np.array(region_centers, dtype=np.float32)

        # Reconstruct matched embeddings and assign to nearest region
        hit_regions = set()
        for idx in indices:
            if idx < 0 or idx >= embedding_index.num_embeddings:
                continue
            emb = embedding_index.index.reconstruct(int(idx)).reshape(1, -1)
            dots = emb @ region_centers_arr.T
            hit_regions.add(int(np.argmax(dots[0])))

        return float(np.clip(len(hit_regions) / total_regions, 0.0, 1.0))

    def _compute_quality_weight(
        self,
        matched_episode_ids: set,
        quality: QualityMetrics,
    ) -> float:
        """Mean quality score of matched episodes."""
        if not matched_episode_ids:
            return quality.aggregate_score
        scores = [
            quality.episode_scores.get(eid, quality.aggregate_score) for eid in matched_episode_ids
        ]
        return float(np.mean(scores))

    def _compute_action_diversity(
        self,
        matched_episode_ids: set,
        embedding_index: EmbeddingIndex,
    ) -> float:
        """Entropy-based diversity of matched episodes."""
        if not matched_episode_ids or len(matched_episode_ids) <= 1:
            return 0.0
        counts = {}
        for eid in embedding_index.episode_ids:
            if eid in matched_episode_ids:
                counts[eid] = counts.get(eid, 0) + 1
        total = sum(counts.values())
        if total == 0:
            return 0.0
        probs = np.array(list(counts.values()), dtype=np.float64) / total
        entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
        max_entropy = np.log(len(counts)) if len(counts) > 1 else 1.0
        return float(np.clip(entropy / max_entropy, 0.0, 1.0))

    @staticmethod
    def _generate_gap_description(
        task: str,
        relevance: float,
        coverage_breadth: float,
        quality_weight: float,
    ) -> str:
        """Generate a human-readable gap description."""
        weakest = min(relevance, coverage_breadth, quality_weight)
        if weakest == relevance:
            return (
                f"Low visual similarity to task '{task}' — dataset may not contain relevant scenes."
            )
        if weakest == coverage_breadth:
            return (
                "Relevant frames found but concentrated in few episodes — "
                "collect more diverse demonstrations."
            )
        return "Matching episodes have low data quality — consider re-collecting demonstrations."

    # ------------------------------------------------------------------
    # Failure zone prediction
    # ------------------------------------------------------------------

    def predict_failure_zones(
        self,
        profile: DatasetProfile,
        target_environment_embeddings: np.ndarray,
    ) -> list[dict]:
        """Predict where a policy will fail in deployment.

        For each deployment embedding, finds nearest neighbour in the
        training data. Frames with distance > threshold are predicted
        failures, then clustered into zones.
        """
        target = target_environment_embeddings.astype(np.float32)
        norms = np.linalg.norm(target, axis=1, keepdims=True)
        target = target / np.maximum(norms, 1e-8)

        sims, _ = profile.embedding_index.index.search(target, 1)
        distances = 1.0 - sims[:, 0]

        threshold = 0.3
        failure_mask = distances > threshold
        failure_indices = np.where(failure_mask)[0]

        if len(failure_indices) == 0:
            return []

        failure_embs = target[failure_indices]
        failure_dists = distances[failure_indices]

        # Cluster failures
        n_clusters = min(5, len(failure_embs))
        if n_clusters < 1:
            return []

        from sklearn.cluster import KMeans

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
            labels = km.fit_predict(failure_embs)

        zones: list[dict] = []
        for c_id in range(n_clusters):
            mask = labels == c_id
            if mask.sum() == 0:
                continue
            center = km.cluster_centers_[c_id]
            severity = float(np.mean(failure_dists[mask]))
            zones.append(
                {
                    "zone_id": c_id,
                    "center": center,
                    "severity": severity,
                    "num_frames": int(mask.sum()),
                    "confidence": float(np.clip(severity, 0.0, 1.0)),
                    "description": (
                        f"Predicted failure zone {c_id}: {int(mask.sum())} frames, "
                        f"avg distance {severity:.3f} from training data."
                    ),
                }
            )

        zones.sort(key=lambda z: z["severity"], reverse=True)
        return zones

    # ------------------------------------------------------------------
    # Profile comparison
    # ------------------------------------------------------------------

    def compare_profiles(
        self,
        profile_a: DatasetProfile,
        profile_b: DatasetProfile,
    ) -> dict:
        """Compare two dataset profiles."""
        from orbit.profile.coverage import CoverageAnalyzer

        analyzer = CoverageAnalyzer()
        overlap = analyzer.compute_overlap(profile_a.embedding_index, profile_b.embedding_index)

        # Capability comparison
        caps_a = {c.task_description: c.score for c in profile_a.capabilities}
        caps_b = {c.task_description: c.score for c in profile_b.capabilities}

        unique_to_a = [t for t, s in caps_a.items() if s >= 0.5 and caps_b.get(t, 0.0) < 0.5]
        unique_to_b = [t for t, s in caps_b.items() if s >= 0.5 and caps_a.get(t, 0.0) < 0.5]

        # Combined coverage estimate
        combined = min(
            1.0,
            profile_a.coverage.overall_coverage_score
            + profile_b.coverage.overall_coverage_score * (1.0 - overlap),
        )

        return {
            "overlap": overlap,
            "unique_to_a": unique_to_a,
            "unique_to_b": unique_to_b,
            "combined_coverage": combined,
        }
