"""Tests for R3M and Voltron embedding extractors.

R3M tests use the R3MManualExtractor which loads weights directly
from torchvision (ImageNet fallback if R3M weights unavailable).
Voltron tests still exercise the random-projection fallback path.
"""

from __future__ import annotations

import numpy as np
import pytest  # noqa: F401
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_images(n: int = 5, size: int = 64) -> list[Image.Image]:
    """Create synthetic RGB images."""
    rng = np.random.default_rng(42)
    return [
        Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))
        for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# R3MEmbeddingExtractor — uses R3MManualExtractor (no r3m pip package needed)
# ---------------------------------------------------------------------------


class TestR3MExtractor:
    """Tests using the R3MManualExtractor backend (ImageNet ResNet50 fallback)."""

    def test_embedding_shape(self):
        from orbit.embeddings.r3m_embeddings import R3MEmbeddingExtractor

        ext = R3MEmbeddingExtractor(device="cpu")
        images = _make_images(4)
        embs = ext.embed_images(images)
        assert embs.shape == (4, 2048)

    def test_embedding_normalized(self):
        from orbit.embeddings.r3m_embeddings import R3MEmbeddingExtractor

        ext = R3MEmbeddingExtractor(device="cpu")
        images = _make_images(3)
        embs = ext.embed_images(images)

        norms = np.linalg.norm(embs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_dimension_property(self):
        from orbit.embeddings.r3m_embeddings import R3MEmbeddingExtractor

        ext = R3MEmbeddingExtractor()
        assert ext.dimension == 2048

    def test_single_image(self):
        from orbit.embeddings.r3m_embeddings import R3MEmbeddingExtractor

        ext = R3MEmbeddingExtractor(device="cpu")
        images = _make_images(1)
        embs = ext.embed_images(images)
        assert embs.shape == (1, 2048)

    def test_batch_consistency(self):
        """Embeddings should be the same regardless of batch size."""
        from orbit.embeddings.r3m_manual import R3MManualExtractor

        images = _make_images(6)

        # Use same extractor instance to ensure same model weights
        ext = R3MManualExtractor(device="cpu")
        embs_small = ext.embed_images(images, batch_size=2)
        embs_large = ext.embed_images(images, batch_size=32)

        np.testing.assert_array_almost_equal(embs_small, embs_large)

    def test_reproducible(self):
        """Same extractor instance produces same embeddings for same input."""
        from orbit.embeddings.r3m_embeddings import R3MEmbeddingExtractor

        images = _make_images(2)

        ext = R3MEmbeddingExtractor(device="cpu")
        embs1 = ext.embed_images(images)
        embs2 = ext.embed_images(images)

        np.testing.assert_array_almost_equal(embs1, embs2)


# ---------------------------------------------------------------------------
# R3MManualExtractor — direct tests
# ---------------------------------------------------------------------------


class TestR3MManualExtractor:
    def test_shape_and_norm(self):
        from orbit.embeddings.r3m_manual import R3MManualExtractor

        ext = R3MManualExtractor(device="cpu")
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        emb = ext.embed_images([img])
        assert emb.shape == (1, 2048)
        np.testing.assert_allclose(np.linalg.norm(emb[0]), 1.0, atol=1e-5)

    def test_numpy_input(self):
        from orbit.embeddings.r3m_manual import R3MManualExtractor

        ext = R3MManualExtractor(device="cpu")
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        emb = ext.embed_images([arr])
        assert emb.shape == (1, 2048)

    def test_empty_input(self):
        from orbit.embeddings.r3m_manual import R3MManualExtractor

        ext = R3MManualExtractor(device="cpu")
        emb = ext.embed_images([])
        assert emb.shape == (0, 2048)

    def test_dimension_property(self):
        from orbit.embeddings.r3m_manual import R3MManualExtractor

        ext = R3MManualExtractor(device="cpu")
        assert ext.dimension == 2048


# ---------------------------------------------------------------------------
# VoltronEmbeddingExtractor — fallback mode
# ---------------------------------------------------------------------------


class TestVoltronFallback:
    def test_fallback_embedding_shape(self):
        from orbit.embeddings.r3m_embeddings import VoltronEmbeddingExtractor

        ext = VoltronEmbeddingExtractor(device="cpu")
        ext._use_fallback = True
        ext._loaded = True

        images = _make_images(3)
        embs = ext.embed_images(images)
        assert embs.shape == (3, 768)

    def test_dimension_property(self):
        from orbit.embeddings.r3m_embeddings import VoltronEmbeddingExtractor

        ext = VoltronEmbeddingExtractor()
        assert ext.dimension == 768

    def test_task_conditioned_fallback(self):
        from orbit.embeddings.r3m_embeddings import VoltronEmbeddingExtractor

        ext = VoltronEmbeddingExtractor(device="cpu")
        ext._use_fallback = True
        ext._loaded = True

        images = _make_images(2)
        embs = ext.get_task_conditioned_embeddings(images, "pick up cup")
        assert embs.shape == (2, 768)


# ---------------------------------------------------------------------------
# ImageEmbedder protocol & factory
# ---------------------------------------------------------------------------


class TestImageEmbedderProtocol:
    def test_r3m_conforms_to_protocol(self):
        from orbit.embeddings import ImageEmbedder
        from orbit.embeddings.r3m_embeddings import R3MEmbeddingExtractor

        ext = R3MEmbeddingExtractor()
        assert isinstance(ext, ImageEmbedder)

    def test_voltron_conforms_to_protocol(self):
        from orbit.embeddings import ImageEmbedder
        from orbit.embeddings.r3m_embeddings import VoltronEmbeddingExtractor

        ext = VoltronEmbeddingExtractor()
        assert isinstance(ext, ImageEmbedder)

    def test_get_extractor_r3m(self):
        from orbit.embeddings import get_extractor

        ext = get_extractor("r3m", device="cpu")
        assert ext.dimension == 2048

    def test_get_extractor_openclip(self):
        from orbit.embeddings import get_extractor

        ext = get_extractor("openclip", device="cpu")
        assert ext.dimension == 512

    def test_get_extractor_unknown_raises(self):
        from orbit.embeddings import get_extractor

        with pytest.raises(ValueError, match="Unknown embedding model"):
            get_extractor("nonexistent")


# ---------------------------------------------------------------------------
# EmbeddingExtractor integration (R3M mode)
# ---------------------------------------------------------------------------


class TestEmbeddingExtractorR3MMode:
    def test_r3m_mode_flag_stored(self):
        from orbit.profile.embedding import EmbeddingExtractor

        ext = EmbeddingExtractor(embedding_model="r3m")
        assert ext.embedding_model == "r3m"

    def test_auto_resolves(self):
        """Auto mode should resolve to a concrete model name."""
        from orbit.profile.embedding import EmbeddingExtractor

        ext = EmbeddingExtractor(embedding_model="auto")
        # Should not remain "auto" after __init__
        # (either resolved in __init__ or fast_mode was set)
        assert ext.embedding_model != "auto" or ext.fast_mode

    def test_r3m_produces_2048_dim(self):
        """R3M extractor should produce 2048-dim embeddings."""
        from orbit.profile.embedding import EmbeddingExtractor

        ext = EmbeddingExtractor(embedding_model="r3m")
        r3m_ext = ext._get_r3m_extractor()

        images = _make_images(2)
        embs = r3m_ext.embed_images(images)
        assert embs.shape[1] == 2048


# ---------------------------------------------------------------------------
# CapabilityScorer with relevance_index
# ---------------------------------------------------------------------------


class TestHybridScoring:
    def test_relevance_index_used_when_provided(self):
        """When relevance_index is provided, it should be used for text search."""
        from orbit.profile.capability import CapabilityScorer
        from orbit.profile.types import (
            CapabilityScore,
            CoverageMap,
            EmbeddingIndex,
            QualityMetrics,
            QualitySignalBreakdown,
        )

        DIM = 64
        rng = np.random.default_rng(42)

        # Primary index (simulating R3M 2048-dim, but using 64 for test)
        primary_embs = rng.standard_normal((20, DIM)).astype(np.float32)
        norms = np.linalg.norm(primary_embs, axis=1, keepdims=True)
        primary_embs = primary_embs / np.maximum(norms, 1e-8)

        import faiss

        primary_idx = faiss.IndexFlatIP(DIM)
        primary_idx.add(primary_embs)
        primary_index = EmbeddingIndex(
            index=primary_idx,
            episode_ids=list(range(20)),
            frame_indices=list(range(20)),
            dimension=DIM,
            num_embeddings=20,
        )

        # Relevance index (simulating SigLIP, same dim for test simplicity)
        rel_embs = rng.standard_normal((20, DIM)).astype(np.float32)
        norms = np.linalg.norm(rel_embs, axis=1, keepdims=True)
        rel_embs = rel_embs / np.maximum(norms, 1e-8)

        rel_idx = faiss.IndexFlatIP(DIM)
        rel_idx.add(rel_embs)
        relevance_index = EmbeddingIndex(
            index=rel_idx,
            episode_ids=list(range(20)),
            frame_indices=list(range(20)),
            dimension=DIM,
            num_embeddings=20,
        )

        coverage = CoverageMap(
            dense_regions=[], sparse_regions=[],
            overall_coverage_score=0.5, umap_projection=None,
        )
        quality = QualityMetrics(
            episode_scores={i: 0.7 for i in range(20)},
            aggregate_score=0.7,
            low_quality_episodes=[],
            mutual_information_estimate=0.5,
            signal_breakdown=QualitySignalBreakdown(
                mutual_information=0.5,
                action_smoothness=0.7,
                episode_completion=0.6,
                observation_consistency=0.8,
                demonstration_quality=0.7,
            ),
        )

        scorer = CapabilityScorer(fast_mode=True)
        # Mock the text encoder to return vectors of matching dim
        scorer._text_encoder_mode = "tfidf"
        scorer._tfidf_dim = DIM

        results = scorer.score_tasks(
            primary_index, coverage, quality,
            ["pick up cube"],
            relevance_index=relevance_index,
        )

        assert len(results) == 1
        assert isinstance(results[0], CapabilityScore)
        assert 0.0 <= results[0].score <= 1.0
