"""Tests for orbit.embeddings.fast_embeddings (FastEmbeddingExtractor)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
from PIL import Image

from orbit.embeddings.fast_embeddings import FAST_EMBED_DIM, FastEmbeddingExtractor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dummy_images(n: int = 5, size: tuple = (64, 64)) -> list[Image.Image]:
    """Create simple dummy PIL images."""
    rng = np.random.default_rng(42)
    images = []
    for _ in range(n):
        arr = rng.integers(0, 255, (*size, 3), dtype=np.uint8)
        images.append(Image.fromarray(arr, "RGB"))
    return images


# ---------------------------------------------------------------------------
# Tests — fallback mode (no OpenCLIP)
# ---------------------------------------------------------------------------


class TestFastEmbeddingFallback:
    def test_fallback_embedding_shape(self):
        """Fallback embeddings should have shape (N, 512)."""
        extractor = FastEmbeddingExtractor()
        extractor._use_fallback = True  # force fallback
        images = _make_dummy_images(5)
        embs = extractor.embed_images(images)
        assert embs.shape == (5, FAST_EMBED_DIM)

    def test_fallback_embedding_normalized(self):
        """Fallback embeddings should be L2-normalized."""
        extractor = FastEmbeddingExtractor()
        extractor._use_fallback = True
        images = _make_dummy_images(10)
        embs = extractor.embed_images(images)
        norms = np.linalg.norm(embs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_fallback_text_embedding_shape(self):
        """Fallback text embeddings should have shape (N, 512)."""
        extractor = FastEmbeddingExtractor()
        extractor._use_fallback = True
        embs = extractor.embed_text(["pick up cube", "navigate terrain"])
        assert embs.shape == (2, FAST_EMBED_DIM)

    def test_fallback_text_embedding_reproducible(self):
        """Same text should produce same embedding."""
        extractor = FastEmbeddingExtractor()
        extractor._use_fallback = True
        embs1 = extractor.embed_text(["pick up cube"])
        embs2 = extractor.embed_text(["pick up cube"])
        np.testing.assert_array_equal(embs1, embs2)

    def test_dimension_property(self):
        """Dimension should be 512."""
        extractor = FastEmbeddingExtractor()
        assert extractor.dimension == FAST_EMBED_DIM


# ---------------------------------------------------------------------------
# Tests — EmbeddingExtractor fast mode integration
# ---------------------------------------------------------------------------


class TestEmbeddingExtractorFastMode:
    def test_fast_mode_flag_stored(self):
        """Fast mode flag should be stored on the extractor."""
        from orbit.profile.embedding import EmbeddingExtractor

        # Force fast_mode=True and prevent auto-detection from changing it
        extractor = EmbeddingExtractor(fast_mode=True)
        assert extractor.fast_mode is True

    def test_fast_extractor_created(self):
        """Fast mode should create a FastEmbeddingExtractor."""
        from orbit.profile.embedding import EmbeddingExtractor

        extractor = EmbeddingExtractor(fast_mode=True)
        fast = extractor._get_fast_extractor()
        assert isinstance(fast, FastEmbeddingExtractor)

    @patch("orbit.profile.embedding.EmbeddingExtractor._get_analyzer", return_value=None)
    def test_fast_mode_skips_siglip(self, mock_analyzer):
        """In fast mode, should not attempt to load SigLIP."""
        from orbit.profile.embedding import EmbeddingExtractor

        extractor = EmbeddingExtractor(fast_mode=True)
        # Force fallback in the fast extractor too
        fast = extractor._get_fast_extractor()
        fast._use_fallback = True

        images = _make_dummy_images(3)
        embs = extractor._embed_images(images)
        assert embs.shape[0] == 3
        assert embs.shape[1] == FAST_EMBED_DIM
        mock_analyzer.assert_not_called()


# ---------------------------------------------------------------------------
# Tests — GPU auto-detection
# ---------------------------------------------------------------------------


class TestGPUAutoDetection:
    def test_no_gpu_switches_to_fast(self):
        """When no GPU is detected on CPU device, should auto-switch to fast mode."""
        import torch as real_torch

        with (
            patch.object(real_torch.cuda, "is_available", return_value=False),
            patch.object(real_torch.backends.mps, "is_available", return_value=False),
        ):
            from orbit.profile.embedding import EmbeddingExtractor

            extractor = EmbeddingExtractor(device="cpu", fast_mode=False)
            assert extractor.fast_mode is True
