"""Tests for orbit.profile.embedding (EmbeddingExtractor)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from orbit.profile.embedding import EmbeddingExtractor
from orbit.profile.types import EmbeddingIndex


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_images(n: int, seed: int = 42) -> list[Image.Image]:
    rng = np.random.default_rng(seed)
    return [
        Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))
        for _ in range(n)
    ]


def _make_synthetic_embeddings(
    n: int, dim: int = 768, seed: int = 42
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    embs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return (embs / np.maximum(norms, 1e-8)).astype(np.float32)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_siglip():
    """Patch transformers to return a fake SigLIP model (768-dim)."""
    import torch

    mock_model = MagicMock()
    mock_processor = MagicMock()

    def fake_process(images, return_tensors="pt"):
        n = len(images) if isinstance(images, list) else 1
        return {"pixel_values": torch.randn(n, 3, 224, 224)}

    mock_processor.side_effect = fake_process

    def fake_get_image_features(**kwargs):
        n = kwargs["pixel_values"].shape[0]
        return torch.randn(n, 768)

    mock_model.get_image_features = MagicMock(side_effect=fake_get_image_features)
    mock_model.eval.return_value = mock_model
    mock_model.to.return_value = mock_model

    with (
        patch("transformers.AutoModel") as mock_auto_model,
        patch("transformers.AutoProcessor") as mock_auto_proc,
    ):
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_auto_proc.from_pretrained.return_value = mock_processor
        yield mock_model, mock_processor


@pytest.fixture
def extractor(mock_siglip):
    """EmbeddingExtractor with mocked SigLIP."""
    return EmbeddingExtractor(batch_size=8)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEmbeddingExtractor:
    def test_extract_from_synthetic(self, extractor, tmp_path):
        """Extract embeddings from synthetic images; verify shape and L2 norm."""
        images = _make_test_images(10)
        paths = []
        for i, img in enumerate(images):
            p = tmp_path / f"img_{i:04d}.png"
            img.save(p)
            paths.append(str(p))

        embeddings = extractor.extract_from_images(paths)
        assert embeddings.shape == (10, 768)
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_faiss_index_search(self, extractor):
        """Build FAISS index and verify nearest-neighbor search."""
        import faiss

        embs = _make_synthetic_embeddings(20, dim=768)
        index = extractor.build_faiss_index(embs)

        query = embs[:1].copy()
        faiss.normalize_L2(query)
        sims, ids = index.search(query, 1)

        assert ids[0, 0] == 0
        assert sims[0, 0] > 0.99

    def test_save_load_roundtrip(self, extractor, tmp_path):
        """Save and load an EmbeddingIndex; verify identical metadata."""
        import faiss

        embs = _make_synthetic_embeddings(15, dim=128)
        faiss_index = extractor.build_faiss_index(embs)
        ei = EmbeddingIndex(
            index=faiss_index,
            episode_ids=list(range(15)),
            frame_indices=list(range(15)),
            dimension=128,
            num_embeddings=15,
        )

        save_path = str(tmp_path / "test_index")
        extractor.save_index(ei, save_path)
        loaded = extractor.load_index(save_path)

        assert loaded.episode_ids == ei.episode_ids
        assert loaded.frame_indices == ei.frame_indices
        assert loaded.dimension == ei.dimension
        assert loaded.num_embeddings == ei.num_embeddings

        # Verify FAISS search gives same results
        query = embs[:1].copy()
        faiss.normalize_L2(query)
        sims_orig, ids_orig = ei.index.search(query, 3)
        sims_loaded, ids_loaded = loaded.index.search(query, 3)
        np.testing.assert_array_equal(ids_orig, ids_loaded)
        np.testing.assert_allclose(sims_orig, sims_loaded, atol=1e-6)

    def test_empty_directory(self, extractor, tmp_path):
        """Empty directory should raise ValueError."""
        with pytest.raises(ValueError, match="No HDF5 files or images"):
            extractor.extract_from_directory(str(tmp_path))

    def test_single_image(self, extractor, tmp_path):
        """Single image should work without crash."""
        img = _make_test_images(1)[0]
        p = tmp_path / "single.png"
        img.save(p)

        embeddings = extractor.extract_from_images([str(p)])
        assert embeddings.shape == (1, 768)
        norm = np.linalg.norm(embeddings[0])
        assert np.isclose(norm, 1.0, atol=1e-5)
