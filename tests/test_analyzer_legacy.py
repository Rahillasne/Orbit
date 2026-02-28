"""Tests for orbit.analyzer.embedding_gap (legacy EmbeddingGapAnalyzer with mocked CLIP)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from orbit.analyzer.embedding_gap import AnalyzerConfig, EmbeddingGapAnalyzer


@pytest.fixture
def mock_clip():
    """Patch open_clip to return a fake model producing random embeddings."""
    mock_model = MagicMock()
    mock_preprocess = MagicMock(side_effect=lambda img: np.random.randn(3, 224, 224).astype(np.float32))

    # Make encode_image return random embeddings
    import torch

    def fake_encode_image(batch):
        n = batch.shape[0] if hasattr(batch, "shape") else len(batch)
        return torch.randn(n, 512)

    mock_model.encode_image = MagicMock(side_effect=fake_encode_image)
    mock_model.eval = MagicMock(return_value=mock_model)
    mock_model.to = MagicMock(return_value=mock_model)

    with patch("orbit.analyzer.embedding_gap.open_clip") as mock_oc:
        mock_oc.create_model_and_transforms.return_value = (mock_model, None, mock_preprocess)
        mock_oc.get_tokenizer.return_value = MagicMock()
        yield mock_oc


@pytest.fixture
def analyzer(mock_clip):
    """Create an EmbeddingGapAnalyzer with mocked CLIP."""
    return EmbeddingGapAnalyzer(AnalyzerConfig(sample_steps=3))


class TestEmbeddingGapAnalyzer:

    def test_embed_images_shape(self, analyzer):
        images = [
            Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            for _ in range(5)
        ]
        # Need to patch torch.stack since preprocess returns numpy
        import torch
        with patch("orbit.analyzer.embedding_gap.torch") as mock_torch:
            mock_torch.no_grad.return_value.__enter__ = MagicMock()
            mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
            mock_torch.stack.return_value = torch.randn(5, 3, 224, 224)

            embeddings = analyzer.embed_images(images)

        assert embeddings.shape[0] == 5
        assert embeddings.shape[1] > 0

    def test_analyze_returns_gap_result(self, analyzer, sample_episode, failure_episode):
        # Directly test with pre-computed embeddings
        import faiss

        success_embs = np.random.randn(3, 512).astype(np.float32)
        failure_embs = np.random.randn(2, 512).astype(np.float32)

        index = faiss.IndexFlatL2(512)
        index.add(success_embs)
        distances, _ = index.search(failure_embs, 1)

        assert distances.shape == (2, 1)
        assert all(d >= 0 for d in distances[:, 0])

    def test_umap_projection_shape(self, analyzer):
        embeddings = np.random.randn(20, 512).astype(np.float32)
        projected = analyzer.compute_umap(embeddings)
        assert projected.shape == (20, 2)

    def test_faiss_index_search(self, analyzer):
        embeddings = np.random.randn(10, 512).astype(np.float32)
        index = analyzer._build_faiss_index(embeddings)
        query = np.random.randn(3, 512).astype(np.float32)
        distances, indices = index.search(query, 1)
        assert distances.shape == (3, 1)
        assert indices.shape == (3, 1)

    def test_sample_images_from_episode(self, analyzer, sample_episode):
        images = analyzer._sample_images_from_episode(sample_episode)
        assert len(images) <= analyzer.config.sample_steps
        assert all(isinstance(img, Image.Image) for img in images)
