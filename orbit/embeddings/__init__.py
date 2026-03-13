"""Embedding extractors for visual feature extraction.

Provides a common protocol and factory for selecting between embedding models:
R3M (robotics-specific), SigLIP (vision-language), OpenCLIP (fast CPU), and Voltron.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@runtime_checkable
class ImageEmbedder(Protocol):
    """Common interface for all image embedding extractors."""

    @property
    def dimension(self) -> int:
        """Dimensionality of the output embeddings."""
        ...

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        """Embed a list of PIL images.

        Returns L2-normalised float32 array of shape ``(N, dimension)``.
        """
        ...


def get_extractor(
    model: str,
    device: str = "cpu",
    batch_size: int = 32,
) -> ImageEmbedder:
    """Factory function to create an embedding extractor.

    Parameters
    ----------
    model:
        One of ``"r3m"``, ``"siglip"``, ``"openclip"``, ``"voltron"``.
    device:
        Torch device (``"cpu"``, ``"cuda"``, ``"mps"``).
    batch_size:
        Batch size for embedding extraction.

    Returns
    -------
    ImageEmbedder
        An embedding extractor conforming to the :class:`ImageEmbedder` protocol.
    """
    if model == "r3m":
        from orbit.embeddings.r3m_embeddings import R3MEmbeddingExtractor

        return R3MEmbeddingExtractor(device=device, batch_size=batch_size)

    if model == "openclip":
        from orbit.embeddings.fast_embeddings import FastEmbeddingExtractor

        return FastEmbeddingExtractor(batch_size=batch_size)

    if model == "siglip":
        return _SigLIPWrapper(device=device, batch_size=batch_size)

    if model == "voltron":
        from orbit.embeddings.r3m_embeddings import VoltronEmbeddingExtractor

        return VoltronEmbeddingExtractor(device=device, batch_size=batch_size)

    raise ValueError(
        f"Unknown embedding model: {model!r}. "
        f"Choose from: 'r3m', 'siglip', 'openclip', 'voltron'."
    )


class _SigLIPWrapper:
    """Wraps EmbeddingAnalyzer to conform to the ImageEmbedder protocol."""

    def __init__(self, device: str = "cpu", batch_size: int = 32) -> None:
        self._device = device
        self._batch_size = batch_size
        self._analyzer = None

    @property
    def dimension(self) -> int:
        return 768

    def _get_analyzer(self):
        if self._analyzer is not None:
            return self._analyzer
        try:
            from orbit.analyzer.embedding_analyzer import EmbeddingAnalyzer
            from orbit.analyzer.models import EmbeddingAnalyzerConfig

            config = EmbeddingAnalyzerConfig(
                model_name="google/siglip-base-patch16-224",
                device=self._device,
                batch_size=self._batch_size,
            )
            self._analyzer = EmbeddingAnalyzer(config)
            return self._analyzer
        except Exception:
            return None

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        analyzer = self._get_analyzer()
        if analyzer is not None:
            return analyzer.embed_images(images)
        # Fallback to random projection (768-dim)
        flat = []
        for img in images:
            arr = np.array(img.convert("RGB").resize((32, 32))).astype(np.float32).ravel()
            flat.append(arr)
        flat_arr = np.vstack(flat)
        rng = np.random.default_rng(42)
        proj = rng.standard_normal((flat_arr.shape[1], 768)).astype(np.float32) / np.sqrt(768)
        projected = flat_arr @ proj
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        return (projected / np.maximum(norms, 1e-8)).astype(np.float32)
