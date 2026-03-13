"""R3M and Voltron embedding extractors for robotics-specific visual features.

R3M (Reusable Representations for Robotic Manipulation) uses a ResNet-50
trained on Ego4D human manipulation videos, producing 2048-dim embeddings
that are semantically meaningful for robotic tasks.

Voltron is a ViT-based robotics model with optional language conditioning.
Currently provided as a stub with fallback to random projection.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

R3M_DIM = 2048
VOLTRON_DIM = 768


class R3MEmbeddingExtractor:
    """Extract 2048-dim visual embeddings using R3M (ResNet-50).

    Uses :class:`R3MManualExtractor` which loads R3M weights directly from
    HuggingFace Hub into a standard torchvision ResNet50, avoiding the
    fragile ``r3m`` pip package.  Falls back to ImageNet ResNet50 features
    if R3M weights are unavailable.  NEVER uses random projections.
    """

    def __init__(self, device: str = "cpu", batch_size: int = 32) -> None:
        self.device = device
        self.batch_size = batch_size
        self._extractor = None
        self._loaded = False

    @property
    def dimension(self) -> int:
        return R3M_DIM

    def _load_model(self) -> None:
        """Lazy-load the R3M extractor."""
        if self._loaded:
            return
        self._loaded = True

        try:
            from orbit.embeddings.r3m_manual import R3MManualExtractor

            self._extractor = R3MManualExtractor(device=self.device)
            logger.info("Loaded R3M extractor (manual weight loading) on %s", self._extractor.device)
        except ImportError as exc:
            raise ImportError(
                "torch and torchvision are required for R3M embeddings. "
                "Install with: pip install torch torchvision"
            ) from exc

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        """Extract embeddings from a list of PIL images.

        Returns L2-normalised float32 array of shape ``(N, 2048)``.
        """
        self._load_model()
        return self._extractor.embed_images(images, batch_size=self.batch_size)


class VoltronEmbeddingExtractor:
    """Extract visual embeddings using Voltron (ViT-based robotics model).

    Voltron supports language-conditioned embeddings for task-aware features.
    Currently a stub that falls back to random projection — full implementation
    planned for a future release.
    """

    def __init__(
        self, device: str = "cpu", batch_size: int = 32, variant: str = "v-cond-base"
    ) -> None:
        self.device = device
        self.batch_size = batch_size
        self.variant = variant
        self._model = None
        self._preprocess = None
        self._use_fallback = False
        self._random_proj: np.ndarray | None = None
        self._loaded = False

    @property
    def dimension(self) -> int:
        return VOLTRON_DIM

    def _load_model(self) -> bool:
        """Lazy-load Voltron model. Returns True on success."""
        if self._loaded:
            return not self._use_fallback
        self._loaded = True

        try:
            from voltron import load

            self._model, self._preprocess = load(
                self.variant, device=self.device, freeze=True
            )
            logger.info("Loaded Voltron %s on %s", self.variant, self.device)
            return True

        except ImportError:
            logger.warning(
                "voltron-robotics package not installed — falling back to random projection. "
                "Install with: pip install voltron-robotics"
            )
            self._use_fallback = True
            return False
        except Exception as exc:
            logger.warning("Failed to load Voltron model: %s — using fallback", exc)
            self._use_fallback = True
            return False

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        """Extract embeddings from a list of PIL images.

        Returns L2-normalised float32 array of shape ``(N, 768)``.
        """
        self._load_model()

        if self._use_fallback:
            return self._fallback_embed(images)

        return self._embed_with_voltron(images)

    def _embed_with_voltron(self, images: list[Image.Image]) -> np.ndarray:
        """Batch embed images using Voltron model."""
        import torch

        all_embeddings: list[np.ndarray] = []

        for i in tqdm(
            range(0, len(images), self.batch_size), desc="Voltron embedding", leave=False
        ):
            batch_images = images[i : i + self.batch_size]
            tensors = []
            for img in batch_images:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                tensors.append(self._preprocess(img))

            batch_tensor = torch.stack(tensors).to(self.device)

            with torch.no_grad():
                embeddings = self._model(batch_tensor)

            all_embeddings.append(embeddings.cpu().numpy())

        result = np.vstack(all_embeddings).astype(np.float32)
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        return (result / np.maximum(norms, 1e-8)).astype(np.float32)

    def get_task_conditioned_embeddings(
        self, images: list[Image.Image], task_description: str
    ) -> np.ndarray:
        """Extract language-conditioned embeddings (Voltron-specific).

        Returns embeddings that are task-aware, enabling better task relevance
        scoring without a separate text encoder.
        """
        self._load_model()

        if self._use_fallback:
            return self._fallback_embed(images)

        # Voltron's language conditioning is model-specific
        # Full implementation requires voltron's tokenizer
        logger.warning(
            "Task-conditioned embeddings not yet fully implemented, "
            "using standard embeddings"
        )
        return self.embed_images(images)

    def _fallback_embed(self, images: list[Image.Image]) -> np.ndarray:
        """Random projection fallback when Voltron is unavailable."""
        flat = []
        for img in images:
            arr = np.array(img.convert("RGB").resize((32, 32))).astype(np.float32).ravel()
            flat.append(arr)
        flat_arr = np.vstack(flat)

        if self._random_proj is None:
            rng = np.random.default_rng(43)  # Different seed from R3M
            input_dim = flat_arr.shape[1]
            self._random_proj = rng.standard_normal((input_dim, VOLTRON_DIM)).astype(
                np.float32
            ) / np.sqrt(VOLTRON_DIM)

        projected = flat_arr @ self._random_proj
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        return (projected / np.maximum(norms, 1e-8)).astype(np.float32)
