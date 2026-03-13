"""Lightweight embedding extractor for CPU-only environments.

Uses OpenCLIP ViT-B/32 which is significantly faster than SigLIP ViT-B/16
on CPU (fewer patches = fewer transformer tokens).  Falls back to random
projection if OpenCLIP is unavailable.
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Output dimension for OpenCLIP ViT-B/32
FAST_EMBED_DIM = 512


class FastEmbeddingExtractor:
    """Lightweight visual embedding extractor using OpenCLIP ViT-B/32.

    Approximately 2-5x faster than SigLIP ViT-B/16 on CPU with ~10-15%
    lower correlation on downstream scoring tasks.
    """

    def __init__(self, batch_size: int = 32) -> None:
        self.batch_size = batch_size
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._random_proj = None
        self._use_fallback = False

    @property
    def dimension(self) -> int:
        """Embedding output dimension."""
        return FAST_EMBED_DIM

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> bool:
        """Load OpenCLIP ViT-B/32. Returns True on success."""
        if self._model is not None:
            return True
        if self._use_fallback:
            return False

        try:
            import open_clip

            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
            model.eval()
            self._model = model
            self._preprocess = preprocess
            self._tokenizer = open_clip.get_tokenizer("ViT-B-32")
            logger.info("Loaded OpenCLIP ViT-B/32 (fast mode)")
            return True
        except Exception:
            logger.warning("OpenCLIP unavailable, falling back to random projection")
            self._use_fallback = True
            return False

    # ------------------------------------------------------------------
    # Image embedding
    # ------------------------------------------------------------------

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        """Embed images with progress bar.

        Returns ``(N, 512)`` float32 array with L2-normalized embeddings.
        """
        if self._load_model():
            return self._embed_openclip(images)
        return self._fallback_embed(images)

    def _embed_openclip(self, images: list[Image.Image]) -> np.ndarray:
        import torch

        all_embeddings: list[np.ndarray] = []
        for i in tqdm(
            range(0, len(images), self.batch_size),
            desc="Embedding images (fast)",
            disable=len(images) <= self.batch_size,
        ):
            batch = images[i : i + self.batch_size]
            tensors = torch.stack([self._preprocess(img) for img in batch])
            with torch.no_grad():
                embs = self._model.encode_image(tensors)
            embs_np = embs.cpu().numpy().astype(np.float32)
            norms = np.linalg.norm(embs_np, axis=1, keepdims=True)
            embs_np = embs_np / np.maximum(norms, 1e-8)
            all_embeddings.append(embs_np)

        return np.vstack(all_embeddings).astype(np.float32)

    def _fallback_embed(self, images: list[Image.Image]) -> np.ndarray:
        """Random projection fallback when OpenCLIP is unavailable."""
        dim = FAST_EMBED_DIM
        flat: list[np.ndarray] = []
        for img in images:
            arr = np.array(img.resize((32, 32))).astype(np.float32).ravel()
            flat.append(arr)
        flat_arr = np.vstack(flat)

        if self._random_proj is None:
            rng = np.random.default_rng(42)
            input_dim = flat_arr.shape[1]
            self._random_proj = rng.standard_normal((input_dim, dim)).astype(
                np.float32
            ) / np.sqrt(dim)

        projected = flat_arr @ self._random_proj
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        return (projected / np.maximum(norms, 1e-8)).astype(np.float32)

    # ------------------------------------------------------------------
    # Text embedding
    # ------------------------------------------------------------------

    def embed_text(self, texts: list[str]) -> np.ndarray:
        """Embed text queries for capability scoring.

        Returns ``(N, 512)`` float32 array with L2-normalized embeddings.
        """
        if self._load_model():
            return self._embed_text_openclip(texts)
        return self._fallback_text_embed(texts)

    def _embed_text_openclip(self, texts: list[str]) -> np.ndarray:
        import torch

        tokens = self._tokenizer(texts)
        with torch.no_grad():
            text_features = self._model.encode_text(tokens)
        embs = text_features.cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return (embs / np.maximum(norms, 1e-8)).astype(np.float32)

    def _fallback_text_embed(self, texts: list[str]) -> np.ndarray:
        """Simple hash-based text embedding fallback."""
        dim = FAST_EMBED_DIM
        result = []
        for text in texts:
            # Use text hash as seed for reproducible random embedding
            seed = hash(text) % (2**31)
            text_rng = np.random.default_rng(seed)
            vec = text_rng.standard_normal(dim).astype(np.float32)
            vec /= np.linalg.norm(vec) + 1e-8
            result.append(vec)
        return np.array(result, dtype=np.float32)
