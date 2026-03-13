"""Embedding extraction and FAISS index construction."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# torch must be imported before faiss on macOS to avoid SIGSEGV
try:
    import torch as _torch  # noqa: F401
except ImportError:
    pass

from orbit.profile.types import EmbeddingIndex

logger = logging.getLogger(__name__)

# Mapping from short names to HuggingFace model IDs (for backward compat)
_MODEL_ALIASES: dict[str, str] = {
    "siglip": "google/siglip-base-patch16-224",
}


class EmbeddingExtractor:
    """Extract visual embeddings and build FAISS indexes for similarity search.

    Supports multiple embedding backends:

    * ``"r3m"`` — R3M ResNet-50 (2048-dim, robotics-specific)
    * ``"siglip"`` — SigLIP ViT-B/16 (768-dim, vision-language)
    * ``"openclip"`` — OpenCLIP ViT-B/32 (512-dim, fast CPU)
    * ``"voltron"`` — Voltron ViT (768-dim, robotics, stub)
    * ``"auto"`` — tries R3M → SigLIP → OpenCLIP → random projection

    When ``fast_mode=True``, always uses OpenCLIP regardless of
    ``embedding_model``.
    """

    KNOWN_MODELS = ("auto", "r3m", "siglip", "openclip", "voltron")

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        device: str = "cpu",
        batch_size: int = 32,
        fast_mode: bool = False,
        embedding_model: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.fast_mode = fast_mode
        self.embedding_model = embedding_model
        self._analyzer = None
        self._fast_extractor = None
        self._r3m_extractor = None
        self._use_fallback = False
        self._random_proj = None
        self._fallback_dim: int = 768  # dynamic based on resolved model

        # If model_name looks like a HuggingFace ID and embedding_model is auto,
        # preserve backward compatibility
        if embedding_model == "auto" and model_name != "google/siglip-base-patch16-224":
            # User explicitly passed a HF model name — treat as SigLIP
            self.embedding_model = "siglip"

        # Auto-detect GPU and switch to fast mode if needed
        if not fast_mode and device == "cpu":
            gpu_available = False
            try:
                import torch

                gpu_available = torch.cuda.is_available() or (
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                )
            except ImportError:
                pass

            if not gpu_available:
                logger.warning(
                    "No GPU detected — switching to fast embedding mode (OpenCLIP ViT-B/32). "
                    "Use --fast explicitly to suppress this warning."
                )
                self.fast_mode = True

        # Resolve "auto" to a concrete model
        if self.embedding_model == "auto" and not self.fast_mode:
            self.embedding_model = self._resolve_auto()

    def _resolve_auto(self) -> str:
        """Resolve 'auto' to the best available model."""
        # Prefer R3M for robotics-specific embeddings
        try:
            import r3m  # noqa: F401

            logger.info("Auto-detected r3m package — using R3M embeddings")
            return "r3m"
        except ImportError:
            pass

        # Fall back to SigLIP
        try:
            import transformers  # noqa: F401

            logger.info("R3M not available — using SigLIP embeddings")
            return "siglip"
        except ImportError:
            pass

        # Last resort: OpenCLIP / fast mode
        logger.info("SigLIP not available — using OpenCLIP (fast mode)")
        return "openclip"

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _get_analyzer(self):
        """Lazily create the underlying EmbeddingAnalyzer."""
        if self._analyzer is not None:
            return self._analyzer
        if self._use_fallback:
            return None
        try:
            from orbit.analyzer.embedding_analyzer import EmbeddingAnalyzer
            from orbit.analyzer.models import EmbeddingAnalyzerConfig

            config = EmbeddingAnalyzerConfig(
                model_name=self.model_name,
                device=self.device,
                batch_size=self.batch_size,
            )
            self._analyzer = EmbeddingAnalyzer(config)
            return self._analyzer
        except Exception:
            logger.warning("SigLIP not available, falling back to random projection")
            self._use_fallback = True
            return None

    def _get_fast_extractor(self):
        """Lazily create the fast embedding extractor."""
        if self._fast_extractor is not None:
            return self._fast_extractor

        from orbit.embeddings.fast_embeddings import FastEmbeddingExtractor

        self._fast_extractor = FastEmbeddingExtractor(batch_size=self.batch_size)
        return self._fast_extractor

    def _get_r3m_extractor(self):
        """Lazily create the R3M embedding extractor."""
        if self._r3m_extractor is not None:
            return self._r3m_extractor

        from orbit.embeddings.r3m_embeddings import R3MEmbeddingExtractor

        self._r3m_extractor = R3MEmbeddingExtractor(
            device=self.device, batch_size=self.batch_size
        )
        return self._r3m_extractor

    def _embed_images(self, images: list[Image.Image]) -> np.ndarray:
        """Embed images using the configured model backend."""
        # Fast mode always uses OpenCLIP
        if self.fast_mode:
            extractor = self._get_fast_extractor()
            return extractor.embed_images(images)

        model = self.embedding_model

        if model == "r3m":
            extractor = self._get_r3m_extractor()
            result = extractor.embed_images(images)
            self._fallback_dim = extractor.dimension
            return result

        if model == "openclip":
            extractor = self._get_fast_extractor()
            return extractor.embed_images(images)

        if model == "voltron":
            from orbit.embeddings.r3m_embeddings import VoltronEmbeddingExtractor

            voltron = VoltronEmbeddingExtractor(
                device=self.device, batch_size=self.batch_size
            )
            result = voltron.embed_images(images)
            self._fallback_dim = voltron.dimension
            return result

        # Default: SigLIP
        analyzer = self._get_analyzer()
        if analyzer is not None:
            return analyzer.embed_images(images)
        return self._fallback_embed(images)

    def _fallback_embed(self, images: list[Image.Image]) -> np.ndarray:
        """Random projection fallback when no model is available."""
        dim = self._fallback_dim
        flat = []
        for img in images:
            arr = np.array(img.resize((32, 32))).astype(np.float32).ravel()
            flat.append(arr)
        flat_arr = np.vstack(flat)

        if self._random_proj is None:
            rng = np.random.default_rng(42)
            input_dim = flat_arr.shape[1]
            self._random_proj = rng.standard_normal((input_dim, dim)).astype(np.float32) / np.sqrt(
                dim
            )

        projected = flat_arr @ self._random_proj
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        return (projected / np.maximum(norms, 1e-8)).astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_from_directory(self, data_dir: str, sample_rate: int = 1) -> EmbeddingIndex:
        """Extract embeddings from all episodes in a directory.

        Supports HDF5 session files (``session_*.h5``) and plain image
        directories.  *sample_rate* controls frame sub-sampling (1 = every
        frame, 5 = every 5th frame).
        """

        data_path = Path(data_dir)
        if not data_path.is_dir():
            raise ValueError(f"Not a directory: {data_dir}")

        h5_files = sorted(data_path.glob("session_*.h5"))
        if not h5_files:
            h5_files = sorted(data_path.glob("*.h5"))

        if h5_files:
            return self._extract_from_hdf5(h5_files, sample_rate)

        # Fall back to image directory
        extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        image_paths: list[Path] = []
        for ext in extensions:
            image_paths.extend(sorted(data_path.glob(ext)))

        if not image_paths:
            raise ValueError(f"No HDF5 files or images found in {data_dir}")

        images = [Image.open(p).convert("RGB") for p in image_paths]
        embeddings = self._embed_images(images)

        index = self.build_faiss_index(embeddings)
        return EmbeddingIndex(
            index=index,
            episode_ids=[0] * len(images),
            frame_indices=list(range(len(images))),
            dimension=embeddings.shape[1],
            num_embeddings=len(embeddings),
        )

    def _extract_from_hdf5(
        self,
        h5_files: list[Path],
        sample_rate: int,
        max_images: int = 5000,
    ) -> EmbeddingIndex:
        """Load images from HDF5 files and extract embeddings.

        Supports two HDF5 formats:

        * **New** — ``/episodes/{id}/images`` dataset (N×H×W×3 uint8 arrays
          embedded directly).  Self-contained, no filesystem dependencies.
        * **Legacy** — ``/episodes/{id}/image_paths`` (file path strings).

        *max_images* caps the total number of images loaded to prevent OOM.
        """
        import h5py
        import os

        all_images: list[Image.Image] = []
        episode_ids: list[int] = []
        frame_indices: list[int] = []

        total_eps = 0
        file_episode_map: list[tuple[Path, list[str]]] = []
        for h5_path in h5_files:
            with h5py.File(h5_path, "r") as f:
                if "episodes" not in f:
                    continue
                ep_keys = list(f["episodes"].keys())
                file_episode_map.append((h5_path, ep_keys))
                total_eps += len(ep_keys)

        if total_eps == 0:
            raise ValueError("No episodes found in HDF5 files")

        for h5_path, ep_keys in file_episode_map:
            with h5py.File(h5_path, "r") as f:
                for ep_key in tqdm(
                    ep_keys,
                    desc=f"Loading episodes from {h5_path.name}",
                    leave=False,
                ):
                    grp = f["episodes"][ep_key]

                    if "images" in grp:
                        # New format: numpy arrays embedded in HDF5
                        img_dataset = grp["images"]
                        sampled_indices = list(range(0, len(img_dataset), sample_rate))
                        for idx in sampled_indices:
                            if len(all_images) >= max_images:
                                break
                            arr = img_dataset[idx]
                            img = Image.fromarray(arr).convert("RGB")
                            all_images.append(img)
                            episode_ids.append(int(ep_key))
                            frame_indices.append(idx)

                    elif "image_paths" in grp:
                        # Legacy format: file path strings
                        raw_paths = grp["image_paths"][:]
                        sampled_indices = list(range(0, len(raw_paths), sample_rate))
                        for idx in sampled_indices:
                            if len(all_images) >= max_images:
                                break
                            path_val = raw_paths[idx]
                            if isinstance(path_val, bytes):
                                path_val = path_val.decode("utf-8")
                            if not path_val:
                                continue
                            if not os.path.exists(path_val):
                                logger.warning("Image not found: %s", path_val)
                                continue
                            try:
                                img = Image.open(path_val).convert("RGB")
                                all_images.append(img)
                                episode_ids.append(int(ep_key))
                                frame_indices.append(idx)
                            except Exception as exc:
                                logger.warning("Failed to load image %s: %s", path_val, exc)

                    if len(all_images) >= max_images:
                        logger.info("Reached max_images cap (%d), stopping", max_images)
                        break
                if len(all_images) >= max_images:
                    break

        if not all_images:
            raise ValueError("No loadable images found in HDF5 episodes")

        embeddings = self._embed_images(all_images)
        index = self.build_faiss_index(embeddings)
        return EmbeddingIndex(
            index=index,
            episode_ids=episode_ids,
            frame_indices=frame_indices,
            dimension=embeddings.shape[1],
            num_embeddings=len(embeddings),
        )

    def extract_from_images(self, image_paths: list[str]) -> np.ndarray:
        """Extract embeddings from a list of image file paths."""
        images = [Image.open(p).convert("RGB") for p in image_paths]
        return self._embed_images(images)

    def build_faiss_index(self, embeddings: np.ndarray):
        """Build FAISS IndexFlatIP from embeddings.

        L2-normalizes embeddings first for cosine similarity via dot product.
        """
        import faiss

        embeddings = embeddings.astype(np.float32).copy()
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return index

    def save_index(self, index: EmbeddingIndex, path: str) -> None:
        """Save FAISS index + metadata to disk."""
        import faiss

        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index.index, str(base.with_suffix(".faiss")))

        meta = {
            "episode_ids": index.episode_ids,
            "frame_indices": index.frame_indices,
            "dimension": index.dimension,
            "num_embeddings": index.num_embeddings,
        }
        with open(base.with_suffix(".json"), "w") as f:
            json.dump(meta, f)

    def load_index(self, path: str) -> EmbeddingIndex:
        """Load cached FAISS index + metadata from disk."""
        import faiss

        base = Path(path)
        faiss_index = faiss.read_index(str(base.with_suffix(".faiss")))

        with open(base.with_suffix(".json")) as f:
            meta = json.load(f)

        return EmbeddingIndex(
            index=faiss_index,
            episode_ids=meta["episode_ids"],
            frame_indices=meta["frame_indices"],
            dimension=meta["dimension"],
            num_embeddings=meta["num_embeddings"],
        )
