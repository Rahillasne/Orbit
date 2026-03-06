"""Embedding extraction and FAISS index construction."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

# torch must be imported before faiss on macOS to avoid SIGSEGV
try:
    import torch as _torch  # noqa: F401
except ImportError:
    pass

from orbit.profile.types import EmbeddingIndex

logger = logging.getLogger(__name__)


class EmbeddingExtractor:
    """Extract visual embeddings and build FAISS indexes for similarity search.

    Delegates to the existing ``EmbeddingAnalyzer`` for SigLIP model loading
    and batched embedding computation.  Falls back to random projection when
    SigLIP / transformers is not installed.
    """

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._analyzer = None
        self._use_fallback = False
        self._random_proj = None

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

    def _embed_images(self, images: list[Image.Image]) -> np.ndarray:
        """Embed images using SigLIP or fallback."""
        analyzer = self._get_analyzer()
        if analyzer is not None:
            return analyzer.embed_images(images)
        return self._fallback_embed(images)

    def _fallback_embed(self, images: list[Image.Image]) -> np.ndarray:
        """Random projection fallback when SigLIP is unavailable."""
        dim = 768
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

    def _extract_from_hdf5(self, h5_files: list[Path], sample_rate: int) -> EmbeddingIndex:
        """Load images from HDF5 files and extract embeddings."""
        import h5py

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

        ep_counter = 0
        for h5_path, ep_keys in file_episode_map:
            with h5py.File(h5_path, "r") as f:
                for ep_key in ep_keys:
                    ep_counter += 1
                    grp = f["episodes"][ep_key]
                    if "image_paths" not in grp:
                        continue

                    raw_paths = grp["image_paths"][:]
                    sampled_indices = list(range(0, len(raw_paths), sample_rate))
                    total_frames = len(sampled_indices)

                    for frame_num, idx in enumerate(sampled_indices):
                        path_val = raw_paths[idx]
                        if isinstance(path_val, bytes):
                            path_val = path_val.decode("utf-8")
                        if not path_val:
                            continue
                        try:
                            img = Image.open(path_val).convert("RGB")
                            all_images.append(img)
                            episode_ids.append(int(ep_key))
                            frame_indices.append(idx)
                        except Exception as exc:
                            logger.warning("Failed to load image %s: %s", path_val, exc)

                    print(
                        f"Extracting embeddings: episode {ep_counter}/{total_eps}, "
                        f"frame {total_frames}/{total_frames}"
                    )

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
