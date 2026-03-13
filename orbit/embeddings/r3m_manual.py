"""R3M embeddings without the r3m pip package.

Downloads R3M ResNet50 weights from HuggingFace Hub and loads them
into a standard torchvision model. This avoids dependency conflicts
with the r3m package which requires specific torch versions.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

logger = logging.getLogger(__name__)


class R3MManualExtractor:
    """Extract 2048-dim R3M embeddings using direct weight loading.

    Fallback chain:
    1. R3M weights from HuggingFace -> robotics-specific 2048-dim embeddings
    2. ImageNet ResNet50 (torchvision) -> general visual 2048-dim embeddings
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = self._resolve_device(device)
        self.model = self._load_model()
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )

    @property
    def dimension(self) -> int:
        return 2048

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "cuda" and torch.cuda.is_available():
            return "cuda"
        if device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if device not in ("cpu",) and device != "mps" and device != "cuda":
            logger.warning("Unknown device %s, falling back to CPU", device)
        if device in ("cuda", "mps"):
            logger.warning("Requested device %s unavailable, using CPU", device)
            return "cpu"
        return "cpu"

    def _load_model(self) -> nn.Module:
        """Load ResNet50 backbone with R3M weights, falling back to ImageNet.

        Tries multiple sources for the R3M weights (Ego4D-pretrained ResNet50):
        1. surajnair/r3m-50 on HuggingFace Hub (official R3M author)
        2. Cached weights from the r3m pip package (~/.r3m/)
        3. ImageNet ResNet50 as last resort
        """
        # --- Source 1: HuggingFace Hub (surajnair/r3m-50) ---
        try:
            from huggingface_hub import hf_hub_download

            weight_path = hf_hub_download(
                repo_id="surajnair/r3m-50",
                filename="pytorch_model.bin",
            )
            state_dict = torch.load(weight_path, map_location="cpu", weights_only=False)
            logger.info("Downloaded R3M weights from HuggingFace (surajnair/r3m-50)")

            backbone = self._load_r3m_state_dict(state_dict)
            if backbone is not None:
                logger.info("Using R3M ResNet50 (robotics) — Ego4D-pretrained embeddings")
                backbone.to(self.device)
                return backbone

        except Exception as e:
            logger.info("Could not load R3M from surajnair/r3m-50: %s", e)

        # --- Source 2: Cached r3m pip package weights ---
        try:
            import os

            r3m_cache = os.path.expanduser("~/.r3m")
            for root, _dirs, files in os.walk(r3m_cache):
                for fname in files:
                    if fname.endswith((".pt", ".pth")):
                        weight_path = os.path.join(root, fname)
                        state_dict = torch.load(weight_path, map_location="cpu", weights_only=False)
                        logger.info("Found cached R3M weights at %s", weight_path)

                        backbone = self._load_r3m_state_dict(state_dict)
                        if backbone is not None:
                            logger.info("Using R3M ResNet50 (robotics) — loaded from pip cache")
                            backbone.to(self.device)
                            return backbone
        except Exception as e:
            logger.info("Could not load R3M from pip cache: %s", e)

        logger.warning("R3M weights unavailable — falling back to ImageNet ResNet50")

        # --- Source 3: ImageNet ResNet50 (always available via torchvision) ---
        try:
            logger.info("Using ImageNet ResNet50 (general) — not robotics-specific")
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        except Exception as e2:
            logger.warning(
                "Could not download ImageNet weights (%s), using untrained ResNet50.",
                e2,
            )
            backbone = models.resnet50(weights=None)

        backbone = nn.Sequential(*list(backbone.children())[:-1])
        backbone.eval()
        backbone.to(self.device)
        return backbone

    def _load_r3m_state_dict(self, state_dict: dict) -> nn.Module | None:
        """Load R3M state dict into a ResNet50 backbone, handling key prefixes."""
        # Handle nested state dicts (e.g. {"state_dict": {...}})
        if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
            state_dict = state_dict["state_dict"]
        if "model_state_dict" in state_dict and isinstance(state_dict["model_state_dict"], dict):
            state_dict = state_dict["model_state_dict"]

        # Clean up key prefixes (R3M uses "module." and "convnet." prefixes)
        cleaned = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "").replace("convnet.", "")
            cleaned[k] = v

        # Load into full ResNet50 first (named layers match), then strip fc head
        full_model = models.resnet50(weights=None)
        result = full_model.load_state_dict(cleaned, strict=False)

        # Verify we loaded the backbone weights (fc.weight/fc.bias missing is fine)
        missing_backbone = [k for k in result.missing_keys if not k.startswith("fc.")]
        if missing_backbone:
            logger.warning("R3M loading missed backbone keys: %s", missing_backbone[:5])
            return None

        unexpected = [k for k in result.unexpected_keys if not k.startswith("fc.")]
        if unexpected:
            logger.debug("R3M unexpected keys (ignored): %s", unexpected[:5])

        # Remove fc layer, keep everything else as feature extractor
        backbone = nn.Sequential(*list(full_model.children())[:-1])
        backbone.eval()
        return backbone

    def embed_images(self, images: list, batch_size: int = 32) -> np.ndarray:
        """Extract embeddings for a list of PIL Images or numpy arrays.

        Returns: (N, 2048) numpy array of L2-normalized embeddings.
        """
        if not images:
            return np.zeros((0, 2048), dtype=np.float32)

        all_embeddings = []

        for i in tqdm(range(0, len(images), batch_size), desc="R3M embedding", leave=False):
            batch = images[i : i + batch_size]
            tensors = []
            for img in batch:
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                if hasattr(img, "mode") and img.mode != "RGB":
                    img = img.convert("RGB")
                tensors.append(self.transform(img))

            batch_tensor = torch.stack(tensors).to(self.device)

            with torch.no_grad():
                embeddings = self.model(batch_tensor)

                # Flatten from (B, 2048, 1, 1) to (B, 2048)
                if embeddings.dim() == 4:
                    embeddings = embeddings.squeeze(-1).squeeze(-1)
                elif embeddings.dim() == 3:
                    embeddings = embeddings.squeeze(-1)

            # L2 normalize
            norms = torch.norm(embeddings, dim=1, keepdim=True).clamp(min=1e-8)
            embeddings = embeddings / norms

            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings).astype(np.float32)
