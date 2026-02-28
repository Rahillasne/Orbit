"""Vision-language model-based failure description generator."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import open_clip
import torch
from PIL import Image

from orbit.logger.schemas import EpisodeRecord


@dataclass
class FailureDescription:
    """Natural language description of a failure mode."""

    episode_id: int
    description: str
    key_frame_indices: list[int] = field(default_factory=list)
    similarity_scores: dict[str, float] = field(default_factory=dict)
    raw_scores: np.ndarray | None = None


class FailureDescriber:
    """Uses OpenCLIP to generate natural language descriptions of failure modes.

    Uses CLIP's contrastive similarity to match pre-defined text descriptions
    to visual observations (zero-shot classification, not generative).
    """

    DEFAULT_FAILURE_PROMPTS: list[str] = [
        "a robot arm that is stuck and not moving",
        "a robot arm that has dropped the object",
        "a robot arm that missed the target location",
        "a robot arm colliding with an obstacle",
        "a robot arm moving erratically and oscillating",
        "a robot arm that failed to grasp the object",
        "a robot arm in an unreachable configuration",
        "a robot arm that pushed the object off the table",
        "an empty gripper with no object",
        "a robot arm moving too slowly",
    ]

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cpu",
        failure_prompts: list[str] | None = None,
    ) -> None:
        self._model_name = model_name
        self._pretrained = pretrained
        self._device = device
        self._failure_prompts = failure_prompts or list(self.DEFAULT_FAILURE_PROMPTS)

        self._model = None
        self._preprocess = None
        self._tokenizer = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self._model_name, pretrained=self._pretrained
        )
        self._model.eval().to(self._device)
        self._tokenizer = open_clip.get_tokenizer(self._model_name)

    def add_failure_prompts(self, prompts: list[str]) -> None:
        """Add custom failure mode prompts."""
        self._failure_prompts.extend(prompts)

    def describe(self, episode: EpisodeRecord, image_key: str = "front") -> FailureDescription:
        """Generate a failure description for a single episode."""
        self._load_model()

        key_frames = self._find_key_frames(episode, image_key)
        if not key_frames:
            return FailureDescription(
                episode_id=episode.episode_id,
                description="No images available for analysis.",
            )

        frame_indices, images = zip(*key_frames)
        scores = self._classify_images(list(images))

        # Build description from top-3 scoring prompts
        sorted_prompts = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_descriptions = [p for p, _ in sorted_prompts[:3]]
        description = (
            f"Failure analysis for episode {episode.episode_id}: "
            + "; ".join(top_descriptions)
            + "."
        )

        return FailureDescription(
            episode_id=episode.episode_id,
            description=description,
            key_frame_indices=list(frame_indices),
            similarity_scores=scores,
        )

    def describe_batch(
        self, episodes: list[EpisodeRecord], image_key: str = "front"
    ) -> list[FailureDescription]:
        """Generate descriptions for multiple episodes."""
        return [self.describe(ep, image_key) for ep in episodes]

    def _find_key_frames(
        self,
        episode: EpisodeRecord,
        image_key: str,
        n_frames: int = 5,
    ) -> list[tuple[int, Image.Image]]:
        """Extract key frames from an episode.

        Selects the last 3 steps (failure outcome) and 2 evenly-spaced
        steps from the middle (context).
        """
        if not episode.steps:
            return []

        n_steps = len(episode.steps)
        indices: list[int] = []

        # Last 3 steps
        end_indices = list(range(max(0, n_steps - 3), n_steps))
        indices.extend(end_indices)

        # 2 evenly-spaced from the middle
        mid_indices = np.linspace(0, n_steps - 1, n_frames - len(end_indices) + 2, dtype=int)
        indices.extend(int(i) for i in mid_indices[1:-1])

        # Deduplicate and sort
        indices = sorted(set(indices))[:n_frames]

        frames: list[tuple[int, Image.Image]] = []
        for idx in indices:
            step = episode.steps[idx]
            if image_key in step.images:
                img = step.images[image_key]
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                frames.append((idx, img))

        return frames

    def _classify_images(self, images: list[Image.Image]) -> dict[str, float]:
        """Compute CLIP similarity scores between images and failure prompts."""
        # Encode images
        img_tensors = torch.stack([self._preprocess(img) for img in images])
        img_tensors = img_tensors.to(self._device)

        # Encode text prompts
        text_tokens = self._tokenizer(self._failure_prompts).to(self._device)

        with torch.no_grad():
            image_features = self._model.encode_image(img_tensors)
            text_features = self._model.encode_text(text_tokens)

            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Cosine similarity: (N_images, N_prompts)
            similarity = (image_features @ text_features.T).cpu().numpy()

        # Average across frames to get per-prompt scores
        mean_scores = similarity.mean(axis=0)

        return {prompt: float(score) for prompt, score in zip(self._failure_prompts, mean_scores)}
