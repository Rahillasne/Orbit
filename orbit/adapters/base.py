"""Base adapter interface for converting external dataset formats to ORBIT episodes."""

from __future__ import annotations

import abc
from collections.abc import Iterator
from typing import Any


class BaseAdapter(abc.ABC):
    """Abstract base for dataset adapters.

    Subclasses convert external formats (LeRobot, robomimic, etc.) into
    ORBIT's profiler-native episode dicts, yielded lazily via
    :meth:`iter_episodes`.

    Each yielded episode dict has the structure::

        {
            "episode_id": int,
            "states": np.ndarray,          # (T, state_dim) float32
            "actions": np.ndarray,         # (T, action_dim) float32
            "images": {str: list[ndarray]},  # camera_key -> list of HWC uint8 frames (optional)
            "metadata": dict,              # task labels, language instructions, etc. (optional)
        }
    """

    @abc.abstractmethod
    def iter_episodes(self) -> Iterator[dict[str, Any]]:
        """Lazily yield episodes in ORBIT-native dict format."""

    @property
    @abc.abstractmethod
    def num_episodes(self) -> int:
        """Total number of episodes available (before any max_episodes cap)."""

    @property
    def metadata(self) -> dict[str, Any]:
        """Dataset-level metadata (override in subclasses)."""
        return {}
