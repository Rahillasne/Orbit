"""Adapter for loading robomimic/robosuite HDF5 datasets into ORBIT's episode format."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from orbit.adapters.base import BaseAdapter

logger = logging.getLogger(__name__)

# Keys robomimic uses for joint state, in priority order.
_STATE_KEYS = (
    "joint_pos",
    "robot0_joint_pos",
    "robot0_eef_pos",
    "object",
)

# Keys robomimic uses for camera images.
_IMAGE_KEYS = (
    "agentview_image",
    "eye_in_hand_image",
    "sideview_image",
    "frontview_image",
    "robot0_eye_in_hand_image",
    "robot0_agentview_image",
)


def _find_state_array(obs_group) -> np.ndarray | None:
    """Find the best state array in an ``obs`` HDF5 group."""
    for key in _STATE_KEYS:
        if key in obs_group:
            return obs_group[key][:].astype(np.float32)
    # Fallback: concatenate all non-image 1D arrays
    arrays = []
    for key in sorted(obs_group.keys()):
        ds = obs_group[key]
        if ds.ndim == 2 and ds.shape[1] < 100:  # likely a state vector, not an image
            arrays.append(ds[:].astype(np.float32))
    if arrays:
        return np.concatenate(arrays, axis=1)
    return None


def _find_image_arrays(obs_group) -> dict[str, np.ndarray]:
    """Extract image datasets from an ``obs`` HDF5 group.

    Returns ``{camera_key: (T, H, W, C) uint8 array}``.
    """
    images: dict[str, np.ndarray] = {}
    for key in _IMAGE_KEYS:
        if key in obs_group:
            arr = obs_group[key][:]
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (arr * 255).clip(0, 255).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
            images[key] = arr
    return images


class RobomimicAdapter(BaseAdapter):
    """Lazily iterate a robomimic/robosuite HDF5 file as ORBIT-native episode dicts.

    Robomimic HDF5 layout::

        data/
          demo_0/
            actions: (T, action_dim)
            obs/
              joint_pos: (T, 7)
              agentview_image: (T, H, W, 3)
              ...
          demo_1/
            ...

    Parameters
    ----------
    hdf5_path:
        Path to the robomimic HDF5 file.
    max_episodes:
        Cap the number of episodes yielded. ``None`` for all.
    """

    def __init__(
        self,
        hdf5_path: str | Path,
        max_episodes: int | None = None,
    ) -> None:
        self._path = Path(hdf5_path)
        self._max_episodes = max_episodes

        if not self._path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self._path}")

        self._demo_keys: list[str] = []
        self._dataset_meta: dict[str, Any] = {}
        self._scan_file()

    def _scan_file(self) -> None:
        """Read demo keys and metadata without loading episode data."""
        import h5py

        with h5py.File(self._path, "r") as f:
            if "data" not in f:
                raise ValueError(
                    f"Expected 'data' group in {self._path}. "
                    f"Top-level keys: {list(f.keys())}"
                )
            data_grp = f["data"]

            # Collect demo keys sorted numerically
            demo_keys = [k for k in data_grp.keys() if k.startswith("demo")]
            demo_keys.sort(key=lambda k: int(re.search(r"(\d+)", k).group(1)))  # type: ignore[union-attr]
            self._demo_keys = demo_keys

            # Gather dataset-level attrs
            meta: dict[str, Any] = {"source": "robomimic", "path": str(self._path)}
            for attr_key in ("env", "env_name", "env_args", "type"):
                if attr_key in data_grp.attrs:
                    val = data_grp.attrs[attr_key]
                    if isinstance(val, bytes):
                        val = val.decode()
                    meta[attr_key] = val
            meta["num_demos"] = len(demo_keys)
            self._dataset_meta = meta

        logger.info(
            "Scanned robomimic file %s: %d demos", self._path, len(self._demo_keys)
        )

    @property
    def num_episodes(self) -> int:
        return len(self._demo_keys)

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._dataset_meta)

    def iter_episodes(self) -> Iterator[dict[str, Any]]:
        """Lazily yield episodes, opening the HDF5 file once."""
        import h5py

        demo_keys = self._demo_keys
        if self._max_episodes is not None:
            demo_keys = demo_keys[: self._max_episodes]

        with h5py.File(self._path, "r") as f:
            data_grp = f["data"]
            for ep_idx, demo_key in enumerate(demo_keys):
                demo = data_grp[demo_key]

                # Actions (required)
                if "actions" not in demo:
                    logger.warning("Skipping %s: no 'actions' dataset", demo_key)
                    continue
                actions = demo["actions"][:].astype(np.float32)

                if len(actions) < 2:
                    continue

                # States
                states: np.ndarray
                if "obs" in demo:
                    found = _find_state_array(demo["obs"])
                    if found is not None:
                        states = found
                    else:
                        states = np.zeros(
                            (len(actions), actions.shape[1]), dtype=np.float32
                        )
                elif "states" in demo:
                    states = demo["states"][:].astype(np.float32)
                else:
                    states = np.zeros(
                        (len(actions), actions.shape[1]), dtype=np.float32
                    )

                # Align lengths
                min_len = min(len(states), len(actions))
                if min_len < 2:
                    continue

                # Images (lazy — only read if obs group exists)
                images: dict[str, list[np.ndarray]] = {}
                if "obs" in demo:
                    raw_images = _find_image_arrays(demo["obs"])
                    for cam_key, img_arr in raw_images.items():
                        images[cam_key] = [img_arr[t] for t in range(min(len(img_arr), min_len))]

                # Per-demo metadata
                ep_metadata: dict[str, Any] = {"demo_key": demo_key}
                for attr_key in demo.attrs:
                    val = demo.attrs[attr_key]
                    if isinstance(val, bytes):
                        val = val.decode()
                    ep_metadata[attr_key] = val

                yield {
                    "episode_id": ep_idx,
                    "states": states[:min_len],
                    "actions": actions[:min_len],
                    "images": images,
                    "metadata": ep_metadata,
                }
