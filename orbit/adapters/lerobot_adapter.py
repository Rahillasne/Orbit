"""Adapter for loading LeRobot datasets into ORBIT's episode format."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import numpy as np

from orbit.adapters.base import BaseAdapter

logger = logging.getLogger(__name__)


def _tensor_to_numpy(tensor) -> np.ndarray:
    """Convert a torch tensor (or array-like) to a numpy array."""
    if hasattr(tensor, "numpy"):
        return tensor.numpy()
    return np.asarray(tensor)


def _image_tensor_to_hwc_uint8(img_tensor) -> np.ndarray:
    """Convert a LeRobot image tensor (CHW float [0,1]) to HWC uint8."""
    arr = _tensor_to_numpy(img_tensor)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype in (np.float32, np.float64):
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return arr


def _detect_camera_keys(sample: dict) -> list[str]:
    """Find all observation image keys in a LeRobot sample."""
    camera_keys = []
    for key in sample:
        if key.startswith("observation.image") or key.startswith("observation.images"):
            camera_keys.append(key)
    return sorted(camera_keys)


class LeRobotAdapter(BaseAdapter):
    """Lazily iterate a LeRobot dataset as ORBIT-native episode dicts.

    Parameters
    ----------
    repo_id_or_path:
        HuggingFace repo ID (e.g. ``"lerobot/pusht"``) or local path.
    max_episodes:
        Cap the number of episodes yielded. ``None`` for all.
    fps_sample:
        Sub-sample rate (1 = every frame, 2 = every other frame, etc.).
    camera_keys:
        Explicit list of image keys to extract. ``None`` auto-detects.
    """

    def __init__(
        self,
        repo_id_or_path: str,
        max_episodes: int | None = None,
        fps_sample: int = 1,
        camera_keys: list[str] | None = None,
    ) -> None:
        self._repo_id = repo_id_or_path
        self._max_episodes = max_episodes
        self._fps_sample = max(1, fps_sample)
        self._camera_keys = camera_keys

        self._ds = None
        self._from_indices: list[int] = []
        self._to_indices: list[int] = []
        self._total_episodes = 0
        self._dataset_meta: dict[str, Any] = {}

        self._load_dataset()

    def _load_dataset(self) -> None:
        """Load the LeRobot dataset (SDK or fallback)."""
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

            ds = LeRobotDataset(self._repo_id)
            self._ds = ds

            ep_index = ds.episode_data_index
            self._from_indices = ep_index["from"].tolist()
            self._to_indices = ep_index["to"].tolist()
            self._total_episodes = len(self._from_indices)

            self._dataset_meta = {
                "source": "lerobot_sdk",
                "repo_id": self._repo_id,
                "total_frames": len(ds),
                "num_episodes": self._total_episodes,
            }
            logger.info(
                "Loaded LeRobot dataset %s: %d episodes, %d frames",
                self._repo_id,
                self._total_episodes,
                len(ds),
            )
        except ImportError:
            logger.info("LeRobot SDK not available, falling back to HuggingFace Hub")
            self._load_via_hub()
        except Exception as exc:
            logger.warning("LeRobot SDK failed (%s), falling back to Hub download", exc)
            self._load_via_hub()

    def _load_via_hub(self) -> None:
        """Fallback: load episode metadata from parquet files via HuggingFace Hub."""
        from pathlib import Path

        local_path = Path(self._repo_id)
        if not local_path.is_dir():
            from huggingface_hub import snapshot_download

            local_path = Path(snapshot_download(self._repo_id, repo_type="dataset"))

        import pandas as pd

        parquet_files = sorted(local_path.glob("data/**/*.parquet")) or sorted(
            local_path.glob("**/*.parquet")
        )
        if not parquet_files:
            raise FileNotFoundError(
                f"No parquet files found in {local_path}. "
                "Dataset may not be in LeRobot v2 format."
            )

        df = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)

        # Detect episode column
        ep_col = None
        for col_name in ("episode_index", "episode_id", "episode"):
            if col_name in df.columns:
                ep_col = col_name
                break
        if ep_col is None:
            raise ValueError(f"Cannot find episode column. Columns: {list(df.columns)}")

        self._hub_df = df
        self._hub_ep_col = ep_col
        self._hub_unique_episodes = sorted(df[ep_col].unique())
        self._total_episodes = len(self._hub_unique_episodes)

        self._dataset_meta = {
            "source": "huggingface_hub",
            "repo_id": self._repo_id,
            "total_frames": len(df),
            "num_episodes": self._total_episodes,
        }
        logger.info(
            "Loaded LeRobot dataset via Hub: %d episodes, %d rows",
            self._total_episodes,
            len(df),
        )

    @property
    def num_episodes(self) -> int:
        return self._total_episodes

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._dataset_meta)

    def iter_episodes(self) -> Iterator[dict[str, Any]]:
        """Lazily yield episodes in ORBIT-native dict format."""
        if self._ds is not None:
            yield from self._iter_via_sdk()
        else:
            yield from self._iter_via_hub()

    def _iter_via_sdk(self) -> Iterator[dict[str, Any]]:
        """Iterate episodes using the LeRobot SDK."""
        n_eps = self._total_episodes
        if self._max_episodes is not None:
            n_eps = min(n_eps, self._max_episodes)

        auto_camera_keys = self._camera_keys

        for ep_id in range(n_eps):
            start_idx = self._from_indices[ep_id]
            end_idx = self._to_indices[ep_id]
            frame_indices = list(range(start_idx, end_idx, self._fps_sample))

            if not frame_indices:
                continue

            states_list: list[np.ndarray] = []
            actions_list: list[np.ndarray] = []
            images: dict[str, list[np.ndarray]] = {}
            ep_metadata: dict[str, Any] = {}

            for idx in frame_indices:
                sample = self._ds[idx]

                # State
                if "observation.state" in sample:
                    states_list.append(
                        _tensor_to_numpy(sample["observation.state"]).astype(np.float32)
                    )

                # Action
                if "action" in sample:
                    actions_list.append(
                        _tensor_to_numpy(sample["action"]).astype(np.float32)
                    )

                # Images — auto-detect camera keys on first sample
                if auto_camera_keys is None:
                    auto_camera_keys = _detect_camera_keys(sample)

                for cam_key in auto_camera_keys:
                    if cam_key in sample:
                        img = _image_tensor_to_hwc_uint8(sample[cam_key])
                        images.setdefault(cam_key, []).append(img)

                # Language instruction (if present)
                if "language_instruction" in sample and not ep_metadata.get(
                    "language_instruction"
                ):
                    instr = sample["language_instruction"]
                    if isinstance(instr, str) and instr:
                        ep_metadata["language_instruction"] = instr

                # Task index
                if "task_index" in sample and "task_index" not in ep_metadata:
                    ti = sample["task_index"]
                    ep_metadata["task_index"] = int(ti) if hasattr(ti, "item") else ti

            if not actions_list:
                continue

            # Build states — if missing, use zeros matching action dim
            if states_list:
                states = np.stack(states_list)
            else:
                action_dim = actions_list[0].shape[0]
                states = np.zeros((len(actions_list), action_dim), dtype=np.float32)

            actions = np.stack(actions_list)

            # Align lengths
            min_len = min(len(states), len(actions))
            if min_len < 2:
                continue

            yield {
                "episode_id": ep_id,
                "states": states[:min_len],
                "actions": actions[:min_len],
                "images": images,
                "metadata": ep_metadata,
            }

    def _iter_via_hub(self) -> Iterator[dict[str, Any]]:
        """Iterate episodes from parquet data (Hub fallback)."""
        from orbit.profile.loaders import DatasetLoader

        df = self._hub_df
        ep_col = self._hub_ep_col
        unique_eps = self._hub_unique_episodes

        if self._max_episodes is not None:
            unique_eps = unique_eps[: self._max_episodes]

        state_cols = DatasetLoader._detect_columns(df, "observation.state")
        action_cols = DatasetLoader._detect_columns(df, "action")

        if not action_cols:
            raise ValueError(f"Cannot find action columns. Columns: {list(df.columns)}")

        for ep_id in unique_eps:
            ep_mask = df[ep_col] == ep_id
            ep_indices = df.index[ep_mask].tolist()
            sampled = ep_indices[:: self._fps_sample]
            ep_df = df.loc[sampled]

            if state_cols:
                states = DatasetLoader._extract_array_column(ep_df, state_cols)
            else:
                actions_tmp = DatasetLoader._extract_array_column(ep_df, action_cols)
                states = np.zeros_like(actions_tmp)

            actions = DatasetLoader._extract_array_column(ep_df, action_cols)

            min_len = min(len(states), len(actions))
            if min_len < 2:
                continue

            ep_metadata: dict[str, Any] = {}
            if "language_instruction" in df.columns:
                instructions = ep_df["language_instruction"].dropna().unique()
                if len(instructions) > 0:
                    ep_metadata["language_instruction"] = str(instructions[0])

            yield {
                "episode_id": int(ep_id),
                "states": states[:min_len],
                "actions": actions[:min_len],
                "images": {},
                "metadata": ep_metadata,
            }
