"""Dataset loaders for converting external formats to ORBIT's HDF5 format."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Load and convert datasets from various formats into ORBIT's HDF5 format."""

    @staticmethod
    def from_lerobot(
        repo_id_or_path: str,
        output_dir: str | Path,
        max_episodes: int | None = 50,
        fps_sample: int = 1,
    ) -> Path:
        """Convert a LeRobot dataset to ORBIT's HDF5 format.

        Tries the LeRobot SDK first, falls back to downloading raw files
        from HuggingFace Hub and parsing parquet + video.

        Parameters
        ----------
        repo_id_or_path:
            HuggingFace repo ID (e.g. ``"lerobot/aloha_static_cups_open"``)
            or local path to an already-downloaded dataset.
        output_dir:
            Directory to write the converted HDF5 + images.
        max_episodes:
            Maximum number of episodes to convert.  ``None`` for all.
        fps_sample:
            Sub-sample rate for video frame extraction (1 = every frame).

        Returns
        -------
        Path
            Path to the output directory containing the HDF5 file.
        """
        output_dir = Path(output_dir)
        h5_path = output_dir / "session_lerobot.h5"

        if h5_path.exists():
            logger.info("Cached HDF5 found at %s — skipping conversion", h5_path)
            return output_dir

        output_dir.mkdir(parents=True, exist_ok=True)

        # Path A: LeRobot SDK
        try:
            return DatasetLoader._convert_via_lerobot_sdk(
                repo_id_or_path, output_dir, max_episodes, fps_sample
            )
        except ImportError:
            logger.info("LeRobot SDK not available, falling back to HuggingFace Hub")
        except Exception as exc:
            logger.warning("LeRobot SDK conversion failed (%s), trying fallback", exc)

        # Path B: Raw download + parquet/video parsing
        return DatasetLoader._convert_via_hub_download(
            repo_id_or_path, output_dir, max_episodes, fps_sample
        )

    @staticmethod
    def from_hdf5_directory(data_dir: str | Path) -> list[dict]:
        """Load episodes from ORBIT's native HDF5 format.

        Returns list of ``{episode_id, states, actions}`` dicts.
        """
        import h5py

        data_path = Path(data_dir)
        h5_files = sorted(data_path.glob("session_*.h5")) or sorted(data_path.glob("*.h5"))

        episodes: list[dict] = []
        for h5_path in h5_files:
            with h5py.File(h5_path, "r") as f:
                if "episodes" not in f:
                    continue
                for ep_key in f["episodes"]:
                    grp = f["episodes"][ep_key]
                    states = None
                    actions = None
                    if "states" in grp:
                        states = grp["states"][:]
                    elif "joint_positions" in grp:
                        states = grp["joint_positions"][:]
                    if "actions" in grp:
                        actions = grp["actions"][:]
                    if states is not None and actions is not None:
                        min_len = min(len(states), len(actions))
                        if min_len >= 2:
                            episodes.append(
                                {
                                    "episode_id": int(ep_key),
                                    "states": states[:min_len],
                                    "actions": actions[:min_len],
                                }
                            )
        return episodes

    @staticmethod
    def from_image_directory(data_dir: str | Path) -> list[str]:
        """Load image paths from a directory.

        Returns sorted list of image paths (jpg, png, jpeg).
        """
        data_path = Path(data_dir)
        paths: list[str] = []
        for ext in ("*.jpg", "*.png", "*.jpeg"):
            paths.extend(str(p) for p in sorted(data_path.glob(ext)))
        return sorted(paths)

    # ------------------------------------------------------------------
    # Path A: LeRobot SDK
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_via_lerobot_sdk(
        repo_id: str,
        output_dir: Path,
        max_episodes: int | None,
        fps_sample: int,
    ) -> Path:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset(repo_id)
        logger.info("Loaded LeRobot dataset: %d frames", len(ds))

        # Determine episode boundaries
        ep_index = ds.episode_data_index
        from_indices = ep_index["from"].tolist()
        to_indices = ep_index["to"].tolist()
        num_episodes = len(from_indices)
        if max_episodes is not None:
            num_episodes = min(num_episodes, max_episodes)

        logger.info("Converting %d episodes", num_episodes)

        img_dir = output_dir / "images"
        img_dir.mkdir(exist_ok=True)

        episodes: list[dict] = []
        all_image_paths: dict[int, list[str]] = {}

        for ep_id in range(num_episodes):
            start_idx = from_indices[ep_id]
            end_idx = to_indices[ep_id]
            frame_indices = list(range(start_idx, end_idx, fps_sample))

            states_list = []
            actions_list = []
            img_paths: list[str] = []

            for fi, idx in enumerate(frame_indices):
                sample = ds[idx]

                # Extract state
                if "observation.state" in sample:
                    state = sample["observation.state"]
                    if hasattr(state, "numpy"):
                        state = state.numpy()
                    states_list.append(np.asarray(state, dtype=np.float32))

                # Extract action
                if "action" in sample:
                    action = sample["action"]
                    if hasattr(action, "numpy"):
                        action = action.numpy()
                    actions_list.append(np.asarray(action, dtype=np.float32))

                # Extract image — try common camera keys
                img_tensor = None
                for cam_key in [
                    "observation.images.top",
                    "observation.image",
                    "observation.images.wrist",
                    "observation.images.front",
                ]:
                    if cam_key in sample:
                        img_tensor = sample[cam_key]
                        break

                if img_tensor is not None:
                    from PIL import Image

                    if hasattr(img_tensor, "numpy"):
                        img_arr = img_tensor.numpy()
                    else:
                        img_arr = np.asarray(img_tensor)
                    # LeRobot tensors are CHW float [0,1] — convert to HWC uint8
                    if img_arr.ndim == 3 and img_arr.shape[0] in (1, 3):
                        img_arr = np.transpose(img_arr, (1, 2, 0))
                    if img_arr.dtype in (np.float32, np.float64):
                        img_arr = (img_arr * 255).clip(0, 255).astype(np.uint8)
                    img = Image.fromarray(img_arr)
                    img_path = img_dir / f"ep{ep_id}_f{fi}.png"
                    img.save(img_path)
                    img_paths.append(str(img_path))

            if states_list and actions_list:
                episodes.append(
                    {
                        "episode_id": ep_id,
                        "states": np.stack(states_list),
                        "actions": np.stack(actions_list),
                    }
                )
                all_image_paths[ep_id] = img_paths

            if (ep_id + 1) % 10 == 0 or ep_id == num_episodes - 1:
                logger.info("Converted %d/%d episodes", ep_id + 1, num_episodes)

        DatasetLoader._write_hdf5(output_dir, episodes, all_image_paths)
        return output_dir

    # ------------------------------------------------------------------
    # Path B: Hub download + parquet/video parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_via_hub_download(
        repo_id: str,
        output_dir: Path,
        max_episodes: int | None,
        fps_sample: int,
    ) -> Path:
        from huggingface_hub import snapshot_download

        local_path = Path(repo_id)
        if not local_path.is_dir():
            logger.info("Downloading %s from HuggingFace Hub...", repo_id)
            local_path = Path(snapshot_download(repo_id, repo_type="dataset"))

        logger.info("Dataset at %s", local_path)

        # Read parquet data
        import pandas as pd

        parquet_files = sorted(local_path.glob("data/**/*.parquet"))
        if not parquet_files:
            parquet_files = sorted(local_path.glob("**/*.parquet"))

        if not parquet_files:
            raise FileNotFoundError(
                f"No parquet files found in {local_path}. Dataset may not be in LeRobot v2 format."
            )

        df = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
        logger.info("Loaded %d rows from %d parquet files", len(df), len(parquet_files))

        # Detect episode column
        ep_col = None
        for col_name in ("episode_index", "episode_id", "episode"):
            if col_name in df.columns:
                ep_col = col_name
                break
        if ep_col is None:
            raise ValueError(f"Cannot find episode column. Columns: {list(df.columns)}")

        # Detect state and action columns
        state_cols = DatasetLoader._detect_columns(df, "observation.state")
        action_cols = DatasetLoader._detect_columns(df, "action")

        if not action_cols:
            raise ValueError(f"Cannot find action columns. Columns: {list(df.columns)}")

        # Build episodes
        unique_episodes = sorted(df[ep_col].unique())
        if max_episodes is not None:
            unique_episodes = unique_episodes[:max_episodes]

        episodes: list[dict] = []
        all_image_paths: dict[int, list[str]] = {}

        # Collect the global frame indices we need per episode
        ep_frame_indices: dict[int, list[int]] = {}
        for ep_id in unique_episodes:
            ep_mask = df[ep_col] == ep_id
            ep_indices = df.index[ep_mask].tolist()
            sampled = ep_indices[::fps_sample]
            ep_df = df.loc[sampled]

            if state_cols:
                states = DatasetLoader._extract_array_column(ep_df, state_cols)
            else:
                states = DatasetLoader._extract_array_column(ep_df, action_cols)
                logger.warning(
                    "No state columns found; using actions as states for episode %s",
                    ep_id,
                )

            actions = DatasetLoader._extract_array_column(ep_df, action_cols)

            if len(states) >= 2:
                episodes.append(
                    {
                        "episode_id": int(ep_id),
                        "states": states,
                        "actions": actions,
                    }
                )
                ep_frame_indices[int(ep_id)] = sampled

        # Extract video frames for the selected episodes
        img_dir = output_dir / "images"
        img_dir.mkdir(exist_ok=True)

        video_file = DatasetLoader._find_best_video(local_path)
        if video_file:
            # Collect all needed global frame indices
            needed_indices: set[int] = set()
            for indices in ep_frame_indices.values():
                needed_indices.update(indices)

            frame_path_map = DatasetLoader._extract_specific_frames(
                video_file, img_dir, sorted(needed_indices)
            )

            # Map back to episodes
            for ep_id, indices in ep_frame_indices.items():
                all_image_paths[ep_id] = [
                    frame_path_map[idx] for idx in indices if idx in frame_path_map
                ]
            logger.info(
                "Extracted %d frames from video for %d episodes",
                len(frame_path_map),
                len(ep_frame_indices),
            )
        else:
            logger.warning("No video files found; episodes will have no images")

        logger.info("Built %d episodes from parquet data", len(episodes))
        DatasetLoader._write_hdf5(output_dir, episodes, all_image_paths)
        return output_dir

    # ------------------------------------------------------------------
    # Video frame extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _find_best_video(dataset_dir: Path) -> Path | None:
        """Find the first camera video file in a LeRobot dataset.

        LeRobot v2 stores one concatenated video per camera under
        ``videos/<camera_key>/chunk-*/file-*.mp4``.  We pick the first
        camera alphabetically.
        """
        video_dir = dataset_dir / "videos"
        if not video_dir.exists():
            return None
        videos = sorted(video_dir.glob("**/*.mp4"))
        return videos[0] if videos else None

    @staticmethod
    def _extract_specific_frames(
        video_path: Path,
        output_dir: Path,
        frame_indices: list[int],
    ) -> dict[int, str]:
        """Extract specific frames (by global index) from a video.

        Returns ``{global_frame_index: saved_image_path}``.
        """
        if not frame_indices:
            return {}

        needed = set(frame_indices)
        result: dict[int, str] = {}

        # Try OpenCV
        try:
            import cv2

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open {video_path}")

            max_idx = max(needed)
            frame_idx = 0
            while frame_idx <= max_idx:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx in needed:
                    out_path = output_dir / f"frame_{frame_idx:06d}.png"
                    cv2.imwrite(str(out_path), frame)
                    result[frame_idx] = str(out_path)
                frame_idx += 1
            cap.release()
            logger.info("Extracted %d/%d frames via OpenCV", len(result), len(needed))
            return result
        except ImportError:
            pass
        except Exception as exc:
            logger.warning("OpenCV frame extraction failed: %s", exc)

        # Fallback: extract all frames with ffmpeg, then keep only needed
        try:
            pattern = str(output_dir / "frame_%06d.png")
            cmd = [
                "ffmpeg",
                "-i",
                str(video_path),
                pattern,
                "-y",
                "-loglevel",
                "error",
            ]
            subprocess.run(cmd, check=True, capture_output=True, timeout=600)
            # ffmpeg names frames starting from 1
            for idx in needed:
                ffmpeg_path = output_dir / f"frame_{idx + 1:06d}.png"
                target_path = output_dir / f"frame_{idx:06d}.png"
                if ffmpeg_path.exists():
                    if ffmpeg_path != target_path:
                        ffmpeg_path.rename(target_path)
                    result[idx] = str(target_path)
            return result
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            logger.warning("ffmpeg frame extraction failed: %s", exc)

        logger.warning("No video extraction method available for %s", video_path)
        return {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _write_hdf5(
        output_dir: Path,
        episodes: list[dict],
        image_paths: dict[int, list[str]],
    ) -> Path:
        """Write episodes to ORBIT-format HDF5."""
        import h5py

        h5_path = output_dir / "session_lerobot.h5"
        with h5py.File(h5_path, "w") as f:
            f.attrs["session_id"] = "lerobot_conversion"
            eps_grp = f.create_group("episodes")
            for ep in episodes:
                eid = ep["episode_id"]
                grp = eps_grp.create_group(str(eid))
                grp.create_dataset("states", data=ep["states"].astype(np.float32))
                grp.create_dataset("actions", data=ep["actions"].astype(np.float32))

                paths = image_paths.get(eid, [])
                if paths:
                    dt = h5py.string_dtype()
                    grp.create_dataset("image_paths", data=paths, dtype=dt)

        logger.info("Wrote %d episodes to %s", len(episodes), h5_path)
        return h5_path

    @staticmethod
    def _extract_array_column(df, cols: list[str]) -> np.ndarray:
        """Extract a 2D float32 array from DataFrame columns.

        Handles both scalar columns (action.0, action.1, ...) and
        object columns containing numpy arrays (action → [array, array, ...]).
        """
        if len(cols) == 1 and df[cols[0]].dtype == object:
            # Column contains array-like objects (e.g. np.ndarray per row)
            return np.stack(df[cols[0]].values).astype(np.float32)
        return df[cols].values.astype(np.float32)

    @staticmethod
    def _detect_columns(df, prefix: str) -> list[str]:
        """Find DataFrame columns matching a prefix pattern."""
        # Exact match first
        if prefix in df.columns:
            return [prefix]
        # Dotted sub-columns (e.g. action.0, action.1)
        cols = [c for c in df.columns if c.startswith(prefix + ".")]
        if cols:
            return sorted(cols, key=lambda c: c)
        # Underscore variants (e.g. action_0, action_1)
        underscore_prefix = prefix.replace(".", "_")
        cols = [c for c in df.columns if c.startswith(underscore_prefix)]
        return sorted(cols, key=lambda c: c)

    @staticmethod
    def _parse_episode_id_from_filename(stem: str) -> int | None:
        """Parse episode ID from filenames like 'episode_000000'."""
        import re

        match = re.search(r"(\d+)", stem)
        if match:
            return int(match.group(1))
        return None
