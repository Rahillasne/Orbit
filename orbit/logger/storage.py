"""Storage backends for persisting episode data."""

from __future__ import annotations

import datetime
import json
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from uuid import UUID

import h5py
import numpy as np
from filelock import FileLock
from PIL import Image

from orbit.logger.schemas import (
    DeploymentSession,
    Episode,
    EpisodeFrame,
    EpisodeRecord,
    LoggerConfig,
    Outcome,
    StepRecord,
)

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class StorageBackend(ABC):
    """Abstract base class for episode storage."""

    def __init__(self, config: LoggerConfig) -> None:
        self.config = config
        self.storage_dir = Path(config.storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def create_session(self, session: DeploymentSession) -> Path:
        """Create a new session file on disk."""
        ...

    @abstractmethod
    def begin_episode(self, session_id: UUID, episode: Episode) -> None:
        """Register a new episode inside the session file."""
        ...

    @abstractmethod
    def append_frame(self, session_id: UUID, episode_id: UUID, frame: EpisodeFrame) -> None:
        """Append a single frame to an in-progress episode (incremental write)."""
        ...

    @abstractmethod
    def end_episode(
        self,
        session_id: UUID,
        episode_id: UUID,
        outcome: Outcome,
        end_time: datetime.datetime,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Finalize an episode's metadata."""
        ...

    @abstractmethod
    def save_episode(self, session_id: UUID, episode: Episode) -> Path:
        """Batch-write a complete episode (all frames at once)."""
        ...

    @abstractmethod
    def load_episode(self, session_id: UUID, episode_id: UUID) -> Episode:
        """Load an episode from storage."""
        ...

    @abstractmethod
    def list_episodes(
        self,
        session_id: UUID | None = None,
        task: str | None = None,
        outcome: Outcome | None = None,
        start_date: datetime.datetime | None = None,
        end_date: datetime.datetime | None = None,
    ) -> list[tuple[UUID, UUID]]:
        """Return ``(session_id, episode_id)`` pairs, optionally filtered."""
        ...

    @abstractmethod
    def delete_episode(self, session_id: UUID, episode_id: UUID) -> None:
        """Remove an episode from storage."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release any held resources."""
        ...


# ---------------------------------------------------------------------------
# HDF5 backend (one file per session)
# ---------------------------------------------------------------------------


class HDF5Storage(StorageBackend):
    """HDF5-based storage: one ``.h5`` file per :class:`DeploymentSession`.

    Internal layout::

        session_{uuid}.h5
          attrs: session_id, environment_description, policy_version
          /episodes/{episode_id}/
            attrs: task_name, robot_id, outcome, policy_checkpoint,
                   start_time, end_time, metadata (JSON)
            joint_positions  [N, DOF] float32  (resizable)
            gripper_state    [N]      float32  (resizable)
            actions          [N, D]   float32  (resizable)
            rewards          [N]      float32  (resizable)
            timestamps       [N]      float64  (resizable)
            image_paths      [N]      str      (resizable)
            frame_metadata   [N]      str      (resizable, JSON)
    """

    def _session_path(self, session_id: UUID) -> Path:
        return self.storage_dir / f"session_{session_id}.h5"

    def _lock_path(self, session_id: UUID) -> Path:
        return self.storage_dir / f"session_{session_id}.h5.lock"

    def _lock(self, session_id: UUID) -> FileLock:
        return FileLock(self._lock_path(session_id), timeout=self.config.lock_timeout)

    # -- session lifecycle ---------------------------------------------------

    def create_session(self, session: DeploymentSession) -> Path:
        path = self._session_path(session.session_id)
        with self._lock(session.session_id):
            with h5py.File(path, "w") as f:
                f.attrs["session_id"] = str(session.session_id)
                f.attrs["environment_description"] = session.environment_description
                f.attrs["policy_version"] = session.policy_version
                f.create_group("episodes")
        return path

    # -- episode lifecycle ---------------------------------------------------

    def begin_episode(self, session_id: UUID, episode: Episode) -> None:
        with self._lock(session_id):
            with h5py.File(self._session_path(session_id), "a") as f:
                ep_key = f"episodes/{episode.episode_id}"
                grp = f.create_group(ep_key)
                grp.attrs["task_name"] = episode.task_name
                grp.attrs["robot_id"] = episode.robot_id
                grp.attrs["outcome"] = episode.outcome.value
                grp.attrs["policy_checkpoint"] = episode.policy_checkpoint
                grp.attrs["start_time"] = episode.start_time.isoformat()
                grp.attrs["end_time"] = ""
                grp.attrs["metadata"] = json.dumps(episode.metadata)

    def append_frame(self, session_id: UUID, episode_id: UUID, frame: EpisodeFrame) -> None:
        with self._lock(session_id):
            with h5py.File(self._session_path(session_id), "a") as f:
                grp = f[f"episodes/{episode_id}"]

                jp = np.array(frame.joint_positions, dtype=np.float32)
                act = np.array(frame.action, dtype=np.float32)

                if "timestamps" not in grp:
                    # First frame — create resizable datasets.
                    dof = len(frame.joint_positions)
                    action_dim = len(frame.action)
                    grp.create_dataset(
                        "timestamps",
                        data=np.array([frame.timestamp], dtype=np.float64),
                        maxshape=(None,),
                        chunks=True,
                    )
                    grp.create_dataset(
                        "joint_positions",
                        data=jp.reshape(1, dof),
                        maxshape=(None, dof),
                        chunks=True,
                        dtype=np.float32,
                    )
                    grp.create_dataset(
                        "gripper_state",
                        data=np.array([frame.gripper_state], dtype=np.float32),
                        maxshape=(None,),
                        chunks=True,
                    )
                    grp.create_dataset(
                        "actions",
                        data=act.reshape(1, action_dim),
                        maxshape=(None, action_dim),
                        chunks=True,
                        dtype=np.float32,
                    )
                    grp.create_dataset(
                        "rewards",
                        data=np.array(
                            [frame.reward if frame.reward is not None else np.nan],
                            dtype=np.float32,
                        ),
                        maxshape=(None,),
                        chunks=True,
                    )
                    dt = h5py.string_dtype()
                    grp.create_dataset(
                        "image_paths",
                        data=np.array([frame.image_path], dtype=object),
                        maxshape=(None,),
                        chunks=True,
                        dtype=dt,
                    )
                    grp.create_dataset(
                        "frame_metadata",
                        data=np.array([json.dumps(frame.metadata)], dtype=object),
                        maxshape=(None,),
                        chunks=True,
                        dtype=dt,
                    )
                else:
                    # Append to existing datasets.
                    n = grp["timestamps"].shape[0]
                    numeric_pairs: list[tuple[str, np.ndarray]] = [
                        ("timestamps", np.array([frame.timestamp], dtype=np.float64)),
                        ("joint_positions", jp.reshape(1, -1)),
                        ("gripper_state", np.array([frame.gripper_state], dtype=np.float32)),
                        ("actions", act.reshape(1, -1)),
                        (
                            "rewards",
                            np.array(
                                [frame.reward if frame.reward is not None else np.nan],
                                dtype=np.float32,
                            ),
                        ),
                    ]
                    for ds_name, arr in numeric_pairs:
                        ds = grp[ds_name]
                        new_shape = list(ds.shape)
                        new_shape[0] = n + 1
                        ds.resize(new_shape)
                        ds[n] = arr if arr.ndim == 0 else arr[0]

                    string_pairs: list[tuple[str, str]] = [
                        ("image_paths", frame.image_path),
                        ("frame_metadata", json.dumps(frame.metadata)),
                    ]
                    for ds_name, val in string_pairs:
                        ds = grp[ds_name]
                        ds.resize((n + 1,))
                        ds[n] = val

    def end_episode(
        self,
        session_id: UUID,
        episode_id: UUID,
        outcome: Outcome,
        end_time: datetime.datetime,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._lock(session_id):
            with h5py.File(self._session_path(session_id), "a") as f:
                grp = f[f"episodes/{episode_id}"]
                grp.attrs["outcome"] = outcome.value
                grp.attrs["end_time"] = end_time.isoformat()
                if metadata:
                    existing = json.loads(grp.attrs["metadata"])
                    existing.update(metadata)
                    grp.attrs["metadata"] = json.dumps(existing)

    # -- batch write ---------------------------------------------------------

    def save_episode(self, session_id: UUID, episode: Episode) -> Path:
        path = self._session_path(session_id)
        with self._lock(session_id):
            with h5py.File(path, "a") as f:
                ep_key = f"episodes/{episode.episode_id}"
                if ep_key in f:
                    del f[ep_key]
                grp = f.create_group(ep_key)

                grp.attrs["task_name"] = episode.task_name
                grp.attrs["robot_id"] = episode.robot_id
                grp.attrs["outcome"] = episode.outcome.value
                grp.attrs["policy_checkpoint"] = episode.policy_checkpoint
                grp.attrs["start_time"] = episode.start_time.isoformat()
                grp.attrs["end_time"] = episode.end_time.isoformat() if episode.end_time else ""
                grp.attrs["metadata"] = json.dumps(episode.metadata)

                if episode.frames:
                    timestamps = np.array([fr.timestamp for fr in episode.frames], dtype=np.float64)
                    joint_pos = np.array(
                        [fr.joint_positions for fr in episode.frames], dtype=np.float32
                    )
                    gripper = np.array(
                        [fr.gripper_state for fr in episode.frames], dtype=np.float32
                    )
                    actions = np.array([fr.action for fr in episode.frames], dtype=np.float32)
                    rewards = np.array(
                        [fr.reward if fr.reward is not None else np.nan for fr in episode.frames],
                        dtype=np.float32,
                    )

                    grp.create_dataset("timestamps", data=timestamps)
                    grp.create_dataset("joint_positions", data=joint_pos)
                    grp.create_dataset("gripper_state", data=gripper)
                    grp.create_dataset("actions", data=actions)
                    grp.create_dataset("rewards", data=rewards)

                    dt = h5py.string_dtype()
                    img_paths = [fr.image_path for fr in episode.frames]
                    grp.create_dataset("image_paths", data=img_paths, dtype=dt)
                    meta_strs = [json.dumps(fr.metadata) for fr in episode.frames]
                    grp.create_dataset("frame_metadata", data=meta_strs, dtype=dt)

        return path

    # -- reading -------------------------------------------------------------

    def load_episode(self, session_id: UUID, episode_id: UUID) -> Episode:
        path = self._session_path(session_id)
        if not path.exists():
            raise FileNotFoundError(f"Session file not found: {path}")

        with self._lock(session_id):
            with h5py.File(path, "r") as f:
                ep_key = f"episodes/{episode_id}"
                if ep_key not in f:
                    raise FileNotFoundError(
                        f"Episode {episode_id} not found in session {session_id}"
                    )
                grp = f[ep_key]
                return self._read_episode(grp, episode_id)

    def _read_episode(self, grp: h5py.Group, episode_id: UUID) -> Episode:
        meta = json.loads(grp.attrs["metadata"])
        end_time_str = str(grp.attrs["end_time"])

        frames: list[EpisodeFrame] = []
        if "timestamps" in grp:
            timestamps = grp["timestamps"][:]
            joint_pos = grp["joint_positions"][:]
            gripper = grp["gripper_state"][:]
            actions = grp["actions"][:]
            rewards = grp["rewards"][:]
            img_paths = [s.decode() if isinstance(s, bytes) else s for s in grp["image_paths"][:]]
            frame_metas = [
                s.decode() if isinstance(s, bytes) else s for s in grp["frame_metadata"][:]
            ]

            for i in range(len(timestamps)):
                reward_val = float(rewards[i])
                frames.append(
                    EpisodeFrame(
                        timestamp=float(timestamps[i]),
                        joint_positions=joint_pos[i].tolist(),
                        gripper_state=float(np.clip(gripper[i], 0.0, 1.0)),
                        image_path=img_paths[i],
                        action=actions[i].tolist(),
                        reward=None if np.isnan(reward_val) else reward_val,
                        metadata=json.loads(frame_metas[i]),
                    )
                )

        return Episode(
            episode_id=episode_id,
            task_name=str(grp.attrs["task_name"]),
            robot_id=str(grp.attrs["robot_id"]),
            start_time=datetime.datetime.fromisoformat(str(grp.attrs["start_time"])),
            end_time=(datetime.datetime.fromisoformat(end_time_str) if end_time_str else None),
            frames=frames,
            outcome=Outcome(str(grp.attrs["outcome"])),
            policy_checkpoint=str(grp.attrs["policy_checkpoint"]),
            metadata=meta,
        )

    # -- listing / filtering -------------------------------------------------

    def list_episodes(
        self,
        session_id: UUID | None = None,
        task: str | None = None,
        outcome: Outcome | None = None,
        start_date: datetime.datetime | None = None,
        end_date: datetime.datetime | None = None,
    ) -> list[tuple[UUID, UUID]]:
        results: list[tuple[UUID, UUID]] = []

        if session_id is not None:
            session_files = [self._session_path(session_id)]
        else:
            session_files = sorted(self.storage_dir.glob("session_*.h5"))

        for spath in session_files:
            if not spath.exists():
                continue
            sid = UUID(spath.stem.replace("session_", ""))
            with self._lock(sid):
                with h5py.File(spath, "r") as f:
                    if "episodes" not in f:
                        continue
                    for ep_name in f["episodes"]:
                        grp = f[f"episodes/{ep_name}"]
                        if task and str(grp.attrs["task_name"]) != task:
                            continue
                        if outcome and str(grp.attrs["outcome"]) != outcome.value:
                            continue
                        if start_date or end_date:
                            st = str(grp.attrs["start_time"])
                            ep_start = datetime.datetime.fromisoformat(st)
                            if start_date and ep_start < start_date:
                                continue
                            if end_date and ep_start > end_date:
                                continue
                        results.append((sid, UUID(ep_name)))

        return results

    def delete_episode(self, session_id: UUID, episode_id: UUID) -> None:
        path = self._session_path(session_id)
        if not path.exists():
            return
        with self._lock(session_id):
            with h5py.File(path, "a") as f:
                ep_key = f"episodes/{episode_id}"
                if ep_key in f:
                    del f[ep_key]

        # Clean up images on disk
        img_dir = self.storage_dir / "images" / str(episode_id)
        if img_dir.exists():
            shutil.rmtree(img_dir)

    def close(self) -> None:
        pass  # No persistent handles to release


# ---------------------------------------------------------------------------
# LeRobot exporter
# ---------------------------------------------------------------------------


class LeRobotExporter:
    """Export Orbit episodes to LeRobot-compatible format (Parquet + images)."""

    def export_episode(self, episode: Episode, output_dir: Path) -> Path:
        """Export a single episode as a Parquet file with an images directory.

        Layout::

            output_dir/
              data/episode_{id}.parquet
              images/{camera_key}/episode_{id}/frame_{i:06d}.png
        """
        output_dir = Path(output_dir)
        data_dir = output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        ep_id_short = str(episode.episode_id)[:8]

        # Build tabular data
        rows: list[dict[str, Any]] = []
        for i, frame in enumerate(episode.frames):
            row: dict[str, Any] = {
                "frame_index": i,
                "timestamp": frame.timestamp,
                "gripper_state": frame.gripper_state,
                "reward": frame.reward,
                "image_path": frame.image_path,
            }
            for j, jp in enumerate(frame.joint_positions):
                row[f"joint_position_{j}"] = jp
            for j, a in enumerate(frame.action):
                row[f"action_{j}"] = a
            rows.append(row)

        import pandas as pd

        parquet_path = data_dir / f"episode_{ep_id_short}.parquet"
        pd.DataFrame(rows).to_parquet(parquet_path, compression="snappy", index=False)

        # Copy images referenced by image_path
        images_base = output_dir / "images"
        for i, frame in enumerate(episode.frames):
            if frame.image_path:
                src = Path(frame.image_path)
                if src.exists():
                    cam_key = src.stem.split("_", 1)[-1] if "_" in src.stem else "front"
                    dest_dir = images_base / cam_key / f"episode_{ep_id_short}"
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest = dest_dir / f"frame_{i:06d}.png"
                    shutil.copy2(src, dest)

        return parquet_path

    def export_session(self, session: DeploymentSession, output_dir: Path) -> Path:
        """Export all episodes in a session."""
        output_dir = Path(output_dir)
        for episode in session.episodes:
            self.export_episode(episode, output_dir)
        return output_dir


# ---------------------------------------------------------------------------
# DEPRECATED: Legacy storage backends.
# ---------------------------------------------------------------------------


class LegacyHDF5Storage:
    """HDF5 storage backend retained for reading old per-episode files."""

    def __init__(self, config: LoggerConfig) -> None:
        self.config = config
        self.storage_dir = Path(config.storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _episode_path(self, episode_id: int) -> Path:
        return self.storage_dir / f"episode_{episode_id:06d}.h5"

    def save_episode(self, episode: EpisodeRecord) -> Path:
        path = self._episode_path(episode.episode_id)
        with h5py.File(path, "w") as f:
            f.attrs["episode_id"] = episode.episode_id
            f.attrs["task"] = episode.task
            f.attrs["total_reward"] = episode.total_reward
            f.attrs["success"] = episode.success if episode.success is not None else False
            f.attrs["num_steps"] = episode.num_steps
            f.attrs["start_time"] = episode.start_time.isoformat()
            f.attrs["end_time"] = episode.end_time.isoformat() if episode.end_time else ""
            f.attrs["metadata"] = json.dumps(episode.metadata)

            if episode.steps:
                timestamps = np.array([s.timestamp for s in episode.steps], dtype=np.float64)
                rewards = np.array([s.reward for s in episode.steps], dtype=np.float32)
                dones = np.array([s.done for s in episode.steps], dtype=bool)
                f.create_dataset("timestamps", data=timestamps)
                f.create_dataset("rewards", data=rewards)
                f.create_dataset("dones", data=dones)

                actions = np.array([_to_list(s.action) for s in episode.steps], dtype=np.float32)
                f.create_dataset("actions", data=actions)

                obs_group = f.create_group("observations")
                obs_keys = episode.steps[0].observation.keys()
                for key in obs_keys:
                    values = [_to_list(s.observation[key]) for s in episode.steps]
                    obs_group.create_dataset(key, data=np.array(values, dtype=np.float32))

                infos = [json.dumps(s.info) for s in episode.steps]
                f.create_dataset("infos", data=infos)

                if self.config.save_images:
                    img_group = f.create_group("images")
                    image_keys = _collect_image_keys(episode.steps)
                    for img_key in image_keys:
                        cam_group = img_group.create_group(img_key)
                        for i, step in enumerate(episode.steps):
                            if img_key in step.images:
                                img_arr = _image_to_array(step.images[img_key])
                                cam_group.create_dataset(
                                    f"step_{i:06d}",
                                    data=img_arr,
                                    compression="gzip",
                                    compression_opts=4,
                                )
        return path

    def load_episode(self, episode_id: int) -> EpisodeRecord:
        path = self._episode_path(episode_id)
        if not path.exists():
            raise FileNotFoundError(f"Episode {episode_id} not found at {path}")

        with h5py.File(path, "r") as f:
            metadata = json.loads(f.attrs["metadata"])
            end_time_str = f.attrs["end_time"]

            steps: list[StepRecord] = []
            if "timestamps" in f:
                timestamps = f["timestamps"][:]
                rewards = f["rewards"][:]
                dones = f["dones"][:]
                actions = f["actions"][:]
                obs_keys = list(f["observations"].keys()) if "observations" in f else []
                obs_data = {k: f["observations"][k][:] for k in obs_keys}
                infos = [json.loads(s) for s in f["infos"][:]] if "infos" in f else []
                image_keys = list(f["images"].keys()) if "images" in f else []

                for i in range(len(timestamps)):
                    observation = {k: obs_data[k][i].tolist() for k in obs_keys}
                    images: dict[str, Any] = {}
                    for img_key in image_keys:
                        ds_name = f"step_{i:06d}"
                        if ds_name in f["images"][img_key]:
                            img_arr = f["images"][img_key][ds_name][:]
                            images[img_key] = Image.fromarray(img_arr)

                    steps.append(
                        StepRecord(
                            step_index=i,
                            timestamp=float(timestamps[i]),
                            observation=observation,
                            action=actions[i].tolist(),
                            reward=float(rewards[i]),
                            done=bool(dones[i]),
                            info=infos[i] if i < len(infos) else {},
                            images=images,
                        )
                    )

            return EpisodeRecord(
                episode_id=int(f.attrs["episode_id"]),
                task=str(f.attrs["task"]),
                steps=steps,
                metadata=metadata,
                start_time=datetime.datetime.fromisoformat(str(f.attrs["start_time"])),
                end_time=(datetime.datetime.fromisoformat(end_time_str) if end_time_str else None),
                total_reward=float(f.attrs["total_reward"]),
                success=bool(f.attrs["success"]),
                num_steps=int(f.attrs["num_steps"]),
            )

    def list_episodes(self) -> list[int]:
        episodes: list[int] = []
        for path in self.storage_dir.glob("episode_*.h5"):
            try:
                ep_id = int(path.stem.split("_")[1])
                episodes.append(ep_id)
            except (ValueError, IndexError):
                continue
        return sorted(episodes)

    def delete_episode(self, episode_id: int) -> None:
        path = self._episode_path(episode_id)
        if path.exists():
            path.unlink()


class LegacyParquetStorage:
    """Parquet storage backend retained for reading old per-episode files."""

    def __init__(self, config: LoggerConfig) -> None:
        self.config = config
        self.storage_dir = Path(config.storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _episode_path(self, episode_id: int) -> Path:
        return self.storage_dir / f"episode_{episode_id:06d}.parquet"

    def _meta_path(self, episode_id: int) -> Path:
        return self.storage_dir / f"episode_{episode_id:06d}_meta.json"

    def _images_dir(self, episode_id: int) -> Path:
        return self.storage_dir / "images" / f"episode_{episode_id:06d}"

    def save_episode(self, episode: EpisodeRecord) -> Path:
        import pandas as pd

        path = self._episode_path(episode.episode_id)
        rows: list[dict[str, Any]] = []
        for step in episode.steps:
            row: dict[str, Any] = {
                "step_index": step.step_index,
                "timestamp": step.timestamp,
                "reward": step.reward,
                "done": step.done,
            }
            for key, value in step.observation.items():
                values = _to_list(value)
                if isinstance(values, list):
                    for j, v in enumerate(values):
                        row[f"obs.{key}_{j}"] = v
                else:
                    row[f"obs.{key}"] = values
            action_vals = _to_list(step.action)
            if isinstance(action_vals, list):
                for j, v in enumerate(action_vals):
                    row[f"action_{j}"] = v
            else:
                row["action_0"] = action_vals
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_parquet(path, compression="snappy", index=False)

        if self.config.save_images:
            image_keys = _collect_image_keys(episode.steps)
            if image_keys:
                img_dir = self._images_dir(episode.episode_id)
                img_dir.mkdir(parents=True, exist_ok=True)
                for i, step in enumerate(episode.steps):
                    for img_key in image_keys:
                        if img_key in step.images:
                            img = step.images[img_key]
                            if isinstance(img, np.ndarray):
                                img = Image.fromarray(img)
                            img.save(img_dir / f"step_{i:06d}_{img_key}.png")

        meta = {
            "episode_id": episode.episode_id,
            "task": episode.task,
            "total_reward": episode.total_reward,
            "success": episode.success,
            "num_steps": episode.num_steps,
            "start_time": episode.start_time.isoformat(),
            "end_time": episode.end_time.isoformat() if episode.end_time else None,
            "metadata": episode.metadata,
        }
        self._meta_path(episode.episode_id).write_text(json.dumps(meta, indent=2))
        return path

    def load_episode(self, episode_id: int) -> EpisodeRecord:
        import pandas as pd

        path = self._episode_path(episode_id)
        meta_path = self._meta_path(episode_id)
        if not path.exists():
            raise FileNotFoundError(f"Episode {episode_id} not found at {path}")

        df = pd.read_parquet(path)
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        steps: list[StepRecord] = []
        obs_cols = [c for c in df.columns if c.startswith("obs.")]
        action_cols = sorted([c for c in df.columns if c.startswith("action_")])
        obs_key_groups: dict[str, list[str]] = {}
        for col in obs_cols:
            parts = col[4:].rsplit("_", 1)
            key = parts[0]
            obs_key_groups.setdefault(key, []).append(col)

        img_dir = self._images_dir(episode_id)
        for _, row in df.iterrows():
            observation: dict[str, Any] = {}
            for key, cols in obs_key_groups.items():
                sorted_cols = sorted(cols)
                observation[key] = [float(row[c]) for c in sorted_cols]
            action = [float(row[c]) for c in action_cols]
            images: dict[str, Any] = {}
            if img_dir.exists():
                step_idx = int(row["step_index"])
                for img_path in img_dir.glob(f"step_{step_idx:06d}_*.png"):
                    cam_key = img_path.stem.split("_", 2)[-1]
                    images[cam_key] = Image.open(img_path)
            steps.append(
                StepRecord(
                    step_index=int(row["step_index"]),
                    timestamp=float(row["timestamp"]),
                    observation=observation,
                    action=action,
                    reward=float(row["reward"]),
                    done=bool(row["done"]),
                    images=images,
                )
            )

        return EpisodeRecord(
            episode_id=meta.get("episode_id", episode_id),
            task=meta.get("task", "unknown"),
            steps=steps,
            metadata=meta.get("metadata", {}),
            start_time=(
                datetime.datetime.fromisoformat(meta["start_time"])
                if "start_time" in meta
                else datetime.datetime.now()
            ),
            end_time=(
                datetime.datetime.fromisoformat(meta["end_time"]) if meta.get("end_time") else None
            ),
            total_reward=meta.get("total_reward", 0.0),
            success=meta.get("success"),
            num_steps=meta.get("num_steps", len(steps)),
        )

    def list_episodes(self) -> list[int]:
        episodes: list[int] = []
        for path in self.storage_dir.glob("episode_*.parquet"):
            try:
                ep_id = int(path.stem.split("_")[1])
                episodes.append(ep_id)
            except (ValueError, IndexError):
                continue
        return sorted(episodes)

    def delete_episode(self, episode_id: int) -> None:
        for path in [self._episode_path(episode_id), self._meta_path(episode_id)]:
            if path.exists():
                path.unlink()
        img_dir = self._images_dir(episode_id)
        if img_dir.exists():
            shutil.rmtree(img_dir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_list(value: Any) -> Any:
    """Convert numpy arrays or scalars to Python lists."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _image_to_array(img: Any) -> np.ndarray:
    """Convert a PIL Image or numpy array to uint8 numpy array."""
    if isinstance(img, Image.Image):
        return np.array(img, dtype=np.uint8)
    if isinstance(img, np.ndarray):
        return img.astype(np.uint8)
    raise TypeError(f"Unsupported image type: {type(img)}")


def _collect_image_keys(steps: list[StepRecord]) -> list[str]:
    """Collect all unique image keys across steps."""
    keys: set[str] = set()
    for step in steps:
        keys.update(step.images.keys())
    return sorted(keys)
