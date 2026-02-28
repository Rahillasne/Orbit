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
import pandas as pd
from filelock import FileLock
from PIL import Image

from orbit_modules.schemas import (
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
    def create_session(self, session: DeploymentSession) -> Path: ...

    @abstractmethod
    def begin_episode(self, session_id: UUID, episode: Episode) -> None: ...

    @abstractmethod
    def append_frame(self, session_id: UUID, episode_id: UUID, frame: EpisodeFrame) -> None: ...

    @abstractmethod
    def end_episode(
        self,
        session_id: UUID,
        episode_id: UUID,
        outcome: Outcome,
        end_time: datetime.datetime,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...

    @abstractmethod
    def save_episode(self, session_id: UUID, episode: Episode) -> Path: ...

    @abstractmethod
    def load_episode(self, session_id: UUID, episode_id: UUID) -> Episode: ...

    @abstractmethod
    def list_episodes(
        self,
        session_id: UUID | None = None,
        task: str | None = None,
        outcome: Outcome | None = None,
        start_date: datetime.datetime | None = None,
        end_date: datetime.datetime | None = None,
    ) -> list[tuple[UUID, UUID]]: ...

    @abstractmethod
    def delete_episode(self, session_id: UUID, episode_id: UUID) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# HDF5 backend (one file per session)
# ---------------------------------------------------------------------------


class HDF5Storage(StorageBackend):
    """HDF5-based storage: one .h5 file per DeploymentSession."""

    def _session_path(self, session_id: UUID) -> Path:
        return self.storage_dir / f"session_{session_id}.h5"

    def _lock_path(self, session_id: UUID) -> Path:
        return self.storage_dir / f"session_{session_id}.h5.lock"

    def _lock(self, session_id: UUID) -> FileLock:
        return FileLock(self._lock_path(session_id), timeout=self.config.lock_timeout)

    def create_session(self, session: DeploymentSession) -> Path:
        path = self._session_path(session.session_id)
        with self._lock(session.session_id):
            with h5py.File(path, "w") as f:
                f.attrs["session_id"] = str(session.session_id)
                f.attrs["environment_description"] = session.environment_description
                f.attrs["policy_version"] = session.policy_version
                f.create_group("episodes")
        return path

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
                    dof = len(frame.joint_positions)
                    action_dim = len(frame.action)
                    grp.create_dataset("timestamps", data=np.array([frame.timestamp], dtype=np.float64), maxshape=(None,), chunks=True)
                    grp.create_dataset("joint_positions", data=jp.reshape(1, dof), maxshape=(None, dof), chunks=True, dtype=np.float32)
                    grp.create_dataset("gripper_state", data=np.array([frame.gripper_state], dtype=np.float32), maxshape=(None,), chunks=True)
                    grp.create_dataset("actions", data=act.reshape(1, action_dim), maxshape=(None, action_dim), chunks=True, dtype=np.float32)
                    grp.create_dataset("rewards", data=np.array([frame.reward if frame.reward is not None else np.nan], dtype=np.float32), maxshape=(None,), chunks=True)
                    dt = h5py.string_dtype()
                    grp.create_dataset("image_paths", data=np.array([frame.image_path], dtype=object), maxshape=(None,), chunks=True, dtype=dt)
                    grp.create_dataset("frame_metadata", data=np.array([json.dumps(frame.metadata)], dtype=object), maxshape=(None,), chunks=True, dtype=dt)
                else:
                    n = grp["timestamps"].shape[0]
                    numeric_pairs = [
                        ("timestamps", np.array([frame.timestamp], dtype=np.float64)),
                        ("joint_positions", jp.reshape(1, -1)),
                        ("gripper_state", np.array([frame.gripper_state], dtype=np.float32)),
                        ("actions", act.reshape(1, -1)),
                        ("rewards", np.array([frame.reward if frame.reward is not None else np.nan], dtype=np.float32)),
                    ]
                    for ds_name, arr in numeric_pairs:
                        ds = grp[ds_name]
                        new_shape = list(ds.shape)
                        new_shape[0] = n + 1
                        ds.resize(new_shape)
                        ds[n] = arr if arr.ndim == 0 else arr[0]
                    string_pairs = [
                        ("image_paths", frame.image_path),
                        ("frame_metadata", json.dumps(frame.metadata)),
                    ]
                    for ds_name, val in string_pairs:
                        ds = grp[ds_name]
                        ds.resize((n + 1,))
                        ds[n] = val

    def end_episode(self, session_id: UUID, episode_id: UUID, outcome: Outcome, end_time: datetime.datetime, metadata: dict[str, Any] | None = None) -> None:
        with self._lock(session_id):
            with h5py.File(self._session_path(session_id), "a") as f:
                grp = f[f"episodes/{episode_id}"]
                grp.attrs["outcome"] = outcome.value
                grp.attrs["end_time"] = end_time.isoformat()
                if metadata:
                    existing = json.loads(grp.attrs["metadata"])
                    existing.update(metadata)
                    grp.attrs["metadata"] = json.dumps(existing)

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
                    joint_pos = np.array([fr.joint_positions for fr in episode.frames], dtype=np.float32)
                    gripper = np.array([fr.gripper_state for fr in episode.frames], dtype=np.float32)
                    actions = np.array([fr.action for fr in episode.frames], dtype=np.float32)
                    rewards = np.array([fr.reward if fr.reward is not None else np.nan for fr in episode.frames], dtype=np.float32)
                    grp.create_dataset("timestamps", data=timestamps)
                    grp.create_dataset("joint_positions", data=joint_pos)
                    grp.create_dataset("gripper_state", data=gripper)
                    grp.create_dataset("actions", data=actions)
                    grp.create_dataset("rewards", data=rewards)
                    dt = h5py.string_dtype()
                    grp.create_dataset("image_paths", data=[fr.image_path for fr in episode.frames], dtype=dt)
                    grp.create_dataset("frame_metadata", data=[json.dumps(fr.metadata) for fr in episode.frames], dtype=dt)
        return path

    def load_episode(self, session_id: UUID, episode_id: UUID) -> Episode:
        path = self._session_path(session_id)
        if not path.exists():
            raise FileNotFoundError(f"Session file not found: {path}")
        with self._lock(session_id):
            with h5py.File(path, "r") as f:
                ep_key = f"episodes/{episode_id}"
                if ep_key not in f:
                    raise FileNotFoundError(f"Episode {episode_id} not found in session {session_id}")
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
            frame_metas = [s.decode() if isinstance(s, bytes) else s for s in grp["frame_metadata"][:]]
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

    def list_episodes(self, session_id: UUID | None = None, task: str | None = None, outcome: Outcome | None = None, start_date: datetime.datetime | None = None, end_date: datetime.datetime | None = None) -> list[tuple[UUID, UUID]]:
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
        img_dir = self.storage_dir / "images" / str(episode_id)
        if img_dir.exists():
            shutil.rmtree(img_dir)

    def close(self) -> None:
        pass
