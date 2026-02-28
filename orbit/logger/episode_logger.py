"""Core episode logging class for robot learning."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
from PIL import Image

from orbit.logger.schemas import (
    DeploymentSession,
    Episode,
    EpisodeFrame,
    LoggerConfig,
    Outcome,
)
from orbit.logger.storage import HDF5Storage, StorageBackend


class EpisodeLogger:
    """Logger for recording robot learning episodes.

    Designed as a **drop-in addition** to any LeRobot inference script.
    The key design principle is *zero overhead* on the policy inference
    loop — image saving and statistics computation happen on background
    threads.

    Usage::

        config = LoggerConfig(storage_dir="./data", task_name="pick_cup", robot_dof=6)
        with EpisodeLogger(config) as logger:
            logger.start_episode(task_name="pick_cup")
            for step in policy_loop:
                logger.log_frame(
                    joint_positions=[...],
                    gripper_state=0.5,
                    action=[...],
                    reward=0.1,
                    images={"front": front_image},
                )
            logger.end_episode(outcome=Outcome.SUCCESS)
            print(logger.summary())
    """

    def __init__(self, config: LoggerConfig | None = None) -> None:
        self.config = config or LoggerConfig()
        self._storage: StorageBackend = HDF5Storage(self.config)

        # Session
        self._session = DeploymentSession(
            environment_description=self.config.environment_description,
            policy_version=self.config.policy_version,
        )
        self._session_path = self._storage.create_session(self._session)

        # Episode state
        self._current_episode: Episode | None = None
        self._frame_counter: int = 0

        # Completed episodes for summary
        self._completed_episodes: list[Episode] = []

        # Post-episode hooks: callables invoked after each episode ends
        self._post_episode_hooks: list[Callable[[Episode], None]] = []

        # Background image-saving thread
        self._image_queue: queue.Queue[tuple[Path, Image.Image | np.ndarray] | None] = queue.Queue()
        self._image_save_count: int = 0
        self._image_thread = threading.Thread(
            target=self._image_writer_loop, daemon=True, name="orbit-image-writer"
        )
        self._image_thread.start()

    # -- context manager -----------------------------------------------------

    def __enter__(self) -> EpisodeLogger:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # Auto-end an in-progress episode on exception.
        if self._current_episode is not None:
            outcome = Outcome.FAILURE if exc_type is not None else Outcome.UNKNOWN
            self.end_episode(outcome=outcome)
        self.wait_for_images()
        self._shutdown_image_thread()
        self._storage.close()

    # -- background image writer ---------------------------------------------

    def _image_writer_loop(self) -> None:
        """Drain the queue and write images to disk."""
        while True:
            item = self._image_queue.get()
            if item is None:  # shutdown sentinel
                self._image_queue.task_done()
                break
            img_path, image = item
            try:
                img_path.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image.astype(np.uint8))
                image.save(img_path)
                self._image_save_count += 1
            finally:
                self._image_queue.task_done()

    def _shutdown_image_thread(self) -> None:
        self._image_queue.put(None)  # sentinel
        self._image_thread.join(timeout=30)

    def wait_for_images(self) -> None:
        """Block until all queued images have been written to disk."""
        self._image_queue.join()

    # -- post-episode hooks --------------------------------------------------

    def add_post_episode_hook(self, hook: Callable[[Episode], None]) -> None:
        """Register a callback to be invoked after each episode ends.

        The hook receives the finalized ``Episode`` object.  Exceptions
        in hooks are logged but do not propagate (fire-and-forget).
        """
        self._post_episode_hooks.append(hook)

    def remove_post_episode_hook(self, hook: Callable[[Episode], None]) -> None:
        """Remove a previously registered hook."""
        self._post_episode_hooks.remove(hook)

    def _fire_hooks(self, episode: Episode) -> None:
        """Run all registered post-episode hooks."""
        for hook in self._post_episode_hooks:
            try:
                hook(episode)
            except Exception:
                logging.getLogger(__name__).exception("Post-episode hook %s failed", hook)

    # -- episode lifecycle ---------------------------------------------------

    def start_episode(
        self,
        task_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UUID:
        """Start recording a new episode. Returns the episode UUID."""
        if self._current_episode is not None:
            raise RuntimeError(
                f"Episode {self._current_episode.episode_id} is already in progress. "
                "Call end_episode() first."
            )

        self._frame_counter = 0
        episode = Episode(
            task_name=task_name or self.config.task_name,
            robot_id=self.config.robot_id,
            policy_checkpoint=self.config.policy_checkpoint,
            metadata=metadata or {},
        )
        self._current_episode = episode
        self._storage.begin_episode(self._session.session_id, episode)
        return episode.episode_id

    def log_frame(
        self,
        joint_positions: list[float],
        gripper_state: float,
        action: list[float],
        reward: float | None = None,
        images: dict[str, Image.Image | np.ndarray] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a single frame in the current episode.

        Images are dispatched to a background thread so this call returns
        as fast as possible.
        """
        if self._current_episode is None:
            raise RuntimeError("No episode in progress. Call start_episode() first.")

        # DOF validation
        if len(joint_positions) != self.config.robot_dof:
            raise ValueError(
                f"Expected {self.config.robot_dof} joint positions, got {len(joint_positions)}"
            )

        # Build the image path and dispatch saving to the background thread.
        image_path = ""
        if images and self.config.save_images:
            ep_id = str(self._current_episode.episode_id)
            for key, img in images.items():
                img_dest = (
                    Path(self.config.storage_dir)
                    / "images"
                    / ep_id
                    / f"{self._frame_counter:06d}_{key}.png"
                )
                image_path = str(img_dest)
                self._image_queue.put((img_dest, img))

        frame = EpisodeFrame(
            timestamp=time.time(),
            joint_positions=joint_positions,
            gripper_state=gripper_state,
            image_path=image_path,
            action=action,
            reward=reward,
            metadata=metadata or {},
        )

        # Incremental write to storage
        self._storage.append_frame(
            self._session.session_id,
            self._current_episode.episode_id,
            frame,
        )
        self._current_episode.frames.append(frame)
        self._frame_counter += 1

        # Auto-end if max frames reached
        if self._frame_counter >= self.config.max_frames_per_episode:
            self.end_episode(outcome=Outcome.FAILURE, metadata={"auto_ended": True})

    def end_episode(
        self,
        outcome: Outcome = Outcome.UNKNOWN,
        metadata: dict[str, Any] | None = None,
    ) -> Episode:
        """Finish the current episode, finalize in storage, and return it."""
        if self._current_episode is None:
            raise RuntimeError("No episode in progress. Call start_episode() first.")

        import datetime

        episode = self._current_episode
        episode.outcome = outcome
        episode.end_time = datetime.datetime.now()

        if metadata:
            episode.metadata.update(metadata)

        self._storage.end_episode(
            self._session.session_id,
            episode.episode_id,
            outcome,
            episode.end_time,
            metadata,
        )

        self._completed_episodes.append(episode)
        self._session.episodes.append(episode)
        self._current_episode = None

        # Fire post-episode hooks
        self._fire_hooks(episode)

        return episode

    # -- querying ------------------------------------------------------------

    def get_episode(self, episode_id: UUID) -> Episode:
        """Load a previously saved episode by ID."""
        return self._storage.load_episode(self._session.session_id, episode_id)

    def list_episodes(
        self,
        task: str | None = None,
        outcome: Outcome | None = None,
    ) -> list[tuple[UUID, UUID]]:
        """List saved episode IDs, optionally filtered."""
        return self._storage.list_episodes(
            session_id=self._session.session_id,
            task=task,
            outcome=outcome,
        )

    # -- statistics ----------------------------------------------------------

    def summary(self) -> dict[str, dict[str, Any]]:
        """Return per-episode statistics for all completed episodes.

        Returns a dict keyed by episode UUID (as string) with values::

            {
                "duration": float | None,
                "num_frames": int,
                "avg_action_magnitude": float,
                "total_reward": float,
                "outcome": str,
            }
        """
        result: dict[str, dict[str, Any]] = {}
        for ep in self._completed_episodes:
            result[str(ep.episode_id)] = {
                "duration": ep.duration,
                "num_frames": ep.num_frames,
                "avg_action_magnitude": round(ep.avg_action_magnitude, 4),
                "total_reward": round(ep.total_reward, 4),
                "outcome": ep.outcome.value,
            }
        return result

    @property
    def session_id(self) -> UUID:
        """The UUID of the current deployment session."""
        return self._session.session_id

    @property
    def current_episode_id(self) -> UUID | None:
        """The UUID of the currently recording episode, or ``None``."""
        if self._current_episode is not None:
            return self._current_episode.episode_id
        return None

    @property
    def num_episodes(self) -> int:
        """Number of completed episodes in this session."""
        return len(self._completed_episodes)


# ---------------------------------------------------------------------------
# Detector hook factory
# ---------------------------------------------------------------------------


def create_detector_hook(
    pipeline: Any,
    update_outcome: bool = True,
    webhook_url: str | None = None,
) -> Callable[[Episode], None]:
    """Create a post-episode hook that runs a ``DetectorPipeline``.

    Args:
        pipeline: A ``DetectorPipeline`` instance.
        update_outcome: If ``True``, set ``episode.outcome = FAILURE``
            when failures are detected and the outcome was ``UNKNOWN``.
        webhook_url: Optional URL to POST detection results to.

    Returns:
        A callable suitable for :meth:`EpisodeLogger.add_post_episode_hook`.
    """
    log = logging.getLogger(__name__)

    def _hook(episode: Episode) -> None:
        result = pipeline.run(episode)
        if not result.is_failure:
            return

        log.warning(
            "Episode %s: %d failure(s) detected (prob=%.0f%%)",
            episode.episode_id,
            len(result.detections),
            result.failure_probability * 100,
        )

        # Store detections in episode metadata
        episode.metadata["detector_results"] = {
            "failure_probability": result.failure_probability,
            "detections": [
                {
                    "detector": d.detector_name,
                    "confidence": d.confidence,
                    "frame_idx": d.frame_idx,
                    "description": d.description,
                }
                for d in result.detections
            ],
        }

        if update_outcome and episode.outcome == Outcome.UNKNOWN:
            episode.outcome = Outcome.FAILURE

        if webhook_url:
            _fire_webhook(webhook_url, episode, result)

    return _hook


def _fire_webhook(url: str, episode: Episode, result: Any) -> None:
    """Best-effort HTTP POST of detection results."""
    try:
        payload = json.dumps(
            {
                "episode_id": str(episode.episode_id),
                "task_name": episode.task_name,
                "failure_probability": result.failure_probability,
                "num_detections": len(result.detections),
            }
        ).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        logging.getLogger(__name__).warning("Webhook POST to %s failed", url)
