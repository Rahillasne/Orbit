"""Backward-compatibility conversions between new and legacy logger models."""

from __future__ import annotations

import datetime

from orbit.logger.schemas import (
    Episode,
    EpisodeFrame,
    EpisodeRecord,
    Outcome,
    StepRecord,
)


def episode_to_legacy(episode: Episode) -> EpisodeRecord:
    """Convert a new *Episode* to a legacy *EpisodeRecord*.

    This allows downstream modules (detector, analyzer, prescriber, vlm,
    dashboard) to keep working with the old model while the logger itself
    uses the new schema.
    """
    steps: list[StepRecord] = []
    for i, frame in enumerate(episode.frames):
        steps.append(
            StepRecord(
                step_index=i,
                timestamp=frame.timestamp,
                observation={"joint_positions": frame.joint_positions},
                action=frame.action,
                reward=frame.reward if frame.reward is not None else 0.0,
                done=(i == len(episode.frames) - 1),
                info=frame.metadata,
                images={},
            )
        )

    if episode.outcome == Outcome.SUCCESS:
        success: bool | None = True
    elif episode.outcome == Outcome.FAILURE:
        success = False
    else:
        success = None

    return EpisodeRecord(
        episode_id=hash(episode.episode_id) % (2**31),
        task=episode.task_name,
        steps=steps,
        metadata=episode.metadata,
        start_time=episode.start_time,
        end_time=episode.end_time,
        total_reward=episode.total_reward,
        success=success,
        num_steps=episode.num_frames,
    )


def legacy_to_episode(
    record: EpisodeRecord,
    robot_id: str = "default",
) -> Episode:
    """Convert a legacy *EpisodeRecord* to a new *Episode*."""
    frames: list[EpisodeFrame] = []
    for step in record.steps:
        joint_positions = step.observation.get("joint_positions", [])
        if not isinstance(joint_positions, list):
            joint_positions = list(joint_positions)

        action = step.action
        if not isinstance(action, list):
            action = list(action)

        frames.append(
            EpisodeFrame(
                timestamp=step.timestamp,
                joint_positions=joint_positions,
                gripper_state=0.0,
                action=action,
                reward=step.reward,
                metadata=step.info,
            )
        )

    if record.success is True:
        outcome = Outcome.SUCCESS
    elif record.success is False:
        outcome = Outcome.FAILURE
    else:
        outcome = Outcome.UNKNOWN

    return Episode(
        task_name=record.task,
        robot_id=robot_id,
        start_time=record.start_time,
        end_time=record.end_time or datetime.datetime.now(),
        frames=frames,
        outcome=outcome,
        metadata=record.metadata,
    )
