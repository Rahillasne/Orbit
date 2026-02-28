"""
Orbit Quickstart Example
========================
Demonstrates the logger with a mock policy loop.
This is the first thing new users see — keep it simple.
"""

import numpy as np
from PIL import Image

from orbit.logger.episode_logger import EpisodeLogger
from orbit.logger.schemas import LoggerConfig, Outcome


def mock_policy(observation: dict) -> list[float]:
    """Fake policy that returns random actions."""
    return (np.random.randn(6) * 0.01).tolist()


def main():
    # 1. Configure the logger
    config = LoggerConfig(
        storage_dir="./quickstart_data",
        task_name="pick_and_place",
        robot_dof=6,
        policy_checkpoint="policy_v1.pt",
        policy_version="v1.0",
        environment_description="sim_tabletop",
    )

    # 2. Use the logger as a context manager
    with EpisodeLogger(config) as logger:
        for ep_idx in range(3):
            episode_id = logger.start_episode()
            print(f"Recording episode {episode_id}...")

            # Simulated policy inference loop
            for step in range(50):
                joint_pos = (np.random.randn(6) * 0.1).tolist()
                action = mock_policy({"joint_positions": joint_pos})
                gripper = float(np.clip(np.random.random(), 0, 1))
                image = Image.fromarray(
                    np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                )

                logger.log_frame(
                    joint_positions=joint_pos,
                    gripper_state=gripper,
                    action=action,
                    reward=0.1 if step < 40 else 1.0,
                    images={"front": image},
                )

            episode = logger.end_episode(outcome=Outcome.SUCCESS)
            print(
                f"  Frames: {episode.num_frames}, "
                f"Reward: {episode.total_reward:.1f}, "
                f"Duration: {episode.duration:.2f}s"
            )

        # 3. Print session summary
        print("\nSession summary:")
        for ep_id, stats in logger.summary().items():
            print(f"  {ep_id[:8]}...: {stats}")

        print(f"\nTotal episodes: {logger.num_episodes}")
        print(f"Data saved to: {config.storage_dir}")


if __name__ == "__main__":
    main()
