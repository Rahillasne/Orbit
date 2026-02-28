#!/usr/bin/env python3
"""Generate a synthetic deployment with 20 episodes for dashboard testing.

Creates:
  - 12 successes: normal joint trajectories, good-quality images
  - 5 "lighting" failures: dark images, some stalling
  - 3 "position" failures: extreme joint positions near workspace bounds

Usage:
    python scripts/generate_synthetic_deployment.py [--output-dir ./test_deployments]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from PIL import Image

from orbit.logger.episode_logger import EpisodeLogger
from orbit.logger.schemas import LoggerConfig, Outcome


def _generate_normal_image(step: int, brightness: int = 160) -> Image.Image:
    """Generate a synthetic image resembling a normal observation."""
    rng = np.random.default_rng(step)
    base = rng.integers(brightness - 40, brightness + 40, (64, 64, 3), dtype=np.uint8)
    # Add a simple shape to simulate an object
    cx, cy = 32, 32
    for dx in range(-5, 6):
        for dy in range(-5, 6):
            if dx * dx + dy * dy <= 25:
                base[cy + dy, cx + dx] = [200, 80, 80]
    return Image.fromarray(base)


def _generate_dark_image(step: int) -> Image.Image:
    """Generate a dark image simulating lighting failure."""
    rng = np.random.default_rng(step + 10000)
    base = rng.integers(10, 40, (64, 64, 3), dtype=np.uint8)
    return Image.fromarray(base)


def _generate_edge_image(step: int) -> Image.Image:
    """Generate an image with an object at the edge (position failure)."""
    rng = np.random.default_rng(step + 20000)
    base = rng.integers(120, 180, (64, 64, 3), dtype=np.uint8)
    # Object at edge of frame
    for dx in range(10):
        for dy in range(-5, 6):
            y = 32 + dy
            x = min(63, 58 + dx)
            if 0 <= y < 64:
                base[y, x] = [80, 80, 200]
    return Image.fromarray(base)


def generate_deployment(output_dir: str = "./test_deployments") -> str:
    """Generate a complete synthetic deployment and return the session path."""
    config = LoggerConfig(
        storage_dir=output_dir,
        task_name="pick_and_place",
        robot_dof=6,
        save_images=True,
        policy_checkpoint="synthetic_policy_v1.pt",
        policy_version="v1.0-synthetic",
        environment_description="sim_tabletop_synthetic",
    )

    rng = np.random.default_rng(42)

    with EpisodeLogger(config) as logger:
        # === 12 Success episodes ===
        for ep_idx in range(12):
            logger.start_episode(task_name="pick_and_place")
            n_frames = rng.integers(30, 60)
            for step in range(n_frames):
                t = step / n_frames
                # Smooth trajectory: sinusoidal joint motion
                joints = (
                    np.sin(np.linspace(0, np.pi, 6) + ep_idx * 0.5 + t * 2)
                    * 0.3
                ).tolist()
                action = (rng.standard_normal(6) * 0.02).tolist()
                gripper = float(np.clip(0.2 + t * 0.6, 0, 1))
                # Progressive reward: small early, big at end
                reward = 0.05 + t * 0.2

                img = _generate_normal_image(step + ep_idx * 100)

                logger.log_frame(
                    joint_positions=joints,
                    gripper_state=gripper,
                    action=action,
                    reward=reward,
                    images={"front": img},
                )
            logger.end_episode(outcome=Outcome.SUCCESS)
            print(f"  [SUCCESS] Episode {ep_idx + 1}/20 ({n_frames} frames)")

        # === 5 Lighting failure episodes ===
        for ep_idx in range(5):
            logger.start_episode(
                task_name="pick_and_place",
                metadata={"failure_type": "lighting"},
            )
            n_frames = rng.integers(15, 35)
            for step in range(n_frames):
                t = step / n_frames
                # Erratic movement in the dark
                joints = (rng.standard_normal(6) * 0.1).tolist()
                # Near-zero action (stalling in the dark)
                action = (rng.standard_normal(6) * 0.001).tolist()
                gripper = float(np.clip(rng.random(), 0, 1))
                reward = -0.3 - rng.random() * 0.2

                img = _generate_dark_image(step + ep_idx * 100)

                logger.log_frame(
                    joint_positions=joints,
                    gripper_state=gripper,
                    action=action,
                    reward=reward,
                    images={"front": img},
                )
            logger.end_episode(
                outcome=Outcome.FAILURE,
                metadata={"failure_type": "lighting"},
            )
            print(
                f"  [FAILURE:lighting] Episode {12 + ep_idx + 1}/20 "
                f"({n_frames} frames)"
            )

        # === 3 Position failure episodes ===
        for ep_idx in range(3):
            logger.start_episode(
                task_name="pick_and_place",
                metadata={"failure_type": "position"},
            )
            n_frames = rng.integers(20, 40)
            for step in range(n_frames):
                t = step / n_frames
                # Extreme joint positions near workspace bounds
                joints = (
                    np.sign(rng.standard_normal(6)) * (2.5 + rng.random(6) * 0.5)
                ).tolist()
                action = (rng.standard_normal(6) * 0.05).tolist()
                # Gripper opens unexpectedly (drop)
                gripper = 0.1 if step < n_frames // 2 else 0.9
                reward = -0.5

                img = _generate_edge_image(step + ep_idx * 100)

                logger.log_frame(
                    joint_positions=joints,
                    gripper_state=gripper,
                    action=action,
                    reward=reward,
                    images={"front": img},
                )
            logger.end_episode(
                outcome=Outcome.FAILURE,
                metadata={"failure_type": "position"},
            )
            print(
                f"  [FAILURE:position] Episode {17 + ep_idx + 1}/20 "
                f"({n_frames} frames)"
            )

        print(f"\nGenerated {logger.num_episodes} episodes")
        print(f"Session ID: {logger.session_id}")
        print(f"Data saved to: {output_dir}")

        return str(logger.session_id)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic deployment data for Orbit dashboard testing."
    )
    parser.add_argument(
        "--output-dir",
        default="./test_deployments",
        help="Output directory for the session .h5 file (default: ./test_deployments)",
    )
    args = parser.parse_args()

    print(f"Generating synthetic deployment in {args.output_dir}...")
    session_id = generate_deployment(args.output_dir)
    print(f"\nDone! Launch the dashboard with:")
    print(f"  orbit dashboard --data-dir {args.output_dir}")


if __name__ == "__main__":
    main()
