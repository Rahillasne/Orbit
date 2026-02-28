#!/usr/bin/env python3
"""
Test 6: Performance Benchmarking — Real-world scenario timing.
"""
import os
import sys
import time
import tempfile
import subprocess
from pathlib import Path
from uuid import uuid4

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS = {}


def bench_episode_logging():
    """1. Episode logging speed. Target: < 1ms per frame overhead."""
    from orbit.logger.episode_logger import EpisodeLogger
    from orbit.logger.schemas import LoggerConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        config = LoggerConfig(storage_dir=tmpdir, task_name="bench", save_images=False)
        logger = EpisodeLogger(config)

        logger.start_episode()

        n_frames = 1000
        start = time.time()
        for i in range(n_frames):
            logger.log_frame(
                joint_positions=[0.1 * i] * 6,
                gripper_state=0.5,
                action=[0.01] * 6,
                reward=0.1,
                images={},
                metadata={},
            )
        elapsed = time.time() - start
        ms_per_frame = elapsed / n_frames * 1000
        from orbit.logger.schemas import Outcome
        logger.end_episode(outcome=Outcome.SUCCESS)

        RESULTS["log_frame_ms"] = ms_per_frame
        print(f"  Frame logging: {ms_per_frame:.3f} ms/frame ({n_frames} frames in {elapsed:.2f}s)")


def bench_episode_logging_with_images():
    """1b. Episode logging with images. Target: < 5ms per frame."""
    from orbit.logger.episode_logger import EpisodeLogger
    from orbit.logger.schemas import LoggerConfig
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmpdir:
        config = LoggerConfig(storage_dir=tmpdir, task_name="bench_img", save_images=True)
        logger = EpisodeLogger(config)

        # Create a reusable 640x480 image
        img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))

        logger.start_episode()

        n_frames = 200
        start = time.time()
        for i in range(n_frames):
            logger.log_frame(
                joint_positions=[0.1 * i] * 6,
                gripper_state=0.5,
                action=[0.01] * 6,
                reward=0.1,
                images={"front": img},
                metadata={},
            )
        elapsed = time.time() - start
        ms_per_frame = elapsed / n_frames * 1000
        from orbit.logger.schemas import Outcome
        logger.end_episode(outcome=Outcome.SUCCESS)

        RESULTS["log_frame_with_img_ms"] = ms_per_frame
        print(f"  Frame logging (w/ images): {ms_per_frame:.3f} ms/frame ({n_frames} frames in {elapsed:.2f}s)")


def bench_detector_pipeline():
    """4. Detector pipeline speed. Target: < 100ms per episode."""
    from orbit.logger.schemas import LoggerConfig
    from orbit.logger.storage import HDF5Storage
    from orbit.detector.heuristic import (
        DetectorPipeline,
        GripperDropDetector,
        StallDetector,
        OutOfBoundsDetector,
        TimeoutDetector,
        RewardThresholdDetector,
    )
    from uuid import UUID

    data_dir = "/tmp/orbit-test-data"
    session_file = list(Path(data_dir).glob("session_*.h5"))[0]
    config = LoggerConfig(storage_dir=data_dir)
    storage = HDF5Storage(config)
    session_id = UUID(session_file.stem.replace("session_", ""))
    episode_list = storage.list_episodes(session_id=session_id)
    episodes = [storage.load_episode(sid, eid) for sid, eid in episode_list]

    pipeline = DetectorPipeline(
        detectors=[
            GripperDropDetector(),
            StallDetector(),
            OutOfBoundsDetector(),
            TimeoutDetector(),
            RewardThresholdDetector(),
        ]
    )

    start = time.time()
    results = pipeline.run_batch(episodes)
    elapsed = time.time() - start
    ms_per_episode = elapsed / len(episodes) * 1000

    RESULTS["detect_per_episode_ms"] = ms_per_episode
    print(f"  Detection pipeline: {ms_per_episode:.1f} ms/episode ({len(episodes)} episodes in {elapsed:.3f}s)")


def bench_dashboard_start():
    """5. Dashboard initial load time. Target: < 5 seconds to first render."""
    import requests

    # Kill any existing
    os.system("lsof -ti:8503 | xargs kill 2>/dev/null")
    time.sleep(1)

    start = time.time()
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "orbit/dashboard/app.py",
         "--server.port", "8503", "--server.headless", "true",
         "--", "--data-dir", "/tmp/orbit-test-data"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=str(Path(__file__).resolve().parent.parent),
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent)},
    )

    # Poll until it responds
    ready = False
    for _ in range(60):  # max 60 seconds
        try:
            r = requests.get("http://localhost:8503/_stcore/health", timeout=2)
            if r.status_code == 200:
                ready = True
                break
        except Exception:
            pass
        time.sleep(1)

    elapsed = time.time() - start

    if ready:
        RESULTS["dashboard_startup_sec"] = elapsed
        print(f"  Dashboard startup: {elapsed:.1f}s")
    else:
        RESULTS["dashboard_startup_sec"] = float("inf")
        print(f"  Dashboard startup: FAILED (timed out after {elapsed:.1f}s)")

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def bench_full_pipeline():
    """6. Full pipeline speed. Target: < 60 seconds on CPU for 20 episodes."""
    from orbit.logger.schemas import LoggerConfig
    from orbit.logger.storage import HDF5Storage
    from orbit.detector.heuristic import (
        DetectorPipeline,
        GripperDropDetector,
        StallDetector,
        OutOfBoundsDetector,
        TimeoutDetector,
        RewardThresholdDetector,
    )
    from orbit.detector.legacy import DetectionResult
    from orbit.prescriber.prescriber import Prescriber
    from uuid import UUID

    data_dir = "/tmp/orbit-test-data"
    session_file = list(Path(data_dir).glob("session_*.h5"))[0]

    start = time.time()

    # Load
    t0 = time.time()
    config = LoggerConfig(storage_dir=data_dir)
    storage = HDF5Storage(config)
    session_id = UUID(session_file.stem.replace("session_", ""))
    episode_list = storage.list_episodes(session_id=session_id)
    episodes = [storage.load_episode(sid, eid) for sid, eid in episode_list]
    load_time = time.time() - t0

    # Detect
    t0 = time.time()
    pipeline = DetectorPipeline(
        detectors=[
            GripperDropDetector(),
            StallDetector(),
            OutOfBoundsDetector(),
            TimeoutDetector(),
            RewardThresholdDetector(),
        ]
    )
    results = pipeline.run_batch(episodes)
    detect_time = time.time() - t0

    # Prescribe
    t0 = time.time()
    legacy_results = []
    for ep, result in zip(episodes, results):
        if result.is_failure:
            legacy_results.append(
                DetectionResult(
                    episode_id=hash(str(ep.episode_id)) % (2**31),
                    is_failure=True,
                    failure_reasons=[d.description for d in result.detections],
                    confidence=result.failure_probability,
                )
            )
    prescriber = Prescriber()
    report = prescriber.prescribe(legacy_results)
    prescribe_time = time.time() - t0

    total_time = time.time() - start
    RESULTS["full_pipeline_sec"] = total_time

    print(f"  Full pipeline breakdown:")
    print(f"    Load:      {load_time:.2f}s")
    print(f"    Detect:    {detect_time:.3f}s")
    print(f"    Prescribe: {prescribe_time:.3f}s")
    print(f"    TOTAL:     {total_time:.2f}s")


def print_summary():
    """Print results table."""
    print("\n" + "=" * 70)
    print("PERFORMANCE RESULTS")
    print("=" * 70)

    benchmarks = [
        ("log_frame_ms", "Frame logging (no images)", 1.0, "ms"),
        ("log_frame_with_img_ms", "Frame logging (with images)", 5.0, "ms"),
        ("detect_per_episode_ms", "Detection per episode", 100.0, "ms"),
        ("dashboard_startup_sec", "Dashboard startup", 5.0, "s"),
        ("full_pipeline_sec", "Full pipeline (20 ep)", 60.0, "s"),
    ]

    print(f"\n{'Metric':<35} {'Target':>10} {'Actual':>10} {'Status':>8}")
    print("-" * 70)

    for key, label, target, unit in benchmarks:
        actual = RESULTS.get(key)
        if actual is None:
            status = "SKIP"
            actual_str = "N/A"
        elif actual <= target:
            status = "✅"
            actual_str = f"{actual:.2f}{unit}"
        elif actual <= target * 2:
            status = "⚠️"
            actual_str = f"{actual:.2f}{unit}"
        elif actual <= target * 5:
            status = "⚠️⚠️"
            actual_str = f"{actual:.2f}{unit}"
        else:
            status = "❌"
            actual_str = f"{actual:.2f}{unit}"

        print(f"  {label:<33} {target:>8.1f}{unit:>2} {actual_str:>10} {status:>8}")


def main():
    print("ORBIT v1.0 — Performance Benchmarking")
    print("=" * 60)

    print("\n1. Episode Logging (no images)")
    bench_episode_logging()

    print("\n2. Episode Logging (with images)")
    bench_episode_logging_with_images()

    print("\n3. Detector Pipeline")
    bench_detector_pipeline()

    print("\n4. Dashboard Startup")
    bench_dashboard_start()

    print("\n5. Full Pipeline (load → detect → prescribe)")
    bench_full_pipeline()

    print_summary()


if __name__ == "__main__":
    main()
