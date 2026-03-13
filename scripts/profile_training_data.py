#!/usr/bin/env python3
"""Profile all training datasets locally and cache features.

Runs on CPU/MPS (no Modal, no GPU required). For each downloadable dataset
in ground_truth_comprehensive.json:

1. Streams parquet data for actions/states (no full download)
2. Downloads only the first video chunk file (~10-20MB) for frames
3. Extracts R3M/ImageNet embeddings on CPU/MPS
4. Runs the full ORBIT profiling pipeline
5. Extracts 64-dim features and caches to .npy

Non-downloadable datasets get estimated features from metadata.

Usage:
    python3 scripts/profile_training_data.py --device mps --max-episodes 10
    python3 scripts/profile_training_data.py --device cpu --hf-token YOUR_TOKEN
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _stream_dataset_to_orbit_dir(
    repo_id: str,
    output_dir: Path,
    max_episodes: int = 10,
    fps_sample: int = 5,
    hf_token: str | None = None,
) -> Path:
    """Download minimal data from a LeRobot HF repo and convert to ORBIT format.

    Strategy:
    - Download parquet files (small, ~1MB) for actions/states
    - Download ONLY the first video chunk (~10-20MB) for frames
    - Extract frames with OpenCV, subsample by fps_sample
    - Write ORBIT HDF5 and images
    - Clean up downloaded files after conversion
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download, list_repo_tree

    output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = output_dir / "session_lerobot.h5"
    if h5_path.exists():
        logger.info("Cached HDF5 found at %s", h5_path)
        return output_dir

    # --- Step 1: Download and read parquet data ---
    logger.info("Downloading parquet data for %s...", repo_id)
    try:
        # List files in the repo to find parquet + video paths
        files = list(list_repo_tree(repo_id, repo_type="dataset", recursive=True, token=hf_token))
        file_paths = []
        for f in files:
            p = f.rfilename if hasattr(f, "rfilename") else getattr(f, "path", "")
            if p:
                file_paths.append(p)
    except Exception as e:
        raise RuntimeError(f"Cannot list files in {repo_id}: {e}") from e

    parquet_files = [p for p in file_paths if p.startswith("data/") and p.endswith(".parquet")]
    video_files = [p for p in file_paths if p.startswith("videos/") and p.endswith(".mp4")]

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {repo_id}")

    # Download only first parquet chunk (enough for max_episodes)
    parquet_to_download = parquet_files[:1]
    local_parquets = []
    for pf in parquet_to_download:
        local_path = hf_hub_download(repo_id, pf, repo_type="dataset", token=hf_token)
        local_parquets.append(local_path)

    df = pd.concat([pd.read_parquet(p) for p in local_parquets], ignore_index=True)
    logger.info("Loaded %d rows from parquet", len(df))

    # Detect columns
    ep_col = None
    for col_name in ("episode_index", "episode_id", "episode"):
        if col_name in df.columns:
            ep_col = col_name
            break
    if ep_col is None:
        raise ValueError(f"Cannot find episode column. Columns: {list(df.columns)}")

    # Find action columns
    action_cols = _detect_columns(df, "action")
    state_cols = _detect_columns(df, "observation.state")
    if not action_cols:
        raise ValueError(f"Cannot find action columns in {list(df.columns)}")

    # Build episodes from parquet
    unique_episodes = sorted(df[ep_col].unique())[:max_episodes]

    episodes = []
    ep_frame_indices: dict[int, list[int]] = {}

    for ep_id in unique_episodes:
        ep_mask = df[ep_col] == ep_id
        ep_indices = df.index[ep_mask].tolist()
        sampled = ep_indices[::fps_sample]
        ep_df = df.loc[sampled]

        if state_cols:
            states = _extract_array(ep_df, state_cols)
        else:
            states = _extract_array(ep_df, action_cols)

        actions = _extract_array(ep_df, action_cols)

        if len(states) >= 2:
            episodes.append({
                "episode_id": int(ep_id),
                "states": states,
                "actions": actions,
            })
            ep_frame_indices[int(ep_id)] = sampled

    logger.info("Built %d episodes from parquet", len(episodes))

    # --- Step 2: Download first video chunk and extract frames ---
    img_dir = output_dir / "images"
    img_dir.mkdir(exist_ok=True)
    all_image_paths: dict[int, list[str]] = {}
    all_image_arrays: dict[int, list[np.ndarray]] = {}

    if video_files:
        # Pick first camera's first chunk
        first_video = video_files[0]
        logger.info("Downloading video chunk: %s", first_video)
        try:
            video_local = hf_hub_download(
                repo_id, first_video, repo_type="dataset", token=hf_token
            )

            # Collect all needed global frame indices
            needed_indices: set[int] = set()
            for indices in ep_frame_indices.values():
                needed_indices.update(indices)

            # Extract frames with OpenCV
            frame_map = _extract_frames_opencv(video_local, sorted(needed_indices))

            if not frame_map:
                logger.warning("OpenCV extraction failed, trying ffmpeg...")
                frame_map = _extract_frames_ffmpeg(video_local, sorted(needed_indices), img_dir)

            # Map frames to episodes and save as images
            for ep_id, indices in ep_frame_indices.items():
                ep_paths = []
                ep_arrays = []
                for idx in indices:
                    if idx in frame_map:
                        arr = frame_map[idx]
                        ep_arrays.append(arr)
                        from PIL import Image
                        img = Image.fromarray(arr)
                        img_path = img_dir / f"ep{ep_id}_f{idx}.png"
                        img.save(img_path)
                        ep_paths.append(str(img_path))
                all_image_paths[ep_id] = ep_paths
                all_image_arrays[ep_id] = ep_arrays

            logger.info("Extracted %d frames from video", len(frame_map))
        except Exception as e:
            logger.warning("Video extraction failed: %s", e)
    else:
        logger.warning("No video files in %s — episodes will have no images", repo_id)

    # --- Step 3: Write ORBIT HDF5 ---
    from orbit.profile.loaders import DatasetLoader
    DatasetLoader._write_hdf5(output_dir, episodes, all_image_paths, all_image_arrays)

    return output_dir


def _extract_frames_opencv(video_path: str, frame_indices: list[int]) -> dict[int, np.ndarray]:
    """Extract specific frames from video as RGB numpy arrays using OpenCV."""
    try:
        import cv2
    except ImportError:
        return {}

    needed = set(frame_indices)
    result: dict[int, np.ndarray] = {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    max_idx = max(needed) if needed else 0
    frame_idx = 0
    while frame_idx <= max_idx:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in needed:
            result[frame_idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_idx += 1
    cap.release()
    return result


def _extract_frames_ffmpeg(
    video_path: str, frame_indices: list[int], output_dir: Path
) -> dict[int, np.ndarray]:
    """Extract frames using ffmpeg as fallback."""
    import subprocess

    tmp_dir = Path(tempfile.mkdtemp(prefix="orbit_ffmpeg_"))
    try:
        pattern = str(tmp_dir / "frame_%06d.png")
        subprocess.run(
            ["ffmpeg", "-i", video_path, pattern, "-y", "-loglevel", "error"],
            check=True, capture_output=True, timeout=300,
        )
        needed = set(frame_indices)
        result = {}
        from PIL import Image
        for idx in needed:
            ffmpeg_path = tmp_dir / f"frame_{idx + 1:06d}.png"
            if ffmpeg_path.exists():
                img = Image.open(ffmpeg_path).convert("RGB")
                result[idx] = np.asarray(img, dtype=np.uint8).copy()
        return result
    except Exception as e:
        logger.warning("ffmpeg extraction failed: %s", e)
        return {}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _detect_columns(df, prefix: str) -> list[str]:
    """Find DataFrame columns matching a prefix pattern."""
    if prefix in df.columns:
        return [prefix]
    cols = [c for c in df.columns if c.startswith(prefix + ".")]
    if cols:
        return sorted(cols)
    underscore_prefix = prefix.replace(".", "_")
    cols = [c for c in df.columns if c.startswith(underscore_prefix)]
    return sorted(cols)


def _extract_array(df, cols: list[str]) -> np.ndarray:
    """Extract a 2D float32 array from DataFrame columns."""
    if len(cols) == 1 and df[cols[0]].dtype == object:
        return np.stack(df[cols[0]].values).astype(np.float32)
    return df[cols].values.astype(np.float32)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Profile datasets locally and cache features")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--max-episodes", type=int, default=10)
    parser.add_argument("--fps-sample", type=int, default=5)
    parser.add_argument("--cache-dir", default="benchmarks/cached_features")
    parser.add_argument("--gt-file", default="orbit/benchmarks/ground_truth_comprehensive.json")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token for gated datasets")
    parser.add_argument("--no-skip-existing", action="store_true", default=False)
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    skip_existing = not args.no_skip_existing

    # Set HF token if provided
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = args.hf_token
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # Load ground truth
    with open(args.gt_file) as f:
        gt = json.load(f)

    from orbit.profile.feature_extractor import DatasetFeatureExtractor
    from orbit.profile.profiler import DatasetProfiler
    from orbit.profile.report_card import ReportCardGenerator

    profiler = DatasetProfiler(embedding_model="r3m", device=args.device)
    extractor = DatasetFeatureExtractor()
    report_gen = ReportCardGenerator()

    manifest = {"profiled": [], "estimated": [], "failed": []}

    for entry in gt["benchmarks"]:
        entry_id = entry["id"]
        cache_path = os.path.join(args.cache_dir, f"{entry_id}.npy")
        meta_path = os.path.join(args.cache_dir, f"{entry_id}.json")

        # Skip if already cached
        if skip_existing and os.path.exists(cache_path):
            print(f"  CACHED {entry_id}")
            # Check if profiled or estimated from meta
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                if meta.get("type") == "profiled":
                    manifest["profiled"].append(entry_id)
                else:
                    manifest["estimated"].append(entry_id)
            else:
                manifest["profiled"].append(entry_id)
            continue

        # Non-downloadable or has no repo_id: use estimated features
        if not entry.get("downloadable", False) or not entry.get("repo_id"):
            if "estimated_features" in entry:
                feat = extractor.extract_from_metadata(entry["estimated_features"])
                _save_feature(
                    cache_path, meta_path, feat,
                    entry_id, "estimated",
                    entry.get("reported_success_rate", 0.0),
                )
                manifest["estimated"].append(entry_id)
                print(f"  EST {entry_id}")
            else:
                manifest["failed"].append({"id": entry_id, "reason": "no features available"})
                print(f"  SKIP {entry_id}: not downloadable, no estimated features")
            continue

        # Downloadable: stream + profile
        repo_id = entry["repo_id"]
        print(f"\n{'=' * 60}")
        print(f"Profiling {entry_id} ({repo_id})...")
        start = time.time()

        tmp_dir = None
        try:
            # Download minimal data and convert to ORBIT format
            tmp_dir = Path(tempfile.mkdtemp(prefix=f"orbit_{entry_id}_"))
            output_dir = _stream_dataset_to_orbit_dir(
                repo_id, tmp_dir,
                max_episodes=args.max_episodes,
                fps_sample=args.fps_sample,
                hf_token=hf_token,
            )

            # Run ORBIT profiling pipeline on the converted data
            task_desc = entry.get("task_description", entry.get("task", "manipulation"))
            profile = profiler.profile(str(output_dir), task_descriptions=[task_desc])
            report = report_gen.generate(profile)
            feat = extractor.extract(profile, report)

            elapsed = time.time() - start
            _save_feature(
                cache_path, meta_path, feat,
                entry_id, "profiled",
                entry.get("reported_success_rate", 0.0),
                extra={
                    "time_s": round(elapsed, 1),
                    "num_episodes": profile.num_episodes,
                    "num_frames": profile.num_frames,
                    "overall_grade": report.overall_grade,
                },
            )
            manifest["profiled"].append(entry_id)
            print(f"  OK {entry_id}: grade={report.overall_grade}, "
                  f"eps={profile.num_episodes}, frames={profile.num_frames}, "
                  f"time={elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - start
            manifest["failed"].append({
                "id": entry_id,
                "reason": str(e)[:200],
                "time_s": round(elapsed, 1),
            })
            print(f"  FAIL {entry_id}: {str(e)[:120]} ({elapsed:.1f}s)")

            # Fallback to estimated features if available
            if "estimated_features" in entry:
                feat = extractor.extract_from_metadata(entry["estimated_features"])
                _save_feature(
                    cache_path, meta_path, feat,
                    entry_id, "estimated_fallback",
                    entry.get("reported_success_rate", 0.0),
                )
                manifest["estimated"].append(entry_id)
                print(f"  -> Saved fallback estimated features for {entry_id}")
        finally:
            # Clean up temp dir to save disk space
            if tmp_dir and tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            gc.collect()

    # Save manifest
    manifest_path = os.path.join(args.cache_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'=' * 60}")
    print(
        f"DONE: {len(manifest['profiled'])} profiled, "
        f"{len(manifest['estimated'])} estimated, "
        f"{len(manifest['failed'])} failed"
    )
    print(f"Features cached in {args.cache_dir}/")
    print("Run 'python3 scripts/train_quality_model_local.py' next to train the model.")


def _save_feature(
    cache_path: str,
    meta_path: str,
    feat,
    entry_id: str,
    feat_type: str,
    success_rate: float,
    extra: dict | None = None,
):
    """Save feature vector and metadata to disk."""
    np.save(cache_path, feat)
    meta = {"id": entry_id, "type": feat_type, "success_rate": success_rate}
    if extra:
        meta.update(extra)
    with open(meta_path, "w") as f:
        json.dump(meta, f)


if __name__ == "__main__":
    main()
