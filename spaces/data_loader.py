"""Shared data-loading and caching utilities for the Orbit dashboard (Spaces version)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from uuid import UUID

import pandas as pd
import streamlit as st

from orbit_modules.schemas import Episode, LoggerConfig
from orbit_modules.storage import HDF5Storage


@st.cache_data(ttl=30)
def discover_sessions(data_dir: str) -> list[dict]:
    sessions: list[dict] = []
    data_path = Path(data_dir)
    if not data_path.exists():
        return sessions
    for h5_file in sorted(data_path.glob("session_*.h5")):
        try:
            session_id_str = h5_file.stem.replace("session_", "")
            sid = UUID(session_id_str)
            config = LoggerConfig(storage_dir=str(h5_file.parent))
            storage = HDF5Storage(config)
            pairs = storage.list_episodes(session_id=sid)
            storage.close()
            sessions.append({"session_id": str(sid), "file_path": str(h5_file), "episode_count": len(pairs)})
        except Exception:
            continue
    return sessions


@st.cache_data(ttl=60)
def load_session_episodes(h5_path: str) -> list[dict]:
    path = Path(h5_path)
    session_id = UUID(path.stem.replace("session_", ""))
    config = LoggerConfig(storage_dir=str(path.parent))
    storage = HDF5Storage(config)
    pairs = storage.list_episodes(session_id=session_id)
    episodes: list[dict] = []
    for sid, eid in pairs:
        ep = storage.load_episode(sid, eid)
        episodes.append(ep.model_dump(mode="json"))
    storage.close()
    return episodes


def deserialize_episodes(episode_dicts: list[dict]) -> list[Episode]:
    return [Episode.model_validate(d) for d in episode_dicts]


def episodes_to_summary_df(episodes: list[Episode]) -> pd.DataFrame:
    rows: list[dict] = []
    for ep in episodes:
        rows.append({
            "episode_id": str(ep.episode_id)[:8],
            "episode_id_full": str(ep.episode_id),
            "task_name": ep.task_name,
            "outcome": ep.outcome.value,
            "num_frames": ep.num_frames,
            "duration": ep.duration,
            "total_reward": round(ep.total_reward, 4),
            "avg_action_mag": round(ep.avg_action_magnitude, 4),
            "start_time": ep.start_time,
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def run_detector_pipeline(episodes: list[Episode]) -> list[dict]:
    from orbit_modules.heuristic import (
        DetectorPipeline, GripperDropDetector, OutOfBoundsDetector,
        RewardThresholdDetector, StallDetector, TimeoutDetector,
    )
    pipeline = DetectorPipeline(detectors=[
        GripperDropDetector(), StallDetector(), OutOfBoundsDetector(),
        TimeoutDetector(), RewardThresholdDetector(),
    ])
    results = pipeline.run_batch(episodes)
    serialized = []
    for r in results:
        serialized.append({
            "episode_id": str(r.episode_id),
            "is_failure": r.is_failure,
            "failure_probability": r.failure_probability,
            "detections": [{"detector_name": d.detector_name, "confidence": d.confidence, "frame_idx": d.frame_idx, "description": d.description} for d in r.detections],
            "detector_summaries": r.detector_summaries,
        })
    return serialized


def get_failure_type_counts(detection_results: list[dict]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for r in detection_results:
        if r["is_failure"]:
            for d in r["detections"]:
                counter[d["detector_name"]] += 1
    return dict(counter.most_common())


def run_prescriber(detection_results: list[dict], episodes: list[Episode]) -> dict:
    from orbit_modules.legacy import DetectionResult
    try:
        from orbit_modules.prescriber import Prescriber
    except ImportError:
        return {"prescriptions": [], "summary": "Prescriber unavailable.", "num_failures_analyzed": 0}

    legacy_results: list[DetectionResult] = []
    for r in detection_results:
        reasons = [d["description"] for d in r["detections"]]
        legacy_results.append(DetectionResult(
            episode_id=hash(r["episode_id"]) % (2**31),
            is_failure=r["is_failure"],
            failure_reasons=reasons,
            confidence=r["failure_probability"],
        ))
    prescriber = Prescriber()
    report = prescriber.prescribe(legacy_results)
    return {
        "prescriptions": [
            {"type": p.prescription_type.value, "title": p.title, "description": p.description,
             "priority": p.priority, "confidence": round(p.confidence, 3),
             "evidence": p.evidence, "suggested_params": p.suggested_params}
            for p in report.prescriptions
        ],
        "summary": report.summary,
        "num_failures_analyzed": report.num_failures_analyzed,
    }
