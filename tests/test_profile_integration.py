"""Tests for orbit.profile integration (full pipeline)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import faiss
import h5py
import numpy as np
from PIL import Image

from orbit.profile.embedding import EmbeddingExtractor
from orbit.profile.profiler import DatasetProfiler
from orbit.profile.report import ProfileReporter
from orbit.profile.types import (
    CapabilityScore,
    CoverageMap,
    DatasetProfile,
    EmbeddingIndex,
    QualityMetrics,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_synthetic_dataset(tmp_path: Path, n_episodes: int = 20, frames_per_ep: int = 10):
    """Create a synthetic dataset with images and HDF5 state/action data."""
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    state_dim = 6
    action_dim = 6

    h5_path = tmp_path / "session_test.h5"
    with h5py.File(h5_path, "w") as f:
        f.attrs["session_id"] = "test"
        eps_grp = f.create_group("episodes")

        for ep in range(n_episodes):
            grp = eps_grp.create_group(str(ep))

            # State/action data
            states = rng.standard_normal((frames_per_ep, state_dim)).astype(np.float32)
            W = rng.standard_normal((state_dim, action_dim)).astype(np.float32)
            noise = rng.standard_normal((frames_per_ep, action_dim)) * 0.1
            actions = (states @ W + noise).astype(np.float32)
            grp.create_dataset("states", data=states)
            grp.create_dataset("actions", data=actions)

            # Images
            paths = []
            color = (200, 60, 50) if ep < n_episodes // 2 else (50, 60, 200)
            for fi in range(frames_per_ep):
                arr = np.full((64, 64, 3), color, dtype=np.float32)
                arr += rng.standard_normal((64, 64, 3)).astype(np.float32) * 15
                arr = np.clip(arr, 0, 255).astype(np.uint8)
                img = Image.fromarray(arr)
                img_path = img_dir / f"ep{ep}_f{fi}.png"
                img.save(img_path)
                paths.append(str(img_path))

            dt = h5py.string_dtype()
            grp.create_dataset("image_paths", data=paths, dtype=dt)

    return tmp_path


def _build_mock_profile(n_episodes: int = 10, n_frames: int = 100) -> DatasetProfile:
    """Build a profile directly from synthetic data (no disk I/O)."""
    dim = 64
    rng = np.random.default_rng(42)

    # Embedding index
    embs = rng.standard_normal((n_frames, dim)).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = (embs / np.maximum(norms, 1e-8)).astype(np.float32)
    idx = faiss.IndexFlatIP(dim)
    idx.add(embs)
    ep_ids = [i // (n_frames // n_episodes) for i in range(n_frames)]
    embedding_index = EmbeddingIndex(
        index=idx,
        episode_ids=ep_ids,
        frame_indices=list(range(n_frames)),
        dimension=dim,
        num_embeddings=n_frames,
    )

    # Coverage
    coverage = CoverageMap(
        dense_regions=[{"center": embs[0], "density": 50.0, "size": 20}],
        sparse_regions=[{"center": embs[-1], "density": 5.0, "size": 5}],
        overall_coverage_score=0.7,
        umap_projection=None,
    )

    # Quality
    quality = QualityMetrics(
        episode_scores={i: 0.5 + rng.random() * 0.5 for i in range(n_episodes)},
        aggregate_score=0.75,
        low_quality_episodes=[],
        mutual_information_estimate=1.2,
    )

    # Capabilities (some weak, some strong)
    capabilities = [
        CapabilityScore("pick up red cube", 0.8, 0.9, 8, 0.6, 0.2, None),
        CapabilityScore(
            "open drawer",
            0.3,
            0.5,
            3,
            0.3,
            0.1,
            "Low visual similarity to task 'open drawer'"
            " — dataset may not contain relevant scenes.",
        ),
        CapabilityScore(
            "pour water",
            0.2,
            0.4,
            2,
            0.2,
            0.05,
            "Relevant frames found but concentrated in few episodes"
            " — collect more diverse demonstrations.",
        ),
    ]

    profile = DatasetProfile(
        dataset_name="test_dataset",
        num_episodes=n_episodes,
        num_frames=n_frames,
        embedding_index=embedding_index,
        coverage=coverage,
        capabilities=capabilities,
        quality=quality,
        timestamp="2024-01-01T00:00:00",
    )

    reporter = ProfileReporter()
    profile.prescriptions = reporter.generate_prescriptions(profile)
    return profile


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestProfileIntegration:
    def test_full_pipeline(self, tmp_path):
        """Full profiler.profile() pipeline produces valid DatasetProfile."""
        data_dir = _create_synthetic_dataset(tmp_path)

        # Force fallback mode for embeddings (no SigLIP needed)
        profiler = DatasetProfiler(device="cpu")

        # Patch to force fallback embedding (random projection)
        with patch.object(EmbeddingExtractor, "_get_analyzer", return_value=None):
            profile = profiler.profile(
                str(data_dir),
                task_descriptions=["pick up red cube", "navigate terrain"],
            )

        assert isinstance(profile, DatasetProfile)
        assert profile.num_episodes > 0
        assert profile.num_frames > 0
        assert profile.embedding_index is not None
        assert profile.coverage is not None
        assert profile.quality is not None
        assert len(profile.capabilities) == 2
        print(f"Pipeline: {profile.num_episodes} episodes, {profile.num_frames} frames")
        for cap in profile.capabilities:
            print(f"  {cap.task_description}: {cap.score:.3f}")

    def test_report_markdown(self):
        """Markdown report contains all section headers and is >500 chars."""
        profile = _build_mock_profile()
        reporter = ProfileReporter()
        md = reporter.generate_report(profile, format="markdown")

        assert isinstance(md, str)
        assert len(md) > 500
        for header in [
            "Executive Summary",
            "Coverage Analysis",
            "Capability Breakdown",
            "Quality Assessment",
            "Prescriptions",
        ]:
            assert header in md, f"Missing section: {header}"
        print(f"Markdown report: {len(md)} chars")

    def test_report_json(self):
        """JSON report is valid and contains expected keys."""
        profile = _build_mock_profile()
        reporter = ProfileReporter()
        json_str = reporter.generate_report(profile, format="json")

        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert "dataset_name" in data
        assert "capabilities" in data
        assert "quality" in data
        assert "prescriptions" in data
        assert "coverage" in data
        print(f"JSON report: {len(json_str)} chars, {len(data['capabilities'])} capabilities")

    def test_prescriptions_ordered(self):
        """Prescriptions are sorted by priority (ascending numbers)."""
        profile = _build_mock_profile()
        reporter = ProfileReporter()
        prescriptions = reporter.generate_prescriptions(profile)

        assert len(prescriptions) > 0
        priorities = [p["priority"] for p in prescriptions]
        assert priorities == sorted(priorities)
        # Priority 1 should have lowest capability score
        assert prescriptions[0]["current_capability"] <= prescriptions[-1]["current_capability"]
        for p in prescriptions:
            print(f"  Priority {p['priority']}: {p['task']} (score={p['current_capability']})")

    def test_profile_empty_tasks(self, tmp_path):
        """Profile without task_descriptions still returns coverage and quality."""
        data_dir = _create_synthetic_dataset(tmp_path, n_episodes=5, frames_per_ep=5)
        profiler = DatasetProfiler(device="cpu")

        with patch.object(EmbeddingExtractor, "_get_analyzer", return_value=None):
            profile = profiler.profile(str(data_dir), task_descriptions=None)

        assert isinstance(profile, DatasetProfile)
        assert profile.coverage is not None
        assert profile.quality is not None
        assert len(profile.capabilities) == 0
        assert len(profile.prescriptions) == 0
        print(
            f"Empty tasks: coverage={profile.coverage.overall_coverage_score:.3f},"
            f" quality={profile.quality.aggregate_score:.3f}"
        )
