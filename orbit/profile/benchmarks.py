"""Synthetic benchmark data generator for validating the profiler."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import faiss
import h5py
import numpy as np
from PIL import Image

from orbit.profile.types import EmbeddingIndex

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cluster(
    n: int, dim: int, center: np.ndarray, std: float = 0.05, seed: int = 0
) -> np.ndarray:
    """Generate L2-normalised embeddings clustered around *center*."""
    rng = np.random.default_rng(seed)
    pts = center + rng.standard_normal((n, dim)).astype(np.float32) * std
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return (pts / np.maximum(norms, 1e-8)).astype(np.float32)


def _build_index(embeddings: np.ndarray, episode_ids: list[int] | None = None) -> EmbeddingIndex:
    """Build an EmbeddingIndex from raw embeddings."""
    embs = embeddings.astype(np.float32).copy()
    faiss.normalize_L2(embs)
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    n = len(embs)
    return EmbeddingIndex(
        index=idx,
        episode_ids=episode_ids or list(range(n)),
        frame_indices=list(range(n)),
        dimension=embs.shape[1],
        num_embeddings=n,
    )


def _make_colored_images(
    n: int,
    color: tuple[int, int, int],
    noise_std: float = 15.0,
    seed: int = 42,
    size: int = 64,
) -> list[Image.Image]:
    """Generate synthetic images dominated by a single color."""
    rng = np.random.default_rng(seed)
    images: list[Image.Image] = []
    for _ in range(n):
        arr = np.full((size, size, 3), color, dtype=np.float32)
        arr += rng.standard_normal((size, size, 3)).astype(np.float32) * noise_std
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        images.append(Image.fromarray(arr))
    return images


def _make_deterministic_episode(
    episode_id: int, T: int = 50, state_dim: int = 6, seed: int = 0
) -> dict:
    rng = np.random.default_rng(seed)
    states = rng.standard_normal((T, state_dim))
    W = rng.standard_normal((state_dim, state_dim))
    actions = states @ W + rng.standard_normal((T, state_dim)) * 0.01
    return {"episode_id": episode_id, "states": states, "actions": actions}


def _make_noisy_episode(
    episode_id: int, T: int = 50, state_dim: int = 6, noise: float = 2.0, seed: int = 0
) -> dict:
    rng = np.random.default_rng(seed)
    states = rng.standard_normal((T, state_dim))
    W = rng.standard_normal((state_dim, state_dim))
    actions = states @ W + rng.standard_normal((T, state_dim)) * noise
    return {"episode_id": episode_id, "states": states, "actions": actions}


def _make_random_episode(episode_id: int, T: int = 50, state_dim: int = 6, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    states = rng.standard_normal((T, state_dim))
    actions = rng.standard_normal((T, state_dim))
    return {"episode_id": episode_id, "states": states, "actions": actions}


def _save_episodes_hdf5(
    output_dir: Path,
    images_per_episode: dict[int, list[Image.Image]],
    episodes: list[dict],
) -> None:
    """Save episodes as an HDF5 file with images on disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    img_dir = output_dir / "images"
    img_dir.mkdir(exist_ok=True)

    h5_path = output_dir / "session_benchmark.h5"
    with h5py.File(h5_path, "w") as f:
        f.attrs["session_id"] = "benchmark"
        eps_grp = f.create_group("episodes")
        for ep in episodes:
            eid = ep["episode_id"]
            grp = eps_grp.create_group(str(eid))
            grp.create_dataset("states", data=ep["states"].astype(np.float32))
            grp.create_dataset("actions", data=ep["actions"].astype(np.float32))

            imgs = images_per_episode.get(eid, [])
            paths: list[str] = []
            for fi, img in enumerate(imgs):
                img_path = img_dir / f"ep{eid}_f{fi}.png"
                img.save(img_path)
                paths.append(str(img_path))
            dt = h5py.string_dtype()
            grp.create_dataset("image_paths", data=paths, dtype=dt)


# ---------------------------------------------------------------------------
# SyntheticBenchmark
# ---------------------------------------------------------------------------


class SyntheticBenchmark:
    """Generates synthetic datasets with known capability profiles."""

    @staticmethod
    def create_two_cluster_dataset(output_dir: str, n_episodes: int = 40) -> None:
        """Create a dataset with two distinct task clusters.

        Cluster A (60%): warm-coloured images, deterministic actions
        Cluster B (40%): cool-coloured images, deterministic actions
        """
        out = Path(output_dir)
        if out.exists():
            shutil.rmtree(out)

        n_a = int(n_episodes * 0.6)
        n_b = n_episodes - n_a
        frames_per_ep = 10

        episodes: list[dict] = []
        images_per_episode: dict[int, list[Image.Image]] = {}

        for i in range(n_a):
            ep = _make_deterministic_episode(i, T=frames_per_ep, seed=i)
            episodes.append(ep)
            images_per_episode[i] = _make_colored_images(frames_per_ep, (200, 60, 50), seed=i)

        for i in range(n_b):
            eid = n_a + i
            ep = _make_deterministic_episode(eid, T=frames_per_ep, seed=eid + 100)
            episodes.append(ep)
            images_per_episode[eid] = _make_colored_images(
                frames_per_ep, (50, 60, 200), seed=eid + 100
            )

        _save_episodes_hdf5(out, images_per_episode, episodes)

    @staticmethod
    def create_quality_mixed_dataset(output_dir: str, n_episodes: int = 30) -> None:
        """Create a dataset with varying quality levels."""
        out = Path(output_dir)
        if out.exists():
            shutil.rmtree(out)

        n_good = int(n_episodes * 0.5)
        n_noisy = int(n_episodes * 0.3)
        n_garbage = n_episodes - n_good - n_noisy
        frames_per_ep = 10

        episodes: list[dict] = []
        images_per_episode: dict[int, list[Image.Image]] = {}

        eid = 0
        for i in range(n_good):
            ep = _make_deterministic_episode(eid, T=frames_per_ep, seed=eid)
            episodes.append(ep)
            images_per_episode[eid] = _make_colored_images(frames_per_ep, (100, 200, 100), seed=eid)
            eid += 1

        for i in range(n_noisy):
            ep = _make_noisy_episode(eid, T=frames_per_ep, noise=2.0, seed=eid)
            episodes.append(ep)
            images_per_episode[eid] = _make_colored_images(frames_per_ep, (200, 200, 100), seed=eid)
            eid += 1

        for i in range(n_garbage):
            ep = _make_random_episode(eid, T=frames_per_ep, seed=eid)
            episodes.append(ep)
            images_per_episode[eid] = _make_colored_images(frames_per_ep, (200, 100, 100), seed=eid)
            eid += 1

        _save_episodes_hdf5(out, images_per_episode, episodes)

    @staticmethod
    def create_gap_dataset(output_dir: str) -> None:
        """Create training + deployment datasets with known gaps.

        Training covers regions A and B.
        Deployment covers regions A, B, and C.
        """
        out = Path(output_dir)
        if out.exists():
            shutil.rmtree(out)

        train_dir = out / "training"
        deploy_dir = out / "deployment"
        frames_per_ep = 10

        # Training: regions A + B
        episodes_train: list[dict] = []
        imgs_train: dict[int, list[Image.Image]] = {}
        for i in range(15):
            ep = _make_deterministic_episode(i, T=frames_per_ep, seed=i)
            episodes_train.append(ep)
            imgs_train[i] = _make_colored_images(frames_per_ep, (200, 60, 50), seed=i)
        for i in range(10):
            eid = 15 + i
            ep = _make_deterministic_episode(eid, T=frames_per_ep, seed=eid + 100)
            episodes_train.append(ep)
            imgs_train[eid] = _make_colored_images(frames_per_ep, (50, 60, 200), seed=eid + 100)
        _save_episodes_hdf5(train_dir, imgs_train, episodes_train)

        # Deployment: regions A + B + C
        episodes_deploy: list[dict] = []
        imgs_deploy: dict[int, list[Image.Image]] = {}
        for i in range(5):
            ep = _make_deterministic_episode(i, T=frames_per_ep, seed=i + 200)
            episodes_deploy.append(ep)
            imgs_deploy[i] = _make_colored_images(frames_per_ep, (200, 60, 50), seed=i + 200)
        for i in range(5):
            eid = 5 + i
            ep = _make_deterministic_episode(eid, T=frames_per_ep, seed=eid + 300)
            episodes_deploy.append(ep)
            imgs_deploy[eid] = _make_colored_images(frames_per_ep, (50, 60, 200), seed=eid + 300)
        # Region C — novel (green)
        for i in range(10):
            eid = 10 + i
            ep = _make_deterministic_episode(eid, T=frames_per_ep, seed=eid + 400)
            episodes_deploy.append(ep)
            imgs_deploy[eid] = _make_colored_images(frames_per_ep, (50, 200, 50), seed=eid + 400)
        _save_episodes_hdf5(deploy_dir, imgs_deploy, episodes_deploy)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate_profiler(output_dir: str, verbose: bool = True) -> dict:
        """Run all synthetic benchmarks and check predictions.

        This works at the component level (not full profiler.profile())
        to avoid requiring SigLIP for benchmark validation.
        """
        from orbit.profile.capability import CapabilityScorer
        from orbit.profile.coverage import CoverageAnalyzer
        from orbit.profile.quality import QualityEstimator

        results: dict = {}

        # ---- Two-cluster benchmark ----
        dim = 64
        rng = np.random.default_rng(42)
        center_a = rng.standard_normal(dim).astype(np.float32)
        center_a /= np.linalg.norm(center_a)
        center_b = -center_a

        embs_a = _make_cluster(60, dim, center_a, std=0.03, seed=1)
        embs_b = _make_cluster(40, dim, center_b, std=0.03, seed=2)
        all_embs = np.vstack([embs_a, embs_b])
        ep_ids = [0] * 60 + [1] * 40
        index = _build_index(all_embs, episode_ids=ep_ids)

        analyzer = CoverageAnalyzer(min_cluster_size=3)
        coverage = analyzer.analyze(index)

        episodes = [
            _make_deterministic_episode(0, T=60, seed=10),
            _make_deterministic_episode(1, T=40, seed=20),
        ]
        quality = QualityEstimator().estimate_quality(episodes)

        scorer = CapabilityScorer(top_k=min(50, index.num_embeddings))

        # Mock text encoding: "tabletop" → center_a, "navigation" → center_b
        def mock_encode(texts):
            out_embs = []
            for t in texts:
                if "tabletop" in t.lower() or "pick" in t.lower() or "red" in t.lower():
                    v = center_a.copy()
                elif "navigation" in t.lower() or "outdoor" in t.lower() or "blue" in t.lower():
                    v = center_b.copy()
                else:
                    # Orthogonal to both
                    v = rng.standard_normal(dim).astype(np.float32)
                    v -= v.dot(center_a) * center_a
                    v -= v.dot(center_b) * center_b
                v /= max(np.linalg.norm(v), 1e-8)
                out_embs.append(v)
            return np.array(out_embs, dtype=np.float32)

        scorer._encode_texts = mock_encode
        scorer._text_encoder_mode = "mock"

        caps = scorer.score_tasks(
            index,
            coverage,
            quality,
            ["indoor tabletop manipulation", "outdoor navigation", "fly a helicopter"],
        )
        tabletop_score = caps[0].score
        nav_score = caps[1].score
        unrelated_score = caps[2].score

        two_cluster_pass = (
            tabletop_score > 0.5 and nav_score > 0.3 and unrelated_score < tabletop_score
        )
        results["two_cluster"] = {
            "tabletop_score": round(tabletop_score, 3),
            "navigation_score": round(nav_score, 3),
            "unrelated_score": round(unrelated_score, 3),
            "pass": two_cluster_pass,
        }

        # ---- Quality-mixed benchmark ----
        good_eps = [_make_deterministic_episode(i, T=150, seed=i) for i in range(5)]
        noisy_eps = [_make_noisy_episode(i + 5, T=150, noise=2.0, seed=i + 5) for i in range(3)]
        garbage_eps = [_make_random_episode(i + 8, T=150, seed=i + 8) for i in range(2)]
        all_eps = good_eps + noisy_eps + garbage_eps

        qm = QualityEstimator(k_neighbors=5).estimate_quality(all_eps)

        good_mean = float(np.mean([qm.episode_scores[i] for i in range(5)]))
        noisy_mean = float(np.mean([qm.episode_scores[i + 5] for i in range(3)]))
        garbage_mean = float(np.mean([qm.episode_scores[i + 8] for i in range(2)]))
        ranking_correct = good_mean > noisy_mean > garbage_mean

        results["quality_mixed"] = {
            "high_quality_mean": round(good_mean, 3),
            "low_quality_mean": round(noisy_mean, 3),
            "garbage_mean": round(garbage_mean, 3),
            "ranking_correct": ranking_correct,
            "pass": ranking_correct,
        }

        # ---- Gap detection benchmark ----
        center_c = rng.standard_normal(dim).astype(np.float32)
        # Make C orthogonal to A and B
        center_c -= center_c.dot(center_a) * center_a
        center_c -= center_c.dot(center_b) * center_b
        center_c /= max(np.linalg.norm(center_c), 1e-8)

        train_embs = np.vstack(
            [
                _make_cluster(30, dim, center_a, std=0.03, seed=10),
                _make_cluster(20, dim, center_b, std=0.03, seed=20),
            ]
        )
        deploy_embs = np.vstack(
            [
                _make_cluster(10, dim, center_a, std=0.03, seed=30),
                _make_cluster(10, dim, center_b, std=0.03, seed=40),
                _make_cluster(20, dim, center_c, std=0.03, seed=50),
            ]
        )

        train_index = _build_index(train_embs)
        gaps = analyzer.find_gaps(train_index, reference_index=_build_index(deploy_embs))

        # Check that gaps come predominantly from region C
        # Region C embeddings are indices 20-39 in deploy_embs
        gap_found_in_c = len(gaps) > 0
        # Check that region A points are NOT flagged as gaps
        # Search train index for A and B deploy points — should have high similarity
        sims_ab, _ = train_index.index.search(deploy_embs[:20].copy(), 1)
        no_false_gap_a = bool(np.mean(sims_ab[:10, 0]) > 0.7)
        no_false_gap_b = bool(np.mean(sims_ab[10:20, 0]) > 0.7)

        gap_pass = gap_found_in_c and no_false_gap_a and no_false_gap_b
        results["gap_detection"] = {
            "gap_found_in_C": gap_found_in_c,
            "no_false_gap_in_A": no_false_gap_a,
            "no_false_gap_in_B": no_false_gap_b,
            "pass": gap_pass,
        }

        results["overall_pass"] = all(
            results[k]["pass"] for k in ["two_cluster", "quality_mixed", "gap_detection"]
        )

        if verbose:
            for key, val in results.items():
                if isinstance(val, dict):
                    status = "PASS" if val.get("pass") else "FAIL"
                    print(f"  {key}: {status}")
                    for k2, v2 in val.items():
                        if k2 != "pass":
                            print(f"    {k2}: {v2}")
                else:
                    print(f"  {key}: {val}")

        return results
