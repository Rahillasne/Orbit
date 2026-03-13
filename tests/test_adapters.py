"""Unit tests for dataset adapters (no network access required)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest

from orbit.adapters.base import BaseAdapter
from orbit.adapters.robomimic_adapter import RobomimicAdapter

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_robomimic_hdf5(path: Path, num_demos: int = 3, T: int = 10, with_images: bool = False):
    """Create a minimal robomimic-style HDF5 file."""
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["env"] = "robosuite"
        for i in range(num_demos):
            demo = data.create_group(f"demo_{i}")
            demo.create_dataset("actions", data=np.random.randn(T, 7).astype(np.float32))
            obs = demo.create_group("obs")
            obs.create_dataset("joint_pos", data=np.random.randn(T, 7).astype(np.float32))
            if with_images:
                obs.create_dataset(
                    "agentview_image",
                    data=np.random.randint(0, 255, (T, 64, 64, 3), dtype=np.uint8),
                )


def _make_mock_lerobot_dataset(
    num_episodes: int = 3,
    T_per_episode: int = 10,
    has_state: bool = True,
    has_images: bool = True,
    multi_camera: bool = False,
    has_language: bool = False,
):
    """Create a mock LeRobotDataset object."""
    mock_ds = MagicMock()
    total_frames = num_episodes * T_per_episode
    mock_ds.__len__ = MagicMock(return_value=total_frames)

    from_indices = [i * T_per_episode for i in range(num_episodes)]
    to_indices = [(i + 1) * T_per_episode for i in range(num_episodes)]
    mock_ds.episode_data_index = {
        "from": MagicMock(tolist=MagicMock(return_value=from_indices)),
        "to": MagicMock(tolist=MagicMock(return_value=to_indices)),
    }

    def mock_getitem(idx):
        sample = {"action": np.random.randn(7).astype(np.float32)}
        if has_state:
            sample["observation.state"] = np.random.randn(6).astype(np.float32)
        if has_images:
            # CHW float [0,1] format
            sample["observation.images.top"] = np.random.rand(3, 64, 64).astype(np.float32)
            if multi_camera:
                sample["observation.images.wrist"] = np.random.rand(3, 64, 64).astype(np.float32)
        if has_language:
            sample["language_instruction"] = "pick up the red cube"
        return sample

    mock_ds.__getitem__ = MagicMock(side_effect=mock_getitem)
    return mock_ds


# ------------------------------------------------------------------
# Tests: BaseAdapter
# ------------------------------------------------------------------


class TestBaseAdapter:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseAdapter()  # type: ignore[abstract]

    def test_default_metadata(self):
        class DummyAdapter(BaseAdapter):
            def iter_episodes(self):
                yield from []

            @property
            def num_episodes(self):
                return 0

        adapter = DummyAdapter()
        assert adapter.metadata == {}


# ------------------------------------------------------------------
# Tests: LeRobotAdapter
# ------------------------------------------------------------------


class TestLeRobotAdapter:
    def _make_adapter(self, mock_ds):
        """Patch LeRobotDataset and instantiate adapter."""
        import orbit.adapters.lerobot_adapter as mod

        def patched_load(self_adapter):
            self_adapter._ds = mock_ds
            ep_index = mock_ds.episode_data_index
            self_adapter._from_indices = ep_index["from"].tolist()
            self_adapter._to_indices = ep_index["to"].tolist()
            self_adapter._total_episodes = len(self_adapter._from_indices)
            self_adapter._dataset_meta = {
                "source": "lerobot_sdk",
                "repo_id": self_adapter._repo_id,
            }

        with patch.object(mod.LeRobotAdapter, "_load_dataset", patched_load):
            return mod.LeRobotAdapter("mock/dataset")

    def test_basic_iteration(self):
        mock_ds = _make_mock_lerobot_dataset(num_episodes=3, T_per_episode=10)
        adapter = self._make_adapter(mock_ds)

        assert adapter.num_episodes == 3
        episodes = list(adapter.iter_episodes())
        assert len(episodes) == 3

        for ep in episodes:
            assert "episode_id" in ep
            assert "states" in ep
            assert "actions" in ep
            assert "images" in ep
            assert "metadata" in ep
            assert ep["states"].shape[0] == ep["actions"].shape[0]
            assert ep["states"].dtype == np.float32
            assert ep["actions"].dtype == np.float32

    def test_max_episodes(self):
        mock_ds = _make_mock_lerobot_dataset(num_episodes=5)

        import orbit.adapters.lerobot_adapter as mod

        def patched_load(self_adapter):
            self_adapter._ds = mock_ds
            ep_index = mock_ds.episode_data_index
            self_adapter._from_indices = ep_index["from"].tolist()
            self_adapter._to_indices = ep_index["to"].tolist()
            self_adapter._total_episodes = len(self_adapter._from_indices)
            self_adapter._dataset_meta = {"source": "lerobot_sdk"}

        with patch.object(mod.LeRobotAdapter, "_load_dataset", patched_load):
            adapter = mod.LeRobotAdapter("mock/dataset", max_episodes=2)

        episodes = list(adapter.iter_episodes())
        assert len(episodes) == 2

    def test_no_state_uses_zeros(self):
        mock_ds = _make_mock_lerobot_dataset(has_state=False)
        adapter = self._make_adapter(mock_ds)
        episodes = list(adapter.iter_episodes())

        for ep in episodes:
            # States should be zeros when no observation.state
            assert ep["states"].shape[0] == ep["actions"].shape[0]
            np.testing.assert_array_equal(ep["states"], 0.0)

    def test_no_images(self):
        mock_ds = _make_mock_lerobot_dataset(has_images=False)
        adapter = self._make_adapter(mock_ds)
        episodes = list(adapter.iter_episodes())

        for ep in episodes:
            assert ep["images"] == {}

    def test_multi_camera(self):
        mock_ds = _make_mock_lerobot_dataset(multi_camera=True)
        adapter = self._make_adapter(mock_ds)
        episodes = list(adapter.iter_episodes())

        for ep in episodes:
            assert "observation.images.top" in ep["images"]
            assert "observation.images.wrist" in ep["images"]
            # Each camera should have same number of frames
            for frames in ep["images"].values():
                assert len(frames) == ep["actions"].shape[0]
                assert frames[0].ndim == 3  # HWC
                assert frames[0].dtype == np.uint8

    def test_language_instruction(self):
        mock_ds = _make_mock_lerobot_dataset(has_language=True)
        adapter = self._make_adapter(mock_ds)
        episodes = list(adapter.iter_episodes())

        for ep in episodes:
            assert ep["metadata"]["language_instruction"] == "pick up the red cube"

    def test_fps_sample(self):
        mock_ds = _make_mock_lerobot_dataset(num_episodes=1, T_per_episode=20)

        import orbit.adapters.lerobot_adapter as mod

        def patched_load(self_adapter):
            self_adapter._ds = mock_ds
            ep_index = mock_ds.episode_data_index
            self_adapter._from_indices = ep_index["from"].tolist()
            self_adapter._to_indices = ep_index["to"].tolist()
            self_adapter._total_episodes = len(self_adapter._from_indices)
            self_adapter._dataset_meta = {"source": "lerobot_sdk"}

        with patch.object(mod.LeRobotAdapter, "_load_dataset", patched_load):
            adapter = mod.LeRobotAdapter("mock/dataset", fps_sample=2)

        episodes = list(adapter.iter_episodes())
        assert len(episodes) == 1
        # With fps_sample=2 and 20 frames, we should get 10
        assert episodes[0]["actions"].shape[0] == 10

    def test_metadata_property(self):
        mock_ds = _make_mock_lerobot_dataset()
        adapter = self._make_adapter(mock_ds)
        meta = adapter.metadata
        assert meta["source"] == "lerobot_sdk"
        assert "repo_id" in meta


# ------------------------------------------------------------------
# Tests: RobomimicAdapter
# ------------------------------------------------------------------


class TestRobomimicAdapter:
    def test_basic_iteration(self, tmp_path: Path):
        h5_path = tmp_path / "demo.hdf5"
        _make_robomimic_hdf5(h5_path, num_demos=3, T=10)

        adapter = RobomimicAdapter(h5_path)
        assert adapter.num_episodes == 3

        episodes = list(adapter.iter_episodes())
        assert len(episodes) == 3

        for ep in episodes:
            assert ep["states"].shape == (10, 7)
            assert ep["actions"].shape == (10, 7)
            assert ep["states"].dtype == np.float32
            assert ep["metadata"]["demo_key"].startswith("demo_")

    def test_max_episodes(self, tmp_path: Path):
        h5_path = tmp_path / "demo.hdf5"
        _make_robomimic_hdf5(h5_path, num_demos=5)

        adapter = RobomimicAdapter(h5_path, max_episodes=2)
        episodes = list(adapter.iter_episodes())
        assert len(episodes) == 2

    def test_with_images(self, tmp_path: Path):
        h5_path = tmp_path / "demo.hdf5"
        _make_robomimic_hdf5(h5_path, num_demos=2, T=5, with_images=True)

        adapter = RobomimicAdapter(h5_path)
        episodes = list(adapter.iter_episodes())

        for ep in episodes:
            assert "agentview_image" in ep["images"]
            assert len(ep["images"]["agentview_image"]) == 5
            assert ep["images"]["agentview_image"][0].shape == (64, 64, 3)
            assert ep["images"]["agentview_image"][0].dtype == np.uint8

    def test_no_images(self, tmp_path: Path):
        h5_path = tmp_path / "demo.hdf5"
        _make_robomimic_hdf5(h5_path, with_images=False)

        adapter = RobomimicAdapter(h5_path)
        episodes = list(adapter.iter_episodes())

        for ep in episodes:
            assert ep["images"] == {}

    def test_action_only_dataset(self, tmp_path: Path):
        """Dataset with actions but no obs/joint_pos — states should be zeros."""
        h5_path = tmp_path / "action_only.hdf5"
        with h5py.File(h5_path, "w") as f:
            data = f.create_group("data")
            demo = data.create_group("demo_0")
            demo.create_dataset("actions", data=np.random.randn(10, 4).astype(np.float32))
            # No obs group at all

        adapter = RobomimicAdapter(h5_path)
        episodes = list(adapter.iter_episodes())
        assert len(episodes) == 1
        np.testing.assert_array_equal(episodes[0]["states"], 0.0)
        assert episodes[0]["states"].shape == (10, 4)

    def test_missing_hdf5_file(self):
        with pytest.raises(FileNotFoundError):
            RobomimicAdapter("/nonexistent/path.hdf5")

    def test_invalid_hdf5_structure(self, tmp_path: Path):
        h5_path = tmp_path / "bad.hdf5"
        with h5py.File(h5_path, "w") as f:
            f.create_group("wrong_key")

        with pytest.raises(ValueError, match="Expected 'data' group"):
            RobomimicAdapter(h5_path)

    def test_skips_short_demos(self, tmp_path: Path):
        """Demos with < 2 timesteps should be skipped."""
        h5_path = tmp_path / "short.hdf5"
        with h5py.File(h5_path, "w") as f:
            data = f.create_group("data")
            # 1-frame demo (should be skipped)
            demo0 = data.create_group("demo_0")
            demo0.create_dataset("actions", data=np.random.randn(1, 4).astype(np.float32))
            obs0 = demo0.create_group("obs")
            obs0.create_dataset("joint_pos", data=np.random.randn(1, 7).astype(np.float32))
            # 5-frame demo (should be kept)
            demo1 = data.create_group("demo_1")
            demo1.create_dataset("actions", data=np.random.randn(5, 4).astype(np.float32))
            obs1 = demo1.create_group("obs")
            obs1.create_dataset("joint_pos", data=np.random.randn(5, 7).astype(np.float32))

        adapter = RobomimicAdapter(h5_path)
        episodes = list(adapter.iter_episodes())
        assert len(episodes) == 1
        assert episodes[0]["actions"].shape[0] == 5

    def test_metadata(self, tmp_path: Path):
        h5_path = tmp_path / "demo.hdf5"
        _make_robomimic_hdf5(h5_path)

        adapter = RobomimicAdapter(h5_path)
        meta = adapter.metadata
        assert meta["source"] == "robomimic"
        assert meta["env"] == "robosuite"
        assert meta["num_demos"] == 3

    def test_multi_camera(self, tmp_path: Path):
        """Dataset with multiple camera keys."""
        h5_path = tmp_path / "multicam.hdf5"
        with h5py.File(h5_path, "w") as f:
            data = f.create_group("data")
            demo = data.create_group("demo_0")
            demo.create_dataset("actions", data=np.random.randn(5, 4).astype(np.float32))
            obs = demo.create_group("obs")
            obs.create_dataset("joint_pos", data=np.random.randn(5, 7).astype(np.float32))
            obs.create_dataset(
                "agentview_image",
                data=np.random.randint(0, 255, (5, 64, 64, 3), dtype=np.uint8),
            )
            obs.create_dataset(
                "eye_in_hand_image",
                data=np.random.randint(0, 255, (5, 64, 64, 3), dtype=np.uint8),
            )

        adapter = RobomimicAdapter(h5_path)
        episodes = list(adapter.iter_episodes())
        assert len(episodes) == 1
        assert "agentview_image" in episodes[0]["images"]
        assert "eye_in_hand_image" in episodes[0]["images"]

    def test_empty_dataset(self, tmp_path: Path):
        """Dataset with data group but no demos."""
        h5_path = tmp_path / "empty.hdf5"
        with h5py.File(h5_path, "w") as f:
            f.create_group("data")

        adapter = RobomimicAdapter(h5_path)
        assert adapter.num_episodes == 0
        episodes = list(adapter.iter_episodes())
        assert episodes == []


# ------------------------------------------------------------------
# Tests: Import paths
# ------------------------------------------------------------------


class TestImports:
    def test_import_from_adapters_package(self):
        from orbit.adapters import BaseAdapter, LeRobotAdapter, RobomimicAdapter

        assert BaseAdapter is not None
        assert LeRobotAdapter is not None
        assert RobomimicAdapter is not None
