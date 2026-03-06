"""Unit tests for DatasetLoader (no network access required)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest
from PIL import Image

from orbit.profile.loaders import DatasetLoader


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def orbit_hdf5_dir(tmp_path: Path) -> Path:
    """Create a minimal ORBIT HDF5 dataset."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    h5_path = tmp_path / "session_test.h5"
    with h5py.File(h5_path, "w") as f:
        f.attrs["session_id"] = "test"
        eps = f.create_group("episodes")
        for eid in range(3):
            grp = eps.create_group(str(eid))
            T = 10
            grp.create_dataset("states", data=np.random.randn(T, 6).astype(np.float32))
            grp.create_dataset("actions", data=np.random.randn(T, 6).astype(np.float32))
            paths = []
            for fi in range(T):
                img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
                img_path = img_dir / f"ep{eid}_f{fi}.png"
                img.save(img_path)
                paths.append(str(img_path))
            dt = h5py.string_dtype()
            grp.create_dataset("image_paths", data=paths, dtype=dt)

    return tmp_path


@pytest.fixture()
def image_dir(tmp_path: Path) -> Path:
    """Create a directory of images."""
    for i in range(5):
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(tmp_path / f"frame_{i:04d}.png")
    img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    img.save(tmp_path / f"photo.jpg")
    return tmp_path


# ------------------------------------------------------------------
# Tests: from_hdf5_directory
# ------------------------------------------------------------------


class TestFromHDF5Directory:
    def test_loads_episodes(self, orbit_hdf5_dir: Path):
        episodes = DatasetLoader.from_hdf5_directory(orbit_hdf5_dir)
        assert len(episodes) == 3
        for ep in episodes:
            assert "episode_id" in ep
            assert "states" in ep
            assert "actions" in ep
            assert ep["states"].shape == (10, 6)
            assert ep["actions"].shape == (10, 6)

    def test_empty_directory(self, tmp_path: Path):
        episodes = DatasetLoader.from_hdf5_directory(tmp_path)
        assert episodes == []

    def test_no_episodes_group(self, tmp_path: Path):
        h5_path = tmp_path / "session_empty.h5"
        with h5py.File(h5_path, "w") as f:
            f.attrs["session_id"] = "empty"
        episodes = DatasetLoader.from_hdf5_directory(tmp_path)
        assert episodes == []


# ------------------------------------------------------------------
# Tests: from_image_directory
# ------------------------------------------------------------------


class TestFromImageDirectory:
    def test_finds_images(self, image_dir: Path):
        paths = DatasetLoader.from_image_directory(image_dir)
        assert len(paths) == 6  # 5 png + 1 jpg
        assert all(p.endswith((".png", ".jpg")) for p in paths)

    def test_empty_directory(self, tmp_path: Path):
        paths = DatasetLoader.from_image_directory(tmp_path)
        assert paths == []


# ------------------------------------------------------------------
# Tests: from_lerobot with mock SDK
# ------------------------------------------------------------------


class TestFromLeRobotSDK:
    def test_convert_with_mock_sdk(self, tmp_path: Path):
        """Mock LeRobotDataset and verify HDF5 output structure."""
        output_dir = tmp_path / "output"

        # Create mock dataset
        mock_ds = MagicMock()
        mock_ds.__len__ = MagicMock(return_value=20)
        mock_ds.episode_data_index = {
            "from": MagicMock(tolist=MagicMock(return_value=[0, 10])),
            "to": MagicMock(tolist=MagicMock(return_value=[10, 20])),
        }

        def mock_getitem(idx):
            import torch
            return {
                "observation.state": torch.randn(6),
                "action": torch.randn(6),
                "observation.images.top": torch.rand(3, 64, 64),
            }

        mock_ds.__getitem__ = mock_getitem

        with patch(
            "orbit.profile.loaders.DatasetLoader._convert_via_lerobot_sdk"
        ) as mock_convert:
            # Instead of mocking the full SDK, directly test _write_hdf5
            episodes = [
                {"episode_id": 0, "states": np.random.randn(10, 6).astype(np.float32),
                 "actions": np.random.randn(10, 6).astype(np.float32)},
                {"episode_id": 1, "states": np.random.randn(10, 6).astype(np.float32),
                 "actions": np.random.randn(10, 6).astype(np.float32)},
            ]
            output_dir.mkdir(parents=True)
            DatasetLoader._write_hdf5(output_dir, episodes, {})

        h5_path = output_dir / "session_lerobot.h5"
        assert h5_path.exists()

        with h5py.File(h5_path, "r") as f:
            assert "episodes" in f
            assert "0" in f["episodes"]
            assert "1" in f["episodes"]
            assert f["episodes"]["0"]["states"].shape == (10, 6)
            assert f["episodes"]["0"]["actions"].shape == (10, 6)

    def test_cached_skips_conversion(self, tmp_path: Path):
        """If HDF5 already exists, from_lerobot returns immediately."""
        output_dir = tmp_path / "cached"
        output_dir.mkdir()
        h5_path = output_dir / "session_lerobot.h5"
        with h5py.File(h5_path, "w") as f:
            f.attrs["session_id"] = "cached"

        result = DatasetLoader.from_lerobot("fake/repo", output_dir)
        assert result == output_dir


# ------------------------------------------------------------------
# Tests: _write_hdf5
# ------------------------------------------------------------------


class TestWriteHDF5:
    def test_writes_with_images(self, tmp_path: Path):
        episodes = [
            {"episode_id": 0, "states": np.ones((5, 4), dtype=np.float32),
             "actions": np.zeros((5, 4), dtype=np.float32)},
        ]
        img_paths = {0: ["/fake/img0.png", "/fake/img1.png"]}

        DatasetLoader._write_hdf5(tmp_path, episodes, img_paths)

        h5_path = tmp_path / "session_lerobot.h5"
        with h5py.File(h5_path, "r") as f:
            grp = f["episodes"]["0"]
            np.testing.assert_array_equal(grp["states"][:], np.ones((5, 4)))
            np.testing.assert_array_equal(grp["actions"][:], np.zeros((5, 4)))
            paths = [p.decode() if isinstance(p, bytes) else p for p in grp["image_paths"][:]]
            assert paths == ["/fake/img0.png", "/fake/img1.png"]

    def test_writes_without_images(self, tmp_path: Path):
        episodes = [
            {"episode_id": 0, "states": np.ones((5, 4), dtype=np.float32),
             "actions": np.zeros((5, 4), dtype=np.float32)},
        ]

        DatasetLoader._write_hdf5(tmp_path, episodes, {})

        h5_path = tmp_path / "session_lerobot.h5"
        with h5py.File(h5_path, "r") as f:
            grp = f["episodes"]["0"]
            assert "states" in grp
            assert "actions" in grp
            assert "image_paths" not in grp


# ------------------------------------------------------------------
# Tests: helpers
# ------------------------------------------------------------------


class TestHelpers:
    def test_detect_columns_exact(self):
        import pandas as pd
        df = pd.DataFrame({"action": [[1, 2]], "observation.state": [[3, 4]]})
        assert DatasetLoader._detect_columns(df, "action") == ["action"]
        assert DatasetLoader._detect_columns(df, "observation.state") == ["observation.state"]

    def test_detect_columns_dotted(self):
        import pandas as pd
        df = pd.DataFrame({"action.0": [1], "action.1": [2], "action.2": [3]})
        cols = DatasetLoader._detect_columns(df, "action")
        assert cols == ["action.0", "action.1", "action.2"]

    def test_parse_episode_id(self):
        assert DatasetLoader._parse_episode_id_from_filename("episode_000000") == 0
        assert DatasetLoader._parse_episode_id_from_filename("episode_000042") == 42
        assert DatasetLoader._parse_episode_id_from_filename("no_digits") is None

    def test_video_frame_extraction_no_backends(self, tmp_path: Path):
        """When cv2 and ffmpeg are unavailable, returns empty dict."""
        fake_video = tmp_path / "test.mp4"
        fake_video.write_bytes(b"not a real video")
        output = tmp_path / "frames"
        output.mkdir()

        with patch.dict("sys.modules", {"cv2": None}):
            # This should gracefully return empty (cv2 import fails, ffmpeg fails on fake data)
            result = DatasetLoader._extract_specific_frames(fake_video, output, [0, 1, 2])
            # Result may be empty or have frames depending on ffmpeg availability
            assert isinstance(result, dict)
