"""Global configuration management for Orbit."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class OrbitConfig(BaseSettings):
    """Global Orbit configuration.

    Values can be overridden via constructor arguments or environment
    variables with the ``ORBIT_`` prefix (e.g. ``ORBIT_DATA_DIR``).
    """

    model_config = {"env_prefix": "ORBIT_"}

    # Storage
    data_dir: Path = Path("./orbit_data")
    storage_format: str = "parquet"

    # CLIP / VLM
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    device: str = "cpu"

    # Detector defaults
    reward_threshold: float = -1.0
    action_variance_threshold: float = 0.01

    # Dashboard
    dashboard_port: int = 8501
    dashboard_title: str = "Orbit Dashboard"

    # Logging
    log_level: str = "INFO"


def get_config(**overrides) -> OrbitConfig:
    """Factory function to get the global config with optional overrides."""
    return OrbitConfig(**overrides)
