# Orbit

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/Rahillasne/Orbit/actions/workflows/ci.yml/badge.svg)](https://github.com/Rahillasne/Orbit/actions/workflows/ci.yml)

A robotics/ML debugging toolkit for logging robot learning episodes, detecting failures, analyzing failure modes with vision-language embeddings, and prescribing corrective actions.

## Installation

```bash
git clone https://github.com/Rahillasne/Orbit.git
cd orbit
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Quickstart

```python
import numpy as np
from PIL import Image

from orbit.logger.episode_logger import EpisodeLogger
from orbit.logger.schemas import LoggerConfig, Outcome

# 1. Log episodes
config = LoggerConfig(
    storage_dir="./my_data",
    task_name="pick_and_place",
    robot_dof=6,
)

with EpisodeLogger(config) as logger:
    logger.start_episode()
    for step in range(50):
        joint_pos = (np.random.randn(6) * 0.1).tolist()
        action = (np.random.randn(6) * 0.01).tolist()
        gripper = float(np.clip(np.random.random(), 0, 1))
        image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        logger.log_frame(
            joint_positions=joint_pos,
            gripper_state=gripper,
            action=action,
            reward=0.1,
            images={"front": image},
        )
    episode = logger.end_episode(outcome=Outcome.SUCCESS)

# 2. Detect failures
from orbit.detector.heuristic import DetectorPipeline

pipeline = DetectorPipeline()  # includes all 5 detectors by default
result = pipeline.run(episode)
print(f"Failure: {result.is_failure}, Detections: {len(result.detections)}")

# 3. Generate prescriptions
from orbit.prescriber import Prescriber

prescriber = Prescriber()
report = prescriber.prescribe(detection_results=[result.to_legacy_result()])
for p in report.prescriptions:
    print(f"[{p.priority}] {p.title}: {p.description}")
```

> See [examples/quickstart.py](examples/quickstart.py) for the full working example.

## Modules

| Module | Description |
|--------|-------------|
| `orbit.logger` | Episode logging with HDF5/Parquet storage backends |
| `orbit.detector` | Heuristic-based failure detection (reward thresholds, action variance, consecutive failures) |
| `orbit.analyzer` | Embedding gap analysis using OpenCLIP, FAISS, and UMAP |
| `orbit.prescriber` | Corrective prescription generation from failure patterns |
| `orbit.vlm` | Vision-language model failure description via CLIP zero-shot classification |
| `orbit.dashboard` | Streamlit + Plotly interactive dashboard for episode visualization |

## Dashboard

![Session Overview](docs/dashboard-overview.png)

![Prescriptions](docs/dashboard-prescriptions.png)

Launch the interactive dashboard:

```bash
orbit-dashboard
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Install dev dependencies (`pip install -e ".[dev]"`)
4. Run tests (`pytest tests/`)
5. Run linting (`ruff check orbit/ tests/`)
6. Submit a pull request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
