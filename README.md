# ORBIT — Open Robot Iteration Toolkit

Deployment diagnostics for learned robot policies. Find why your robot fails, and what data to collect next.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)
![Tests](https://img.shields.io/badge/tests-128_passed-brightgreen.svg)
[![Demo](https://img.shields.io/badge/🤗-Try_Demo-yellow.svg)](https://huggingface.co/spaces/Drahils/orbit-demo)

![ORBIT Dashboard](docs/dashboard-overview.png)
*ORBIT dashboard showing deployment failure clusters and data collection prescriptions*

## What It Does

ORBIT watches your robot fail in deployment, diagnoses *why* by comparing deployment data against your training distribution, and prescribes exactly what demonstrations to collect next.

- **Logs** every deployment episode (joints, images, actions, outcomes) to HDF5
- **Detects** failures automatically (gripper drops, stalls, timeouts, out-of-bounds, reward)
- **Analyzes** distribution gaps between training and deployment using vision embeddings (SigLIP)
- **Prescribes** ranked, specific data collection tasks to close the gaps

## Quick Start

```bash
pip install -e .
python scripts/generate_synthetic_deployment.py --output-dir ./demo_data
orbit dashboard --data-dir ./demo_data
```

Or try the [live demo on HuggingFace Spaces](https://huggingface.co/spaces/Drahils/orbit-demo) — no install required.

## How It Works

```
Deploy Robot → ORBIT Logger → Failure Detector → Embedding Analyzer → Prescriber
     ↓              ↓              ↓                   ↓                ↓
  Episodes      Recorded       Failures          Distribution      "Collect 15 demos
                 data          flagged            gap found          in dim lighting"
```

## API Example

```python
from orbit.logger.episode_logger import EpisodeLogger
from orbit.logger.schemas import LoggerConfig, Outcome

config = LoggerConfig(storage_dir="./my_data", task_name="pick_and_place", robot_dof=6)

with EpisodeLogger(config) as logger:
    logger.start_episode()
    for step in range(50):
        logger.log_frame(
            joint_positions=joint_pos, gripper_state=gripper,
            action=action, reward=reward, images={"front": image},
        )
    episode = logger.end_episode(outcome=Outcome.SUCCESS)
```

## Dataset Profiler

Profile any robot dataset to understand its capabilities before training:

```bash
# Profile a local dataset
orbit profile --data-dir ./my_dataset --tasks "pick up cup" "open drawer"

# Profile from HuggingFace Hub
orbit profile --hub-repo lerobot/aloha_sim_insertion_human \
              --tasks "insert peg" --format json

# Compare two datasets
orbit profile-compare --dataset-a ./dataset_v1 --dataset-b ./dataset_v2 \
                      --tasks "pick and place"
```

The profiler analyzes embedding coverage, scores task capabilities, estimates data quality, and generates ranked prescriptions for what to collect next. Access it interactively via the dashboard's "Dataset Profiler" page.

## Built for LeRobot

ORBIT is designed as a companion to HuggingFace's [LeRobot](https://github.com/huggingface/lerobot) framework. It reads LeRobot dataset formats, plugs into LeRobot policy inference loops, and exports prescriptions as LeRobot-compatible data collection tasks.

## Installation

Full install with all dependencies:

```bash
pip install -e .
```

Lightweight install (no torch/transformers, for dashboard-only use):

```bash
pip install -e ".[light]"
```

> **Note:** Requires Python 3.10 or 3.11. The `lerobot` dependency does not yet support Python 3.12.

## Project Status

ORBIT is in active development (v1.1). The core pipeline works end-to-end on manipulation tasks with camera input. v1.1 adds dataset profiling for proactive coverage analysis. Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
