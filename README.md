<p align="center">
  <h1 align="center">ORBIT — Open Robot Iteration Toolkit</h1>
  <p align="center">
    Deployment diagnostics for learned robot policies.<br>
    Find why your robot fails, and what data to collect next.
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/License-Apache_2.0-green.svg" alt="License: Apache 2.0">
  <img src="https://img.shields.io/badge/tests-202_passed-brightgreen.svg" alt="Tests">
  <img src="https://img.shields.io/badge/version-1.1.0-orange.svg" alt="Version">
  <a href="https://huggingface.co/spaces/Drahils/orbit-demo"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Try_Demo-yellow.svg" alt="Demo"></a>
</p>

<p align="center">
  <img src="docs/dashboard-overview.png" alt="ORBIT Dashboard" width="720">
  <br>
  <em>ORBIT dashboard showing deployment session overview</em>
</p>

---

## What It Does

ORBIT watches your robot fail in deployment, diagnoses *why* by comparing deployment data against your training distribution, and prescribes exactly what demonstrations to collect next.

| Stage | What happens |
|-------|-------------|
| **Log** | Record every deployment episode (joints, images, actions, outcomes) to HDF5 |
| **Detect** | Flag failures automatically (gripper drops, stalls, timeouts, out-of-bounds, reward) |
| **Analyze** | Find distribution gaps between training and deployment using SigLIP vision embeddings |
| **Profile** | Score dataset capabilities, coverage density, and data quality per task |
| **Prescribe** | Get ranked, specific data collection tasks to close the gaps |

## Quick Start

```bash
pip install -e ".[profile]"
```

Generate sample data and launch the dashboard:

```bash
python scripts/generate_synthetic_deployment.py --output-dir ./demo_data
orbit dashboard --data-dir ./demo_data
```

Or try the [live demo on HuggingFace Spaces](https://huggingface.co/spaces/Drahils/orbit-demo) — no install required.

## Pipeline

```
Deploy Robot --> ORBIT Logger --> Failure Detector --> Embedding Analyzer --> Prescriber
     |               |                |                      |                   |
  Episodes       Recorded          Failures             Distribution       "Collect 15 demos
                  data              flagged              gap found           in dim lighting"
```

## Dataset Profiler (New in v1.1)

Profile any robot dataset to understand its capabilities *before* training:

```bash
# Profile a local dataset
orbit profile --data-dir ./my_dataset \
              --tasks "pick up cup" --tasks "open drawer"

# Profile from HuggingFace Hub
orbit profile --hub-repo lerobot/aloha_sim_insertion_human \
              --tasks "insert peg" --format json

# Compare two datasets side-by-side
orbit profile-compare --dataset-a ./dataset_v1 --dataset-b ./dataset_v2 \
                      --tasks "pick and place"
```

The profiler analyzes embedding coverage, scores task capabilities, estimates data quality, and generates ranked prescriptions for what to collect next. Also available interactively in the dashboard's **Dataset Profiler** page.

## API Example

```python
from orbit.logger.episode_logger import EpisodeLogger
from orbit.logger.schemas import LoggerConfig, Outcome

config = LoggerConfig(
    storage_dir="./my_data",
    task_name="pick_and_place",
    robot_dof=6,
)

with EpisodeLogger(config) as logger:
    logger.start_episode()
    for step in range(50):
        logger.log_frame(
            joint_positions=joint_pos,
            gripper_state=gripper,
            action=action,
            reward=reward,
            images={"front": image},
        )
    episode = logger.end_episode(outcome=Outcome.SUCCESS)
```

## Installation

**Standard** (core + dataset profiler):

```bash
pip install -e ".[profile]"
```

**Full** (everything including VLM, dashboard, LeRobot):

```bash
pip install -e ".[full]"
```

**Dashboard only** (lightweight, no torch):

```bash
pip install -e ".[dashboard]"
```

> **Note:** Requires Python 3.10 or 3.11. The `lerobot` dependency does not yet support Python 3.12.

## Built for LeRobot

ORBIT is designed as a companion to HuggingFace's [LeRobot](https://github.com/huggingface/lerobot) framework. It reads LeRobot dataset formats, plugs into LeRobot policy inference loops, and exports prescriptions as LeRobot-compatible data collection tasks.

## Project Structure

```
orbit/
  logger/        # Episode logging (HDF5 storage, background image saving)
  detector/      # Heuristic failure detection (5 detectors)
  analyzer/      # Embedding-based distribution gap analysis (SigLIP + FAISS)
  profile/       # Dataset profiling (coverage, capability, quality)   [NEW]
  prescriber/    # Data collection prescription generation
  vlm/           # Vision-language model failure description
  dashboard/     # Interactive Streamlit dashboard (6 pages)
  cli.py         # CLI: orbit dashboard | detect | profile | profile-compare
```

## Project Status

ORBIT v1.1 — actively developed. The core pipeline works end-to-end on manipulation tasks with camera input. v1.1 adds dataset profiling for proactive coverage analysis. Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
