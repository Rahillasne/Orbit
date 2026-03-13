<p align="center">
  <h1 align="center">ORBIT — Open Robot Iteration Toolkit</h1>
  <p align="center">
    Understand what your robot dataset can do before you deploy.
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/License-Apache_2.0-green.svg" alt="License: Apache 2.0">
  <img src="https://img.shields.io/badge/tests-399_passed-brightgreen.svg" alt="Tests">
  <img src="https://img.shields.io/badge/version-1.2.0-orange.svg" alt="Version">
  <a href="https://huggingface.co/spaces/Drahils/orbit-demo"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Try_Demo-yellow.svg" alt="Demo"></a>
</p>

---

You collected 500 episodes. Can your robot actually do the task? What data are you missing?

Your sim policy works great. Will it transfer to real? What's the distribution gap?

ORBIT profiles your robot dataset, scores per-task capability, finds what's missing, and tells you exactly what to collect next.

**Example profiler output:**

| Task | Capability | Confidence | Episodes | Gap |
|------|-----------|------------|----------|-----|
| pick up cup | 0.82 | 0.91 | 47 | — |
| open drawer | 0.34 | 0.78 | 12 | Low action diversity in pull motions |
| stack blocks | 0.15 | 0.65 | 5 | Need 30+ demos, poor lighting coverage |

---

## What ORBIT Does

- **Dataset capability profiling** — Score how ready your data is for each task, before you train.
- **Distribution gap analysis** — Find what visual and action conditions your dataset misses using SigLIP embeddings.
- **Data collection prescriptions** — Get ranked, specific instructions for what demonstrations to collect next.

## Quick Start

```bash
pip install -e ".[profile]"
```

```python
from orbit.profile.profiler import DatasetProfiler

profiler = DatasetProfiler()
result = profiler.profile("./my_dataset", task_descriptions=["pick up cup", "open drawer"])
result.capabilities   # per-task CapabilityScore list
result.prescriptions  # ranked data collection tasks
```

Or use the CLI:

```bash
orbit profile --data-dir ./my_dataset --tasks "pick up cup" --tasks "open drawer"
```

Profile a HuggingFace dataset:

```bash
orbit profile --hub-repo lerobot/aloha_sim_insertion_human \
              --tasks "insert peg" --format json
```

Compare two datasets side-by-side:

```bash
orbit profile-compare --dataset-a ./dataset_v1 --dataset-b ./dataset_v2 \
                      --tasks "pick and place"
```

Try the [live demo on HuggingFace Spaces](https://huggingface.co/spaces/Drahils/orbit-demo) — no install required.

## How It Works

ORBIT extracts visual embeddings from your dataset frames using SigLIP, builds a FAISS coverage index over the embedding space, then scores per-task capability based on action diversity and environment diversity within relevant episodes. Sparse regions in the coverage map become concrete prescriptions: what scenes, lighting conditions, or object configurations you need more demonstrations of.

Works on CPU (fast mode) or GPU (full accuracy). When no GPU is detected, ORBIT automatically uses OpenCLIP ViT-B/32 for faster embedding extraction. You can also force fast mode with `--fast`:

```bash
orbit profile --fast --data-dir ./my_dataset --tasks "pick up cube"
```

Available commands:

| Command | What it does |
|---------|-------------|
| `orbit profile` | Profile a dataset for task capabilities and gaps |
| `orbit profile-compare` | Compare two dataset profiles side-by-side |
| `orbit dashboard` | Launch the interactive Streamlit dashboard |
| `orbit detect` | Run failure detection on deployment sessions |

## Ecosystem

ORBIT fits into the robot learning stack alongside tools that handle training and curation:

| Tool | Role |
|------|------|
| [LeRobot](https://github.com/huggingface/lerobot) | Trains policies from demonstrations |
| [ARES](https://github.com/ARISE-Initiative/ARES) | Curates and filters robot datasets |
| **ORBIT** | Tells you if your data is good enough before you spend GPU hours |

ORBIT reads LeRobot dataset formats natively and exports prescriptions as LeRobot-compatible data collection tasks.

## Also: Deployment Diagnostics

ORBIT can also watch your robot during deployment, detect failures automatically (gripper drops, stalls, timeouts, out-of-bounds), and analyze distribution gaps between your training data and deployment conditions.

```bash
# Run failure detection on a deployment session
orbit detect --session ./deployment_session.h5

# Launch dashboard to explore deployment data
orbit dashboard --data-dir ./orbit_data
```

This is useful once you've trained a policy and want to understand where it fails in the real world — closing the loop back to better data collection.

## Installation

**Recommended** (core + dataset profiler):

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

## Benchmarks

ORBIT's quality predictor achieves **LOOCV Spearman ρ = 0.61 (p < 0.001)** on 78 robotics datasets, validated against published success rates from 9 peer-reviewed papers.

![Spearman rho](https://img.shields.io/badge/Spearman_%CF%81-0.61-blue) ![Datasets](https://img.shields.io/badge/datasets-78-blue) ![p-value](https://img.shields.io/badge/p-<0.001-green)

**Validation:** 78 dataset–task–policy combinations from Diffusion Policy (Chi et al., RSS 2023), ALOHA (Zhao et al., RSS 2023), RoboMimic (Mandlekar et al., CoRL 2021), RT-1, BridgeData V2, DROID, BC-Z, Octo, and LeRobot. Leave-one-out cross-validation with proper train/test scaling. The predictor uses a simplified Random Forest (depth=3, 50 estimators) on 52 dataset features extracted from visual embeddings, quality signals, and scale metrics.

The predictor outputs an expected policy success rate with a calibrated confidence interval. The letter-grade report card is a human-readable summary; the predictor is the validated estimate.

## Status

ORBIT is in beta (v1.2). The profiler and quality predictor work end-to-end on vision-based manipulation tasks. API may change. Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
