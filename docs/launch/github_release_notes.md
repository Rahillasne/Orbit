# ORBIT v1.2.0 — Simplified Predictor, Ablation Studies, Clean Methodology

## What's New

### Simplified Quality Predictor
- **Random Forest model** (depth=3, 50 estimators) replaces the complex 4-model ensemble — simpler, more interpretable, and better validated
- **Action features removed** from the default feature set (52 features instead of 64) — ablation showed action features add noise, not signal (rho 0.63 → 0.61 on LOOCV)
- **PCA disabled by default** — PCA hurts on small training sets (n < 100); removing it improved LOOCV rho from 0.45 to 0.61
- **D4RL locomotion excluded** — ORBIT is scoped to vision-based manipulation; locomotion benchmarks were confounding the model

### Learned Scoring
- `CapabilityScorer.learn_weights()` — capability score demoted to human-readable summary since R² < 0.3; the RF predictor is now the primary output

### Extended Features (opt-in)
- 9 new temporal, cross-episode, embedding geometry, and interaction features via `DatasetFeatureExtractor(extended=True)` for 73-dim feature vectors

### New Modules
- `DatasetQualityModelV2` with isotonic calibration, bootstrap CIs, and OOD detection
- `ReportCardGenerator` with predicted success rate as primary output and graded report card (A–F)
- `orbit validate-benchmark` CLI command
- Adapters for LeRobot and RoboMimic dataset formats

## Key Result

**LOOCV Spearman rho = 0.61 (p < 0.001)** on 78 dataset–task–policy combinations from 9 published papers (Diffusion Policy, ALOHA, RoboMimic, RT-1, BridgeData V2, DROID, BC-Z, Octo, LeRobot).

Feature importance: `task_primary_score`, scale features, and embedding entropy dominate.

## Breaking Changes

- `DatasetQualityModelV2` now defaults to `use_pca=False` (previously `True`)
- New parameters: `model_type` (`"rf"` default, `"ensemble"` available) and `feature_set` (`"reduced"` default, `"full"` available)
- D4RL entries removed from `ground_truth.json` — if you extended the ground truth with locomotion data, you'll need to re-add those entries

## Test Suite

399 tests passing (up from 202 in v1.1).

## Installation

```bash
pip install orbit-robotics
```

With profiling support:

```bash
pip install orbit-robotics[profile]
```

Full install (all optional dependencies):

```bash
pip install orbit-robotics[full]
```

## Full Changelog

See [CHANGELOG.md](https://github.com/Rahillasne/Orbit/blob/main/CHANGELOG.md) for the complete list of changes across all versions.
