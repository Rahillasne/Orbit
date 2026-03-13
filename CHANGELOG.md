# Changelog

## v1.2.0 (2026-03-13)

Quality predictor release. Validated against 78 dataset–task–policy combinations from 9 published papers.

### Model Changes
- **Simplified RF model**: Random Forest (depth=3, 50 estimators) replaces complex 4-model ensemble
- **Removed action features**: 52-feature reduced set (no action) is the default — ablation showed action features add noise (rho 0.63 → 0.61 on LOOCV)
- **Extended features**: 9 new temporal, cross-episode, embedding geometry, and interaction features added to `DatasetFeatureExtractor` (opt-in via `extended=True`)
- **D4RL exclusion**: Locomotion benchmarks removed from ground truth — ORBIT is scoped to vision-based manipulation
- **PCA disabled by default**: `DatasetQualityModelV2(use_pca=False)` — PCA hurts on small training sets (n < 100); LOOCV ρ improved from 0.45 to 0.61 by removing PCA
- **Learned scoring weights**: `CapabilityScorer.learn_weights()` shows R² < 0.3; capability score demoted to human-readable summary, predictor is primary output

### Validation Results
- **LOOCV Spearman ρ = 0.61** (p < 0.001, n=78) — leave-one-out cross-validation on all 78 datasets
- Comprehensive ground truth: 78 entries from 9 papers (Diffusion Policy, ALOHA, RoboMimic, RT-1, BridgeData V2, DROID, BC-Z, Octo, LeRobot)
- Feature importance: task_primary_score, scale features, and embedding entropy dominate

### New Features
- `DatasetFeatureExtractor` with `extended=True` for 73-dim feature vectors
- `DatasetQualityModelV2` with PCA, isotonic calibration, bootstrap CIs, and OOD detection
- `ReportCardGenerator` with predicted success rate as primary output
- Graded report card (A–F) with strengths, weaknesses, gaps, and prescriptions
- `orbit validate-benchmark` CLI command for benchmark validation
- Adapters for LeRobot and RoboMimic dataset formats

### Tests
- 399 tests passing (up from 202 in v1.1)

## v1.1.0 (2026-03-06)

Dataset Profiler release.

### New Features
- **Profile module**: `DatasetProfiler`, `CapabilityScorer`, `CoverageAnalyzer`, `QualityEstimator`, `ProfileReporter`
- **Dashboard**: New "Dataset Profiler" page with coverage map, capability scoring, quality histogram, prescriptions, and dataset comparison mode
- **CLI**: `orbit profile` command for profiling datasets from local directories or HuggingFace Hub
- **CLI**: `orbit profile-compare` command for side-by-side dataset comparison
- Dataset loading from LeRobot format and HDF5 directories
- Synthetic benchmark generation and validation suite

### Bug Fixes
- Fix `transformers` v5 compatibility: handle `BaseModelOutputWithPooling` from `get_image_features()` / `get_text_features()`
- Fix macOS segfault caused by OpenMP threading conflict between FAISS/torch and HDBSCAN/UMAP

### Tests
- 202 tests passing (up from 128 in v1.0)

## v1.0.0 (2026-02-28)

Initial release.

- Episode logging with HDF5 storage and background image saving
- 5 heuristic failure detectors (gripper drop, stall, out-of-bounds, timeout, reward threshold)
- Embedding-based distribution gap analysis (SigLIP + FAISS)
- HDBSCAN failure clustering
- Automated data collection prescriptions
- Interactive Streamlit dashboard (5 pages)
- CLI tools: `orbit dashboard`, `orbit detect`
- Full test suite (128 tests, 34 edge cases)
