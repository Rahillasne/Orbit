# Changelog

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
