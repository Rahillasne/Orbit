# Changelog

## v1.1.0 (2026-03-06)

Dataset Profiler release.

- **Dashboard**: New "Dataset Profiler" page with coverage map, capability scoring, quality histogram, prescriptions, and dataset comparison mode
- **CLI**: `orbit profile` command for profiling datasets from local directories or HuggingFace Hub
- **CLI**: `orbit profile-compare` command for side-by-side dataset comparison
- **Profile module**: `DatasetProfiler`, `CapabilityScorer`, `CoverageAnalyzer`, `QualityEstimator`, `ProfileReporter`
- Dataset loading from LeRobot format and HDF5 directories
- Synthetic benchmark generation for testing

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
