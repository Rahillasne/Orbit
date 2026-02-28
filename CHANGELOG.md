# Changelog

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
