# Contributing to ORBIT

Thanks for your interest in ORBIT! Here's how to get involved.

## Development Setup

```bash
git clone https://github.com/Rahillasne/Orbit.git
cd Orbit
pip install -e ".[dev,profile]"
```

## Running Tests

```bash
# Run full test suite
pytest tests/ -v

# Run a specific test file
pytest tests/test_profile_quality.py -v

# Run with coverage
pytest tests/ --cov=orbit --cov-report=term-missing
```

## Code Style

We use **ruff** for linting and formatting, and **mypy** for type checking.

```bash
# Lint
ruff check orbit/

# Auto-format
ruff format orbit/

# Type check
mypy orbit/ --ignore-missing-imports
```

Configuration lives in `pyproject.toml`:
- Target: Python 3.10
- Line length: 100
- Lint rules: E, F, I, W, UP

## Before Submitting a PR

1. Run `ruff check orbit/` and `ruff format orbit/` — fix all issues
2. Run `pytest tests/ -v` — all tests must pass
3. Add tests for any new features or bug fixes
4. Update `CHANGELOG.md` if your change is user-facing

## How to Profile a New Dataset

```bash
# Install with profiling support
pip install orbit-robotics[profile]

# Profile a local dataset
orbit profile --data-dir ./my_data --tasks "pick up cup" --format json --output profile.json

# Profile a HuggingFace dataset
orbit profile --repo-id lerobot/aloha_sim_insertion_human --tasks "insertion" --format json --output profile.json
```

## How to Submit Profiling Results as a PR

1. Profile your dataset (see above)
2. Add the result JSON to `results/` with a descriptive filename (e.g., `results/my_dataset_report.json`)
3. If you have ground-truth success rates, add a ground truth entry (see below)
4. Open a PR with the title: `Add profile: <dataset_name>`

## How to Add a New Ground Truth Entry

Ground truth entries live in `orbit/benchmarks/ground_truth.json`. Each entry needs:

```json
{
  "dataset": "lerobot/aloha_sim_insertion_human",
  "task": "insertion",
  "policy": "ACT",
  "success_rate": 0.85,
  "source": "Zhao et al., 2023 (Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware)",
  "num_eval_episodes": 50,
  "notes": "Table 1, sim insertion task"
}
```

**Required fields:**
- `dataset` — HuggingFace repo ID or descriptive name
- `task` — task description matching what you'd pass to `orbit profile --tasks`
- `policy` — policy architecture (e.g., ACT, Diffusion Policy, BC-RNN)
- `success_rate` — float 0.0–1.0 from the published paper
- `source` — paper citation with enough detail to find the result

**Optional but appreciated:**
- `num_eval_episodes` — how many episodes the success rate was computed over
- `notes` — where in the paper the number comes from (e.g., "Table 2, row 3")

To submit:
1. Fork the repo and add your entry to `ground_truth.json`
2. Run `orbit validate-benchmark` to verify the entry is well-formed
3. Open a PR with the title: `Add ground truth: <dataset> / <policy>`

## Areas Where Help Is Welcome

- Adding ground truth entries from published papers
- Profiling new datasets and sharing results
- Additional failure detectors for new robot types
- LeRobot dataset integration improvements
- Dashboard visualizations
- Documentation and examples
