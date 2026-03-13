# ORBIT Benchmarks — Ground Truth Data

Ground truth (dataset, task, policy, success_rate) records scraped from published robotics papers, used for validating and training the ORBIT quality predictor.

## Files

| File | Entries | Purpose |
|------|---------|---------|
| `ground_truth.json` | 7 | Original curated set for basic validation |
| `ground_truth_comprehensive.json` | 72 | Extended set for learned predictor training |

The comprehensive file is a **superset** of the original schema — it adds new optional fields while keeping all required fields identical, so `BenchmarkValidator._load_ground_truth()` can load either file.

## Schema Reference

### Required fields (backward-compatible with v1.0)

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier (e.g., `dp_pusht_state`) |
| `repo_id` | string \| null | HuggingFace repo (e.g., `lerobot/pusht`) or `null` if unavailable |
| `task_description` | string | Natural-language task description |
| `reported_success_rate` | float | Best reported success rate (0-1) or normalized score |
| `metric_type` | string | `success_rate` or `normalized_score` (D4RL) |
| `source` | string | Full paper citation |
| `policy` | string | Policy architecture that achieved this result |
| `notes` | string | Caveats, verification status |
| `max_episodes` | int \| null | Episode limit for profiling (null = skip profiling) |

### New fields (v2.0)

| Field | Type | Description |
|-------|------|-------------|
| `dataset` | string | Human-readable dataset name |
| `downloadable` | bool | Whether ORBIT can download and profile this dataset |
| `source_table` | string | Specific table/figure reference in the paper |
| `num_demos` | int | Number of demonstrations in the dataset |
| `estimated_features` | object | Feature estimates for non-downloadable datasets |

### `estimated_features` schema (non-downloadable only)

```json
{
  "num_episodes": 130000,
  "avg_episode_length": 40,
  "action_dims": 7,
  "image_resolution": [320, 240],
  "diversity_estimate": "high",
  "quality_estimate": "high"
}
```

## Source Papers

| # | Key | Paper | Venue | Entries | Downloadable |
|---|-----|-------|-------|---------|-------------|
| 1 | `chi2023diffusion` | Diffusion Policy (Chi et al.) | RSS 2023 | 8 | Yes (LeRobot) |
| 2 | `zhao2023aloha` | ACT / ALOHA (Zhao et al.) | RSS 2023 | 4 | Yes (LeRobot) |
| 3 | `mandlekar2021robomimic` | RoboMimic (Mandlekar et al.) | CoRL 2021 | 18 | Yes (robomimic) |
| 4 | `fu2020d4rl` | D4RL (Fu et al.) | NeurIPS 2020 | 12 | No (state-only) |
| 5 | `brohan2023rt1` | RT-1 (Brohan et al.) | RSS 2023 | 4 | No (proprietary) |
| 6 | `walke2023bridgev2` | Bridge Data V2 (Walke et al.) | CoRL 2023 | 6 | Partial |
| 7 | `khazatsky2024droid` | DROID (Khazatsky et al.) | RSS 2024 | 4 | Yes (HF) |
| 8 | `jang2022bcz` | BC-Z (Jang et al.) | CoRL 2022 | 6 | No (proprietary) |
| 9 | `team2024octo` | Octo (Team et al.) | RSS 2024 | 5 | Partial |
| 10 | `lerobot2024` | LeRobot Benchmarks | HF 2024 | 6 | Yes |

## Caveats and Limitations

1. **Success rates measure policy performance, not dataset quality alone.** The same dataset can yield very different results with different policies (e.g., BC-RNN vs BC on RoboMimic Square). ORBIT uses the best-reported result as the dataset's "ceiling" performance, but also includes weaker policy results to study the relationship between data quality and policy sensitivity.

2. **D4RL entries use `normalized_score` (0-110+), not `success_rate` (0-1).** These are state-only locomotion tasks with no images, so ORBIT's vision-based profiler cannot process them directly. They are included for future state-based profiling and for training the learned predictor with appropriate metric normalization.

3. **Non-downloadable datasets** (RT-1, BC-Z, Octo WidowX) have `estimated_features` based on paper descriptions. These cannot be validated by profiling but provide training signal for the learned predictor.

4. **Some success rates need verification.** Entries marked with "TODO: verify" in their notes should be cross-checked against the cited tables before use in final model training.

5. **RoboMimic `repo_id` values** use the `robomimic/` prefix convention. These datasets are downloaded via the robomimic Python package, not directly from HuggingFace.

## Usage

```python
from orbit.benchmarks.validate_profiler import BenchmarkValidator

# Use the comprehensive dataset
validator = BenchmarkValidator(
    ground_truth_path="orbit/benchmarks/ground_truth_comprehensive.json"
)

# Run validation on downloadable entries only
report = validator.run()

# Or filter to specific entries
report = validator.run(dataset_ids=["dp_pusht_state", "aloha_transfer_cube_human"])
```

## Contributing

To add new entries:

1. Find the paper's main results table
2. Extract (dataset, task, policy, success_rate) for each row
3. Use the best policy result as the ceiling; add additional policies if they show interesting data-quality sensitivity
4. Add a source entry if the paper is new
5. Validate JSON: `python -m json.tool orbit/benchmarks/ground_truth_comprehensive.json`
