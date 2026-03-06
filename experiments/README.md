# ORBIT Profiler Validation Experiments

## What This Tests

This benchmark validates that ORBIT's dataset profiler makes meaningful
predictions: datasets that are good for a task should receive high capability
scores, and datasets that are irrelevant should receive low scores.

Specifically, the experiment tests three claims:

1. **Rank correlation** -- Predicted capability scores correlate with ground-truth
   performance (Spearman rho > 0.7).
2. **Semantic sensitivity** -- Related tasks (e.g., "pick up cup" for a cup-manipulation
   dataset) score higher than unrelated tasks (e.g., "navigate outdoors"). Tested via
   Mann-Whitney U (p < 0.05).
3. **Quality sensitivity** -- Datasets with random/noisy actions receive lower
   capability scores than identical visual data with deterministic actions.

## How to Run

```bash
# From the repository root:
python experiments/benchmark_profiler.py

# Or as a module:
python -m experiments.benchmark_profiler
```

**Dependencies:** Requires the `profile` extras (`pip install -e ".[profile]"`).
The synthetic benchmark needs only numpy, scipy, faiss, and scikit-learn.
The scatter plot requires matplotlib (`pip install matplotlib`).

**Real-data mode** (optional): If LeRobot datasets are downloadable in your
environment, the benchmark will also profile real Hub datasets against published
success rates. This requires network access and the `loader` extras.

## Output Files

All results are saved to `experiments/results/`:

| File | Description |
|------|-------------|
| `benchmark_results.json` | Raw data points, correlations, pass/fail status |
| `validation_scatter.png` | Scatter plot: predicted capability vs ground truth |
| `validation_report.md` | Human-readable report with tables and interpretation |

## Interpreting Results

**The scatter plot** shows each (predicted, actual) data point. Points near the
diagonal indicate accurate predictions. The regression line and correlation
coefficients are annotated.

**Pass criteria:**

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| Spearman rho | > 0.7 | Strong rank correlation |
| Rank accuracy | > 0.7 | >70% of pairwise orderings correct |
| Related vs unrelated | p < 0.05 | Statistically significant separation |

**Synthetic vs real:** Synthetic results are deterministic and always reproducible.
Real-data results depend on network availability and dataset content. Results are
color-coded in the scatter plot (blue = synthetic, red = real).

## Experiment Design

### Synthetic Scenarios

Six scenarios with controlled ground-truth capability:

1. **High-capability manipulation** -- Dense, deterministic data near manipulation embedding
2. **Medium-capability manipulation** -- Sparse, noisy data near manipulation embedding
3. **Low-capability for manipulation** -- Navigation data tested against manipulation query
4. **Quality-degraded** -- Good visual coverage but random (garbage) actions
5. **Multi-task** -- Mixed manipulation + cooking data, tested against 3 different queries
6. **High-capability navigation** -- Dense navigation data (mirror of scenario 1)

### Real-Data Datasets

Four LeRobot Hub datasets with published benchmark success rates:
- `lerobot/aloha_static_cups_open` (85% success)
- `lerobot/aloha_sim_transfer_cube_human` (90% success)
- `lerobot/pusht` (75% success)
- `lerobot/aloha_static_coffee` (70% success)
