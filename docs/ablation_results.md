# Ablation Study Results

## Experiment 1: Feature Group Ablation

| Configuration | Spearman ρ | MAE | RMSE |
|--------------|-----------|-----|------|
| All features (64) | 0.514 | 0.101 | 0.124 |
| Only embedding (20 dims) | 0.577 | 0.096 | 0.120 |
| Only action (12 dims) | 0.299 | 0.118 | 0.143 |
| Only quality (12 dims) | 0.414 | 0.108 | 0.136 |
| Only scale (8 dims) | 0.407 | 0.114 | 0.135 |
| Only task (12 dims) | 0.428 | 0.109 | 0.129 |
| Without embedding (44 dims) | 0.440 | 0.109 | 0.132 |
| Without action (52 dims) | 0.525 | 0.103 | 0.125 |
| Without quality (52 dims) | 0.540 | 0.100 | 0.124 |
| Without scale (56 dims) | 0.539 | 0.100 | 0.123 |
| Without task (52 dims) | 0.578 | 0.096 | 0.121 |

## Experiment 2: Scoring Weight Sensitivity

| Rank | Visual Rel. | Quality | Coverage | Volume | Spearman ρ |
|------|------------|---------|----------|--------|----------|
| 1 | 0.20 | 0.50 | 0.20 | 0.10 | 0.047 |
| 2 | 0.50 | 0.40 | 0.05 | 0.05 | 0.045 |
| 3 | 0.40 | 0.50 | 0.05 | 0.05 | 0.044 |
| 4 | 0.30 | 0.50 | 0.10 | 0.10 | 0.039 |
| 5 | 0.40 | 0.40 | 0.10 | 0.10 | 0.039 |
| 6 | 0.20 | 0.50 | 0.10 | 0.20 | 0.039 |
| 7 | 0.50 | 0.30 | 0.10 | 0.10 | 0.034 |
| 8 | 0.10 | 0.50 | 0.20 | 0.20 | 0.029 |
| 9 | 0.30 | 0.40 | 0.20 | 0.10 | 0.027 |
| 10 | 0.10 | 0.50 | 0.30 | 0.10 | 0.020 |

## Experiment 3: Learning Curve

| Training Size | Spearman ρ (mean) | 90% CI | MAE |
|--------------|------------------|--------|-----|
| 10 | 0.335 | [0.012, 0.546] | 0.119 |
| 15 | 0.332 | [-0.124, 0.649] | 0.120 |
| 20 | 0.496 | [0.149, 0.734] | 0.114 |
| 25 | 0.386 | [-0.006, 0.646] | 0.110 |
| 30 | 0.526 | [0.139, 0.857] | 0.111 |
| 35 | 0.000 | [0.000, 0.000] | 0.113 |
| 37 | 0.514 | [0.514, 0.514] | 0.101 |

## Experiment 5: Error Analysis

- Mean residual: -0.0106
- Std residual: 0.1234
- Within ±0.10: 51.4%
- Within ±0.15: 67.6%
- Within ±0.20: 86.5%
