# Held-Out Evaluation Results

## Setup

- Total datasets: 37
- Train: 27, Test: 10
- Split: Stratified by success rate quartiles
- Model: RF

## Results

| Metric | Value |
|--------|-------|
| Spearman ρ | 0.596 [-0.168, 0.909] |
| Permutation p-value | 0.0724 |
| Pearson r | 0.576 |
| MAE | 0.121 |
| RMSE | 0.140 |
| Within ±0.15 | 50% |
| Within ±0.20 | 90% |

## Per-Dataset Predictions

| Dataset | Actual | Predicted | Error |
|---------|--------|-----------|-------|
| aloha_transfer_cube_human | 0.820 | 0.830 | +0.010 |
| lerobot_aloha_mobile_wash_pan | 0.700 | 0.684 | -0.016 |
| lerobot_aloha_static_vinh_cup_left | 0.680 | 0.716 | +0.036 |
| aloha_transfer_cube_scripted | 0.960 | 0.842 | -0.118 |
| lerobot_roboturk | 0.740 | 0.617 | -0.123 |
| lerobot_aloha_static_battery | 0.880 | 0.719 | -0.161 |
| lerobot_fmb | 0.820 | 0.658 | -0.162 |
| dp_robomimic_lift | 1.000 | 0.830 | -0.170 |
| droid_pick_place_cross_scene | 0.420 | 0.619 | +0.199 |
| lerobot_xarm_push_medium | 0.580 | 0.791 | +0.211 |
