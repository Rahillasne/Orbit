# Model Ablation Results

**Dataset**: 90 datasets  
**Evaluation**: Leave-One-Out Cross-Validation (LOOCV)  
**Target**: Success rate in [0, 1]  

## Model Comparison

| Rank | Model | Spearman ρ | Pearson r | MAE | RMSE | Rank Acc. |
|------|-------|-----------|-----------|-----|------|----------|
| 1 | Model S5: GBR (depth=2, n=50, lr=0.1) | 0.666 | 0.673 | 0.132 | 0.179 | 75.0% |
| 2 | Model S3: RF (depth=3, n=50) | 0.601 | 0.605 | 0.145 | 0.193 | 71.9% |
| 3 | Current: AveragingEnsemble on PCA(12) | 0.598 | 0.581 | 0.156 | 0.204 | 71.8% |
| 4 | Model S6: GBR on PCA(5) | 0.597 | 0.600 | 0.148 | 0.195 | 72.1% |
| 5 | Model S4: RF on PCA(5) | 0.567 | 0.580 | 0.151 | 0.199 | 70.5% |
| 6 | Baseline E: Ridge (alpha=1.0, all 64) | 0.526 | 0.548 | 0.168 | 0.216 | 68.8% |
| 7 | Model S9: Ridge (alpha=10.0, all 64) | 0.524 | 0.517 | 0.165 | 0.213 | 68.5% |
| 8 | Model S8: Lasso (a=0.01) | 0.486 | 0.480 | 0.163 | 0.216 | 67.6% |
| 9 | Model S10: Ridge (alpha=100.0, all 64) | 0.423 | 0.446 | 0.164 | 0.218 | 65.2% |
| 10 | Baseline D: Linear regression (all 64) | 0.323 | 0.326 | 0.247 | 0.323 | 58.6% |
| 11 | Model S7: ElasticNet (a=0.1, l1=0.5) | 0.276 | 0.279 | 0.179 | 0.233 | 59.2% |
| 12 | Model S2: Ridge on PCA(3) | 0.154 | 0.310 | 0.178 | 0.232 | 55.1% |
| 13 | Model S1: Ridge on PCA(5) | 0.116 | 0.277 | 0.181 | 0.236 | 53.5% |
| 14 | Baseline B: Log(episodes) only | 0.098 | 0.203 | 0.185 | 0.238 | 49.4% |
| 15 | Baseline C: Top-1 capability only | -0.394 | -0.409 | 0.197 | 0.247 | 32.5% |
| 16 | Baseline A: Mean predictor | -1.000 | -1.000 | 0.195 | 0.245 | 0.0% |

## Best Model: Model S5: GBR (depth=2, n=50, lr=0.1)

- Spearman ρ = 0.666 (p = 0.0000)
- Pearson r = 0.673 (p = 0.0000)
- MAE = 0.132
- RMSE = 0.179
- Rank Accuracy = 75.0%

## Per-Dataset Predictions (Best Model)

| Dataset | Actual | Predicted | Error |
|---------|--------|-----------|-------|
| d4rl_hopper_expert | 1.000 | 1.000 | +0.000 ✓ |
| d4rl_halfcheetah_expert | 1.000 | 1.000 | +0.000 ✓ |
| lerobot_aloha_static_vinh_cup | 0.720 | 0.719 | -0.001 ✓ |
| d4rl_hopper_medium | 0.580 | 0.577 | -0.003 ✓ |
| aloha_transfer_cube_human | 0.820 | 0.825 | +0.005 ✓ |
| bcz_unseen_tasks | 0.440 | 0.430 | -0.010 ✓ |
| lerobot_aloha_static_coffee | 0.720 | 0.731 | +0.011 ✓ |
| lerobot_aloha_mobile_wash_pan | 0.700 | 0.688 | -0.012 ✓ |
| d4rl_halfcheetah_random | 0.354 | 0.342 | -0.012 ✓ |
| bridge_v2_put_in_container | 0.620 | 0.601 | -0.019 ✓ |
| octo_bridge_pick_place | 0.630 | 0.610 | -0.020 ✓ |
| robomimic_lift_ph_bc | 0.973 | 0.953 | -0.020 ✓ |
| bcz_push_object | 0.670 | 0.645 | -0.025 ✓ |
| lerobot_aloha_static_candy | 0.780 | 0.754 | -0.026 ✓ |
| d4rl_walker2d_medium_replay | 0.267 | 0.294 | +0.027 ✓ |
| d4rl_hopper_medium_replay | 0.486 | 0.457 | -0.029 ✓ |
| lerobot_stanford_kuka | 0.700 | 0.730 | +0.030 ✓ |
| dp_robomimic_toolhang | 0.747 | 0.714 | -0.033 ✓ |
| robomimic_transport_mh_bcrnn | 0.267 | 0.304 | +0.037 ✓ |
| droid_wipe_table | 0.550 | 0.588 | +0.038 ✓ |
| dp_pusht_image | 0.847 | 0.885 | +0.038 ✓ |
| lerobot_aloha_static_screw_driver | 0.680 | 0.720 | +0.040 ✓ |
| robomimic_lift_mh_bc | 0.847 | 0.805 | -0.042 ✓ |
| octo_finetuned_pick_place | 0.780 | 0.823 | +0.043 ✓ |
| robomimic_lift_ph_bcrnn | 1.000 | 0.955 | -0.045 ✓ |
| bridge_v2_drawer_open | 0.650 | 0.695 | +0.045 ✓ |
| lerobot_pusht_subtask | 0.830 | 0.784 | -0.046 ✓ |
| aloha_insertion_human | 0.860 | 0.809 | -0.051 ✓ |
| bcz_unseen_objects | 0.390 | 0.451 | +0.061 ✓ |
| robomimic_square_ph_bc | 0.493 | 0.556 | +0.063 ✓ |
| robomimic_can_mh_bc | 0.720 | 0.784 | +0.064 ✓ |
| lerobot_koch_pick_place_lego | 0.650 | 0.719 | +0.069 ✓ |
| dp_robomimic_lift | 1.000 | 0.929 | -0.071 ✓ |
| dp_robomimic_square | 0.892 | 0.820 | -0.072 ✓ |
| d4rl_halfcheetah_medium_replay | 0.462 | 0.538 | +0.076 ✓ |
| bridge_v2_sweep | 0.550 | 0.628 | +0.078 ✓ |
| robomimic_toolhang_ph_bcrnn | 0.360 | 0.281 | -0.079 ✓ |
| lerobot_iamlab_pickup_insert | 0.760 | 0.681 | -0.079 ✓ |
| robomimic_can_ph_bcrnn | 1.000 | 0.915 | -0.085 ✓ |
| robomimic_can_ph_bc | 1.000 | 0.915 | -0.085 ✓ |
| bcz_open_close | 0.580 | 0.669 | +0.089 ✓ |
| bcz_place_object | 0.720 | 0.631 | -0.089 ✓ |
| lerobot_aloha_static_vinh_cup_left | 0.680 | 0.771 | +0.091 ✓ |
| droid_drawer_open | 0.510 | 0.606 | +0.096 ✓ |
| lerobot_aloha_mobile_wipe_wine | 0.600 | 0.709 | +0.109 ✓ |
| lerobot_xarm_lift_medium | 0.720 | 0.611 | -0.109 ✓ |
| lerobot_xarm_lift_medium_replay | 0.650 | 0.539 | -0.111 ✓ |
| bridge_v2_fold_cloth | 0.490 | 0.377 | -0.113 ✓ |
| lerobot_aloha_static_tape | 0.620 | 0.733 | +0.113 ✓ |
| bridge_v2_pick_place | 0.710 | 0.589 | -0.121 ✓ |
| droid_pick_place_single | 0.680 | 0.552 | -0.128 ✓ |
| dp_robomimic_transport | 0.733 | 0.605 | -0.128 ✓ |
| lerobot_conq_hose | 0.550 | 0.681 | +0.131 ✓ |
| aloha_insertion_scripted | 0.940 | 0.808 | -0.132 ✓ |
| bridge_v2_drawer_close | 0.780 | 0.647 | -0.133 ✓ |
| lerobot_roboturk | 0.740 | 0.606 | -0.134 ✓ |
| robomimic_transport_ph_bcrnn | 0.433 | 0.299 | -0.134 ✓ |
| dp_pusht_state | 0.910 | 0.774 | -0.136 ✓ |
| robomimic_square_mh_bcrnn | 0.527 | 0.384 | -0.143 ✓ |
| rt1_unseen_objects | 0.530 | 0.675 | +0.145 ✓ |
| lerobot_fmb | 0.820 | 0.673 | -0.147 ✓ |
| dp_pusht_keypoints | 0.780 | 0.932 | +0.152 ✗ |
| lerobot_aloha_static_battery | 0.880 | 0.724 | -0.156 ✗ |
| octo_bridge_put_in | 0.480 | 0.640 | +0.160 ✗ |
| lerobot_umi_cup_in_wild | 0.800 | 0.639 | -0.161 ✗ |
| dp_robomimic_can | 0.992 | 0.830 | -0.162 ✗ |
| lerobot_aloha_static_towel | 0.560 | 0.732 | +0.172 ✗ |
| lerobot_aloha_static_fork_pick_up | 0.840 | 0.667 | -0.173 ✗ |
| octo_xarm_pick_place | 0.450 | 0.632 | +0.182 ✗ |
| rt1_unseen_tasks | 0.760 | 0.577 | -0.183 ✗ |
| octo_bridge_drawer | 0.520 | 0.704 | +0.184 ✗ |
| aloha_transfer_cube_scripted | 0.960 | 0.771 | -0.189 ✗ |
| d4rl_walker2d_random | 0.070 | 0.260 | +0.190 ✗ |
| lerobot_xarm_push_medium_replay | 0.520 | 0.715 | +0.195 ✗ |
| droid_pick_place_cross_scene | 0.420 | 0.635 | +0.215 ✗ |
| lerobot_xarm_push_medium | 0.580 | 0.796 | +0.216 ✗ |
| bcz_pick_object | 0.890 | 0.659 | -0.231 ✗ |
| robomimic_lift_mh_bcrnn | 1.000 | 0.766 | -0.234 ✗ |
| lerobot_aloha_static_thread_velcro | 0.480 | 0.739 | +0.259 ✗ |
| robomimic_square_ph_bcrnn | 0.720 | 0.437 | -0.283 ✗ |
| rt1_seen_tasks | 0.970 | 0.687 | -0.283 ✗ |
| d4rl_halfcheetah_medium | 0.444 | 0.751 | +0.307 ✗ |
| robomimic_can_mh_bcrnn | 1.000 | 0.683 | -0.317 ✗ |
| d4rl_hopper_random | 0.108 | 0.431 | +0.323 ✗ |
| rt1_distractor_backgrounds | 0.360 | 0.693 | +0.333 ✗ |
| d4rl_walker2d_expert | 1.000 | 0.559 | -0.441 ✗ |
| robomimic_toolhang_ph_bc | 0.000 | 0.487 | +0.487 ✗ |
| robomimic_transport_ph_bc | 0.000 | 0.491 | +0.491 ✗ |
| robomimic_square_mh_bc | 0.040 | 0.552 | +0.512 ✗ |
| d4rl_walker2d_medium | 0.792 | 0.221 | -0.571 ✗ |

## Diagnosis

- Mean residual: -0.0014
- Median residual: -0.0121
- No significant systematic bias

### 5 Worst Predictions

- **d4rl_walker2d_medium**: actual=0.792, pred=0.221, error=-0.571
- **robomimic_square_mh_bc**: actual=0.040, pred=0.552, error=+0.512
- **robomimic_transport_ph_bc**: actual=0.000, pred=0.491, error=+0.491
- **robomimic_toolhang_ph_bc**: actual=0.000, pred=0.487, error=+0.487
- **d4rl_walker2d_expert**: actual=1.000, pred=0.559, error=-0.441
