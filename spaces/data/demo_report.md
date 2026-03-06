# Dataset Profile: orbit_synthetic_benchmark

## Executive Summary

- **Episodes:** 40
- **Frames:** 400
- **Overall Coverage:** 0.720
- **Quality (MI):** 0.620

**Top strengths:** pick up red cup, open drawer, stack blocks
**Top gaps:** fold cloth napkin, navigate to kitchen, fly a helicopter

## Coverage Analysis

- Dense regions: 3
- Sparse regions: 2
- Overall coverage score: 0.720

## Capability Breakdown

| Task | Score | Confidence | Episodes | Verdict |
|------|-------|------------|----------|---------|
| pick up red cup | 0.850 | 0.900 | 24 | Strong |
| open drawer | 0.780 | 0.850 | 20 | Strong |
| stack blocks | 0.620 | 0.750 | 15 | Adequate |
| wipe surface with sponge | 0.550 | 0.700 | 12 | Adequate |
| pour water into glass | 0.350 | 0.600 | 8 | Weak |
| fold cloth napkin | 0.220 | 0.500 | 5 | Weak |
| navigate to kitchen | 0.080 | 0.300 | 2 | No Coverage |
| fly a helicopter | 0.020 | 0.150 | 0 | No Coverage |

## Quality Assessment

- Aggregate quality: 0.715
- Mutual information: 0.620

## Prescriptions

1. **[fly a helicopter]** (current: 0.020) — Collect 29 demonstrations of 'fly a helicopter'. Focus on: Completely outside the dataset's domain.
2. **[navigate to kitchen]** (current: 0.080) — Collect 27 demonstrations of 'navigate to kitchen'. Focus on: No visual coverage of navigation scenarios. Dataset is tabletop-only.
3. **[fold cloth napkin]** (current: 0.220) — Collect 23 demonstrations of 'fold cloth napkin'. Focus on: Few demonstrations of deformable object manipulation.
4. **[pour water into glass]** (current: 0.350) — Collect 19 demonstrations of 'pour water into glass'. Focus on: Limited coverage of pouring motions and liquid handling.
