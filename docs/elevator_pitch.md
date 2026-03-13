# ORBIT: Predict Robot Policy Success Without Training a Policy

## The Problem

Training a robot manipulation policy takes hours to days on expensive GPUs — and there's no way to know if your dataset will produce a good policy until you try. Failed training runs waste thousands of dollars in compute.

## What ORBIT Does

ORBIT profiles a robot dataset in **2–5 minutes on CPU** and predicts the success rate of the best policy you could train on it. No GPU needed. No policy training. Just dataset statistics.

## How It Works

ORBIT extracts 52 features from your dataset — visual embedding distributions, data quality signals, scale metrics, and task relevance scores — and feeds them into a Random Forest predictor trained on ground truth from 78 datasets across 9 published robotics papers.

## The Key Result

Via leave-one-out cross-validation on 78 datasets, ORBIT achieves **Spearman ρ = 0.61 (p < 0.001)**, correctly ranking the majority of datasets by expected policy performance.

## Three Surprising Findings

1. **Action features hurt.** Removing action-space statistics (smoothness, entropy, modes) *improves* prediction quality. What the robot *sees* matters more than what it *does*.

2. **Hand-tuned scores are random.** The capability score formula (a weighted sum of relevance, quality, coverage, volume) achieves ρ = 0.047 against ground truth — essentially noise. The learned predictor achieves 13× better correlation.

3. **Simplicity wins.** A Random Forest with 50 trees outperforms a 5-model stacking ensemble. In the small-data regime, fewer features and simpler models generalize better.

## What You Get

- A predicted success rate with 90% confidence interval
- A letter-graded report card (A–F) across coverage, quality, diversity, and volume
- Specific prescriptions: what data to collect next and how many demonstrations

## Who It's For

Robotics researchers and engineers who want to know if their dataset is good enough *before* spending GPU-hours training a policy, and what to collect next if it isn't.

**Open source**: `pip install orbit-profiler` | [GitHub](https://github.com/Drahils/orbit)
