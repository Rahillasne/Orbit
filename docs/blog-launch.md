---
title: "ORBIT: Deployment Diagnostics for Robot Policies"
author: Rahil Lasne
date: 2026-02-28
tags: [robotics, deployment, lerobot, open-source]
---

# ORBIT: Deployment Diagnostics for Robot Policies

Every robotics engineer has had this experience: your policy works perfectly in the lab, then fails 40% of the time in deployment, and you have no idea why.

The standard workflow is: collect 200 more demonstrations, retrain, cross fingers. That's expensive, slow, and you're guessing about what data to collect. The LeRobot ecosystem has made policy training dramatically more accessible, but deployment debugging remains a manual, time-consuming process — scrubbing through hours of video, checking reward curves episode by episode, trying to spot patterns by eye.

## What ORBIT Does

ORBIT (Open Robot Iteration Toolkit) automates this diagnosis. It compares your deployment failures against your training data using vision embeddings, clusters the failure modes, and tells you exactly what demonstrations to collect.

The pipeline is four stages: **Log** episodes to HDF5 with zero overhead (images save in a background thread), **Detect** failures with five heuristic detectors (stall, gripper drop, out-of-bounds, timeout, reward), **Analyze** failure distributions using SigLIP embeddings and FAISS similarity search, and **Prescribe** ranked corrective actions with specific parameters.

## A Real Example

I ran ORBIT on a synthetic deployment: 20 episodes of a pick-and-place policy, 60% success rate. Without ORBIT, I'd know "8 episodes failed" and nothing else.

ORBIT identified two distinct failure clusters within seconds. The first cluster (5 episodes) showed near-zero action variance — the policy was stalling. Cross-referencing with frame data revealed these were lighting failures: dark observations caused the policy to freeze. The second cluster (3 episodes) had extreme joint positions near workspace boundaries paired with unexpected gripper drops.

![ORBIT Dashboard](dashboard-overview.png)

The prescriptions were specific: collect 15 demonstrations under varied lighting conditions and 10 with objects near workspace edges. That's 25 targeted demonstrations instead of 200 blind ones.

## Under the Hood

ORBIT is pure Python, pip-installable, and runs on CPU. Detection runs at ~0.1ms per episode. The embedding analyzer uses SigLIP, indexed with FAISS, and clusters failure modes with HDBSCAN. The architecture is modular — adding a custom detector means subclassing `BaseDetector` and implementing one `detect()` method. The codebase ships with 128 tests and 34 edge cases.

## Try It

**Live demo**: [HuggingFace Spaces](https://huggingface.co/spaces/Drahils/orbit-demo) — no install, loads in seconds.

**Run locally**:
```bash
pip install -e .
python scripts/generate_synthetic_deployment.py --output-dir ./demo_data
orbit dashboard --data-dir ./demo_data
```

**GitHub**: [github.com/Rahillasne/Orbit](https://github.com/Rahillasne/Orbit) — Apache 2.0, contributions welcome.

If you're deploying learned policies and hitting the deployment gap, I'd love to hear about your experience. [Open an issue](https://github.com/Rahillasne/Orbit/issues) or reach out.
