# ORBIT v1.0 — Launch Readiness Report

**Date:** 2026-02-27
**Tested by:** Automated UAT Suite (7 tests)
**Verdict:** NOT READY — 5 critical issues must be fixed before public launch

---

## Executive Summary

ORBIT's core architecture is solid. The data pipeline (load → detect → prescribe) works end-to-end in under 100ms, the dashboard launches reliably, and edge case handling is excellent (34/34 edge cases pass). However, the **first-time user experience is broken** due to install issues, a non-functional README quickstart, and a miscalibrated detector that produces 100% false positives.

---

## Test Results Overview

| Test | Verdict | Key Finding |
|------|---------|-------------|
| 1. Fresh Install | ❌ FAIL | Install takes 28+ min, dependency resolution loops on `hf-transfer` |
| 2. Dashboard | ✅ PASS | 5/5 checks pass, loads in ~2s, no errors |
| 3. Data Flow | ⚠️ PARTIAL | Loading ✅, Detection ❌ (100% false positive), Analyzer ❌ (missing `sentencepiece`), Prescriber ✅, Exports ✅ |
| 4. Edge Cases | ✅ PASS | 34/34 edge cases handled gracefully |
| 5. README | ❌ FAIL | Grade: **C** — Quickstart code block has 5 distinct errors, no dashboard screenshot |
| 6. Performance | ⚠️ PARTIAL | Detection blazing fast (0.1ms), but frame logging 10x over target |
| 7. Readiness | ❌ NOT READY | 5 critical issues below |

---

## 1. First Impression (10-Second Test)

**Question:** If a robotics engineer lands on the GitHub page, will they understand what ORBIT does within 10 seconds?

**Answer:** Partially. The README title and module table are clear, but:
- No dashboard screenshot or GIF — the "wow" factor is invisible
- The GitHub links in the README are 404 (repo URL doesn't exist at `rahillasne/orbit`)
- No badges actually work (CI badge links to nonexistent repo)

---

## 2. Time to Value

**Question:** How long from `git clone` to seeing their first useful insight?

**Answer:** **Currently impossible in under 5 minutes.** The dependency resolution alone takes 28+ minutes due to `lerobot`'s `huggingface-hub[hf-transfer]` requirement churning through hundreds of version combinations. Even after install, the quickstart code in the README is completely non-functional.

**With a working README and pinned deps:** Could be under 3 minutes. The synthetic data generator takes 21s, detection takes 3ms, and the dashboard starts in 2s.

---

## 3. The Demo Moment

**What would I show in 2 minutes:**
1. `python scripts/generate_synthetic_deployment.py` (21s) — instant test data
2. `orbit dashboard --data-dir ./test_data` (2s) — interactive dashboard
3. Click through: Overview → see failure timeline → Prescriptions → see actionable recommendations

**Is there a "wow" moment?** Yes — the dashboard is genuinely useful and fast. But you can't get there without fighting the install and CLI setup.

---

## 4. Missing Pieces (What a LeRobot User Would Expect)

1. **LeRobot dataset integration** — No way to point ORBIT at an existing LeRobot dataset directory
2. **Real camera support** — Only works with pre-saved images, no live camera feed
3. **Model comparison** — Can't compare two policies side-by-side
4. **Export to LeRobot format** — The LeRobotExporter exists but isn't exposed in the CLI
5. **Automated report generation** — No CLI command to generate a PDF/HTML report

---

## 5. Complete Bug List

### Critical Bugs (Launch Blockers)

| # | Bug | Location | Impact |
|---|-----|----------|--------|
| 1 | **GripperDropDetector 100% false positive rate** — Flags ALL successful episodes as failures because normal gripper open/close triggers the detector | `orbit/detector/heuristic.py` GripperDropDetector | Core value proposition is broken |
| 2 | **README quickstart code is fictional** — `logger.begin_episode()`, `logger.log_step()`, `end_episode(success=True)` are all wrong method names | `README.md` lines 25-54 | Every new user will immediately crash |
| 3 | **`sentencepiece` not in dependencies** — Embedding analyzer fails with `ModuleNotFoundError` for `sentencepiece` (required by SiglipTokenizer/OpenCLIP) | `pyproject.toml` | Gap analysis and clustering completely broken |
| 4 | **`DetectorPipeline()` creates empty pipeline** — Constructor with no args creates a pipeline with 0 detectors that detects nothing. No warning. | `orbit/detector/heuristic.py` | Silent failure — users think everything is fine when nothing is being checked |
| 5 | **Install takes 28+ minutes** — `lerobot` dependency triggers massive `huggingface-hub` version resolution loop (hundreds of `hf-transfer` warnings) | `pyproject.toml` dependency on `lerobot>=0.4.0` | Users will give up before install finishes |

### Important Bugs (Should Fix)

| # | Bug | Location | Impact |
|---|-----|----------|--------|
| 6 | **CLI entry points not on PATH** — `orbit`, `orbit-dashboard`, `orbit-detect` commands not found after `pip install -e .` | Entry point registration | Users must use `python -m` workarounds |
| 7 | **Frame logging 10x over target** — 10.77ms/frame vs 1ms target (no images). With images: 7ms/frame vs 5ms target | `orbit/logger/episode_logger.py` | Real-time logging would add significant overhead |
| 8 | **GitHub links are 404** — README links to `github.com/rahillasne/orbit` which doesn't exist | `README.md` | Broken badges, no issue tracker link |
| 9 | **No dashboard screenshots** — README advertises Streamlit dashboard but shows no visuals | `README.md` | Missed opportunity for "wow" factor |
| 10 | **`ruff check` reports 54 errors** — Contributing section tells users to run linting, but codebase doesn't pass | Various files | Contributors will be confused |

### Minor Issues

| # | Issue | Location |
|---|-------|----------|
| 11 | `end_episode(outcome="success")` crashes — must use `Outcome.SUCCESS` enum, no string coercion | `orbit/logger/storage.py:267` |
| 12 | Verbose mode (`-v`) floods with hundreds of filelock debug messages | `orbit/detector/cli.py` |
| 13 | `from orbit import Prescriber` crashes if `open-clip-torch` not installed (eager import in `__init__.py`) | `orbit/prescriber/__init__.py` |
| 14 | Legacy `HeuristicDetector` and new `DetectorPipeline` return incompatible result types | `orbit/detector/` |
| 15 | `Prescriber` only accepts legacy `DetectionResult`, not new `PipelineResult` — requires manual conversion | `orbit/prescriber/prescriber.py` |

---

## 6. Priority Fix List (Top 5)

**Ordered by "would cause someone to give up and close the tab":**

### Fix 1: README Quickstart (30 min fix)
Replace the broken inline code with the working `examples/quickstart.py` content, or fix all 5 method name errors. Add a link to the example file. **This is the #1 reason someone would give up.**

### Fix 2: GripperDropDetector Calibration (1-2 hour fix)
The detector needs tuning — either raise the `min_closed_frames` threshold significantly (current default triggers on normal gripper behavior), or change the detection logic to require a larger gripper state delta. Currently makes the entire detection pipeline useless.

### Fix 3: Add `sentencepiece` to Dependencies (5 min fix)
Add `sentencepiece>=0.1.99` to `pyproject.toml` dependencies. Without it, the embedding analyzer (core differentiator of ORBIT) doesn't work at all.

### Fix 4: Default DetectorPipeline Should Include Detectors (15 min fix)
`DetectorPipeline()` with no args should either:
- Include all 5 detectors by default (like the CLI does), or
- Raise a warning/error that no detectors are configured

### Fix 5: Pin lerobot Version or Make Optional (30 min fix)
Either pin `lerobot==0.4.2` to avoid resolver churn, or make it an optional dependency (`pip install orbit[lerobot]`). The current `>=0.4.0` spec causes pip to evaluate thousands of `huggingface-hub` versions.

---

## Performance Summary

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Frame logging (no images) | < 1ms | 10.8ms | ❌ 10x over |
| Frame logging (with images) | < 5ms | 7.1ms | ⚠️ 1.4x over |
| Detection per episode | < 100ms | 0.1ms | ✅ 1000x under |
| Dashboard startup | < 5s | 2.0s | ✅ |
| Full pipeline (20 episodes) | < 60s | 0.08s | ✅ 750x under |

---

## Edge Case Resilience: EXCELLENT

34/34 edge cases pass without crashes:
- Empty directories, corrupted files, single episodes, all-success, all-failure, missing images, zero-frame episodes, extreme rewards — all handled gracefully.

This is a genuine strength of the codebase.

---

## Final Verdict

**ORBIT is NOT ready for v1.0 public launch.**

The core architecture is genuinely good — fast detection, solid data pipeline, robust edge case handling, clean dashboard. But the first-time user experience has multiple broken paths that would cause any new user to give up within minutes.

**Minimum fixes before launch:**
1. Fix README quickstart (non-negotiable)
2. Calibrate GripperDropDetector (core feature is broken)
3. Add `sentencepiece` dependency (analyzer is broken)
4. Add default detectors to `DetectorPipeline()` constructor
5. Pin `lerobot` version or make optional

**Estimated time to fix all 5:** 3-4 hours of focused work.

After these fixes, ORBIT would earn a solid v1.0 launch.
