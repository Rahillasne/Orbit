#!/usr/bin/env python3
"""
Test 3: Data Integrity — Traces data through the entire ORBIT pipeline.
Logger → Detector → Analyzer → Prescriber → Export
"""
import json
import csv
import io
import sys
import time
from collections import Counter
from pathlib import Path
from uuid import UUID

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path("/tmp/orbit-test-data")
SCORECARD = {}


def find_session_file():
    h5_files = list(DATA_DIR.glob("session_*.h5"))
    if not h5_files:
        print("❌ No session .h5 files found in", DATA_DIR)
        sys.exit(1)
    return h5_files[0]


def test_load():
    """1. LOAD: Load the synthetic deployment session."""
    print("=" * 60)
    print("1. LOAD: Loading synthetic deployment session")
    print("=" * 60)

    from orbit.logger.schemas import LoggerConfig
    from orbit.logger.storage import HDF5Storage

    session_file = find_session_file()
    print(f"  Session file: {session_file}")
    print(f"  File size: {session_file.stat().st_size / 1024:.1f} KB")

    config = LoggerConfig(storage_dir=str(DATA_DIR))
    storage = HDF5Storage(config)
    session_id = UUID(session_file.stem.replace("session_", ""))

    episode_list = storage.list_episodes(session_id=session_id)
    print(f"  Episodes listed: {len(episode_list)}")

    episodes = [storage.load_episode(sid, eid) for sid, eid in episode_list]
    print(f"  Number of episodes loaded: {len(episodes)}")

    outcomes = Counter(ep.outcome.value for ep in episodes)
    print(f"  Episode outcomes: {dict(outcomes)}")

    if len(episodes) != 20:
        print(f"  ❌ Expected 20 episodes, got {len(episodes)}")
        SCORECARD["Data Loading"] = "❌"
    elif outcomes.get("success", 0) != 12:
        print(f"  ❌ Expected 12 successes, got {outcomes.get('success', 0)}")
        SCORECARD["Data Loading"] = "❌"
    elif outcomes.get("failure", 0) != 8:
        print(f"  ❌ Expected 8 failures, got {outcomes.get('failure', 0)}")
        SCORECARD["Data Loading"] = "❌"
    else:
        print(f"  ✅ Correct: 12 success, 8 failure")
        SCORECARD["Data Loading"] = "✅"

    failure_types = Counter()
    for ep in episodes:
        if ep.outcome.value == "failure":
            ft = ep.metadata.get("failure_type", "unknown")
            failure_types[ft] += 1
    print(f"  Failure types: {dict(failure_types)}")

    return episodes


def test_detect(episodes):
    """2. DETECT: Run the detector pipeline on every episode."""
    print("\n" + "=" * 60)
    print("2. DETECT: Running detector pipeline")
    print("=" * 60)

    from orbit.detector.heuristic import (
        DetectorPipeline,
        GripperDropDetector,
        StallDetector,
        OutOfBoundsDetector,
        TimeoutDetector,
        RewardThresholdDetector,
    )

    pipeline = DetectorPipeline(
        detectors=[
            GripperDropDetector(),
            StallDetector(),
            OutOfBoundsDetector(),
            TimeoutDetector(),
            RewardThresholdDetector(),
        ]
    )

    true_failures = {str(ep.episode_id) for ep in episodes if ep.outcome.value == "failure"}
    detected_failures = set()

    all_results = {}
    results_list = pipeline.run_batch(episodes)
    for ep, result in zip(episodes, results_list):
        all_results[str(ep.episode_id)] = result
        if result.is_failure:
            detected_failures.add(str(ep.episode_id))

    true_pos = len(true_failures & detected_failures)
    false_pos = len(detected_failures - true_failures)
    false_neg = len(true_failures - detected_failures)

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0

    print(f"  True failures: {len(true_failures)}")
    print(f"  Detected failures: {len(detected_failures)}")
    print(f"  True positives: {true_pos}")
    print(f"  False positives: {false_pos}")
    print(f"  False negatives: {false_neg}")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall: {recall:.1%}")

    detector_counts = Counter()
    for result in results_list:
        for det in result.detections:
            detector_counts[det.detector_name] += 1
    print("\n  Detector breakdown:")
    for det_name, count in detector_counts.most_common():
        print(f"    {det_name}: {count} detections")

    false_pos_ids = detected_failures - true_failures
    if false_pos_ids:
        print(f"\n  ⚠️ False positives ({len(false_pos_ids)} episodes):")
        for fp_id in list(false_pos_ids)[:5]:
            result = all_results[fp_id]
            det_names = [d.detector_name for d in result.detections]
            print(f"    {fp_id[:8]}... flagged by: {det_names}")

    recall_ok = recall >= 0.75
    fp_ok = false_pos <= 2

    if not recall_ok:
        print(f"  ❌ Recall {recall:.1%} < 75%")
    if not fp_ok:
        print(f"  ❌ {false_pos} false positives > 2 max allowed")

    if recall_ok and fp_ok:
        SCORECARD["Detection"] = f"✅ (precision: {precision:.0%}, recall: {recall:.0%})"
    else:
        SCORECARD["Detection"] = f"❌ (precision: {precision:.0%}, recall: {recall:.0%}, FP: {false_pos})"

    return results_list


def test_analyze(episodes):
    """3. ANALYZE: Run embedding gap analysis."""
    print("\n" + "=" * 60)
    print("3. ANALYZE: Running embedding gap analysis")
    print("=" * 60)

    try:
        from orbit.analyzer.embedding_analyzer import EmbeddingAnalyzer

        from PIL import Image

        analyzer = EmbeddingAnalyzer()

        # Index training data from success episodes' images
        print("  Indexing training data from success episodes...")
        success_images = []
        for ep in episodes:
            if ep.outcome.value == "success":
                for frame in ep.frames[:5]:  # Sample 5 frames per success episode
                    img_path = frame.image_path
                    if img_path and Path(img_path).exists():
                        success_images.append(Image.open(img_path))
        print(f"  Collected {len(success_images)} training images from successes")

        if success_images:
            n_indexed = analyzer.index_training_data(success_images)
            print(f"  Indexed {n_indexed} embeddings")

        print("  Computing gap scores (loading OpenCLIP model)...")
        start = time.time()
        frame_results, episode_summaries = analyzer.compute_gap_scores(episodes)
        embed_time = time.time() - start
        print(f"  Time to compute: {embed_time:.1f}s")
        print(f"  Frame results: {len(frame_results)}")
        print(f"  Episode summaries: {len(episode_summaries)}")

        success_gaps = []
        failure_gaps = []
        for ep, summary in zip(episodes, episode_summaries):
            if ep.outcome.value == "success":
                success_gaps.append(summary.mean_gap)
            else:
                failure_gaps.append(summary.mean_gap)

        if success_gaps and failure_gaps:
            mean_s = sum(success_gaps) / len(success_gaps)
            mean_f = sum(failure_gaps) / len(failure_gaps)
            ratio = mean_f / mean_s if mean_s > 0 else float('inf')
            print(f"  Mean gap score (success): {mean_s:.4f}")
            print(f"  Mean gap score (failure): {mean_f:.4f}")
            print(f"  Gap ratio (failure/success): {ratio:.2f}x")

            if mean_f > mean_s:
                print("  ✅ Failed episodes have higher gap scores")
                SCORECARD["Gap Analysis"] = f"✅ (ratio: {ratio:.2f}x)"
            else:
                print("  ❌ Failed episodes do NOT have higher gap scores")
                SCORECARD["Gap Analysis"] = f"❌ (ratio: {ratio:.2f}x)"
        else:
            print("  ⚠️ Could not compare (empty groups)")
            SCORECARD["Gap Analysis"] = "⚠️"

        # Clustering
        print("\n  Running clustering...")
        try:
            cluster_report = analyzer.cluster_failures(episodes, frame_results)
            n_clusters = getattr(cluster_report, 'n_clusters', None)
            if n_clusters is None:
                n_clusters = len(set(getattr(cluster_report, 'labels', []))) - (1 if -1 in getattr(cluster_report, 'labels', []) else 0)
            print(f"  Number of clusters found: {n_clusters}")

            if n_clusters >= 2:
                print("  ✅ Found at least 2 distinct failure clusters")
                SCORECARD["Clustering"] = f"✅ ({n_clusters} clusters)"
            else:
                print(f"  ❌ Expected >= 2 clusters, found {n_clusters}")
                SCORECARD["Clustering"] = f"❌ ({n_clusters} clusters)"
        except Exception as e:
            print(f"  ⚠️ Clustering failed: {e}")
            import traceback
            traceback.print_exc()
            SCORECARD["Clustering"] = f"❌ ({e})"

    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        SCORECARD["Gap Analysis"] = f"❌ (import: {e})"
        SCORECARD["Clustering"] = f"❌ (import: {e})"
    except Exception as e:
        print(f"  ❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        SCORECARD["Gap Analysis"] = f"❌ ({e})"
        SCORECARD["Clustering"] = f"❌ ({e})"


def test_prescribe(episodes, pipeline_results):
    """4. PRESCRIBE: Run the prescriber (using the dashboard bridge)."""
    print("\n" + "=" * 60)
    print("4. PRESCRIBE: Running prescriber")
    print("=" * 60)

    try:
        from orbit.detector.legacy import DetectionResult
        from orbit.prescriber.prescriber import Prescriber

        # Convert new pipeline results to legacy DetectionResult format
        # (this is how the dashboard does it)
        legacy_results = []
        for ep, result in zip(episodes, pipeline_results):
            if result.is_failure:
                reasons = [d.description for d in result.detections]
                legacy_results.append(
                    DetectionResult(
                        episode_id=hash(str(ep.episode_id)) % (2**31),
                        is_failure=True,
                        failure_reasons=reasons,
                        confidence=result.failure_probability,
                    )
                )

        print(f"  Failure episodes to prescribe: {len(legacy_results)}")

        prescriber = Prescriber()
        report = prescriber.prescribe(legacy_results)

        prescriptions = report.prescriptions
        print(f"  Number of prescriptions generated: {len(prescriptions)}")
        print(f"  Summary: {report.summary}")

        if prescriptions:
            print(f"\n  Top 3 prescriptions:")
            for i, rx in enumerate(prescriptions[:3]):
                print(f"    {i+1}. [priority={rx.priority}, conf={rx.confidence:.2f}] {rx.prescription_type.value}")
                print(f"       {rx.title}")
                print(f"       {rx.description[:100]}...")

            all_have_priority = all(rx.priority > 0 for rx in prescriptions)
            all_have_desc = all(len(rx.description) > 0 for rx in prescriptions)

            if all_have_priority and all_have_desc:
                print("  ✅ All prescriptions have non-zero priorities and descriptions")
                SCORECARD["Prescriptions"] = f"✅ ({len(prescriptions)} tasks)"
            else:
                SCORECARD["Prescriptions"] = f"❌ (missing priorities or descriptions)"
        else:
            print("  ⚠️ No prescriptions generated")
            SCORECARD["Prescriptions"] = "⚠️ (0 tasks)"

        return report

    except Exception as e:
        print(f"  ❌ Prescriber failed: {e}")
        import traceback
        traceback.print_exc()
        SCORECARD["Prescriptions"] = f"❌ ({e})"
        return None


def test_export(report):
    """5. EXPORT: Test all export formats."""
    print("\n" + "=" * 60)
    print("5. EXPORT: Testing export formats")
    print("=" * 60)

    if report is None:
        print("  ❌ No report to export")
        SCORECARD["Exports"] = "❌ (no report)"
        return

    export_ok = True

    # JSON
    try:
        json_data = {
            "summary": report.summary,
            "num_failures_analyzed": report.num_failures_analyzed,
            "prescriptions": [
                {
                    "type": rx.prescription_type.value,
                    "title": rx.title,
                    "priority": rx.priority,
                    "confidence": rx.confidence,
                    "description": rx.description,
                    "evidence": rx.evidence,
                }
                for rx in report.prescriptions
            ],
        }
        json_str = json.dumps(json_data, indent=2)
        json.loads(json_str)
        print(f"  ✅ JSON export valid ({len(json_str)} bytes)")
    except Exception as e:
        print(f"  ❌ JSON export failed: {e}")
        export_ok = False

    # CSV
    try:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["type", "title", "priority", "confidence", "description"])
        for rx in report.prescriptions:
            writer.writerow([rx.prescription_type.value, rx.title, rx.priority, rx.confidence, rx.description])
        csv_str = output.getvalue()
        rows = list(csv.reader(io.StringIO(csv_str)))
        expected = len(report.prescriptions) + 1
        if len(rows) == expected:
            print(f"  ✅ CSV export valid ({len(rows)} rows)")
        else:
            print(f"  ⚠️ CSV rows: {len(rows)} vs expected {expected}")
    except Exception as e:
        print(f"  ❌ CSV export failed: {e}")
        export_ok = False

    # Markdown
    try:
        lines = ["# Prescription Report\n", f"**Summary:** {report.summary}\n"]
        lines.append("| # | Type | Title | Priority | Confidence |")
        lines.append("|---|------|-------|----------|------------|")
        for i, rx in enumerate(report.prescriptions):
            lines.append(f"| {i+1} | {rx.prescription_type.value} | {rx.title} | {rx.priority} | {rx.confidence:.2f} |")
        md_str = "\n".join(lines)
        print(f"  ✅ Markdown export valid ({len(md_str)} bytes, has table: {'|' in md_str})")
    except Exception as e:
        print(f"  ❌ Markdown export failed: {e}")
        export_ok = False

    SCORECARD["Exports"] = "✅" if export_ok else "❌"


def main():
    print("ORBIT v1.0 — Data Integrity Walkthrough")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR}\n")

    episodes = test_load()
    pipeline_results = test_detect(episodes)
    test_analyze(episodes)
    report = test_prescribe(episodes, pipeline_results)
    test_export(report)

    print("\n" + "=" * 60)
    print("SCORECARD")
    print("=" * 60)
    for metric, status in SCORECARD.items():
        print(f"  {metric:20s} {status}")


if __name__ == "__main__":
    main()
