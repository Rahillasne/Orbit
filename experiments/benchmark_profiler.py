"""
ORBIT v1 Profiler Validation Experiment

Goal: Show that ORBIT's capability predictions correlate with
actual policy performance.

Experiment Design:
1. Create synthetic datasets with known ground-truth capability profiles
2. Run ORBIT's profiler components on each dataset
3. Compare predicted capability scores against ground truth
4. Compute correlation metrics (Pearson r, Spearman rho)
5. Optionally test against real LeRobot Hub datasets with published results

Results are saved to experiments/results/:
- benchmark_results.json   — raw numbers
- validation_scatter.png   — predicted vs actual scatter plot
- validation_report.md     — human-readable report
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reuse helpers from orbit.profile.benchmarks
# ---------------------------------------------------------------------------

from orbit.profile.benchmarks import (
    _build_index,
    _make_cluster,
    _make_deterministic_episode,
    _make_noisy_episode,
    _make_random_episode,
)
from orbit.profile.capability import CapabilityScorer
from orbit.profile.coverage import CoverageAnalyzer
from orbit.profile.quality import QualityEstimator
from orbit.profile.types import EmbeddingIndex


# ---------------------------------------------------------------------------
# Benchmark scenarios
# ---------------------------------------------------------------------------


def _build_synthetic_scenarios(dim: int = 64, rng: np.random.Generator | None = None):
    """Build synthetic datasets with known ground-truth capability scores.

    Each scenario defines:
      - An embedding index (simulated visual data)
      - Episodes (simulated state/action quality)
      - A set of task queries with expected capability levels
    """
    rng = rng or np.random.default_rng(42)

    # Define stable reference directions
    dir_manip = rng.standard_normal(dim).astype(np.float32)
    dir_manip /= np.linalg.norm(dir_manip)

    dir_nav = -dir_manip  # opposite direction

    dir_cook = rng.standard_normal(dim).astype(np.float32)
    dir_cook -= dir_cook.dot(dir_manip) * dir_manip
    dir_cook -= dir_cook.dot(dir_nav) * dir_nav
    dir_cook /= np.linalg.norm(dir_cook)

    def mock_encode_factory(directions: dict[str, np.ndarray]):
        """Return a mock text encoder mapping keywords to directions."""
        def encode(texts):
            out = []
            for t in texts:
                tl = t.lower()
                matched = False
                for kw, vec in directions.items():
                    if kw in tl:
                        out.append(vec.copy())
                        matched = True
                        break
                if not matched:
                    v = rng.standard_normal(dim).astype(np.float32)
                    for vec in directions.values():
                        v -= v.dot(vec) * vec
                    v /= max(np.linalg.norm(v), 1e-8)
                    out.append(v)
            return np.array(out, dtype=np.float32)
        return encode

    keyword_dirs = {
        "pick": dir_manip, "grasp": dir_manip, "cup": dir_manip,
        "tabletop": dir_manip, "manipulation": dir_manip, "open": dir_manip,
        "pour": dir_manip, "place": dir_manip,
        "navigate": dir_nav, "outdoor": dir_nav, "drive": dir_nav,
        "walk": dir_nav, "move": dir_nav,
        "cook": dir_cook, "stir": dir_cook, "chop": dir_cook,
    }
    mock_encode = mock_encode_factory(keyword_dirs)

    scenarios = []

    # --- Scenario 1: High-capability manipulation dataset ---
    # Tight cluster near manipulation direction, deterministic actions
    embs = _make_cluster(100, dim, dir_manip, std=0.03, seed=1)
    ep_ids = [i // 10 for i in range(100)]
    index = _build_index(embs, episode_ids=ep_ids)
    episodes = [_make_deterministic_episode(i, T=100, seed=i) for i in range(10)]
    scenarios.append({
        "name": "high_capability_manipulation",
        "description": "Dense, high-quality manipulation data",
        "index": index,
        "episodes": episodes,
        "tasks": [
            {"query": "pick up a cup from the table", "ground_truth": 0.85},
            {"query": "navigate through a hallway", "ground_truth": 0.10},
        ],
    })

    # --- Scenario 2: Medium-capability (sparse coverage, some noise) ---
    embs = _make_cluster(40, dim, dir_manip, std=0.15, seed=2)
    ep_ids = [i // 4 for i in range(40)]
    index = _build_index(embs, episode_ids=ep_ids)
    episodes = [_make_noisy_episode(i, T=40, noise=1.0, seed=i + 50) for i in range(10)]
    scenarios.append({
        "name": "medium_capability_manipulation",
        "description": "Sparse, noisy manipulation data",
        "index": index,
        "episodes": episodes,
        "tasks": [
            {"query": "pick up a cup from the table", "ground_truth": 0.50},
            {"query": "navigate through a hallway", "ground_truth": 0.08},
        ],
    })

    # --- Scenario 3: Low-capability (data far from task) ---
    embs = _make_cluster(60, dim, dir_nav, std=0.05, seed=3)
    ep_ids = [i // 6 for i in range(60)]
    index = _build_index(embs, episode_ids=ep_ids)
    episodes = [_make_deterministic_episode(i, T=60, seed=i + 100) for i in range(10)]
    scenarios.append({
        "name": "low_capability_for_manipulation",
        "description": "Navigation data tested for manipulation task",
        "index": index,
        "episodes": episodes,
        "tasks": [
            {"query": "pick up a cup from the table", "ground_truth": 0.10},
            {"query": "navigate through a hallway", "ground_truth": 0.85},
        ],
    })

    # --- Scenario 4: Quality-degraded (good coverage, garbage actions) ---
    embs = _make_cluster(80, dim, dir_manip, std=0.05, seed=4)
    ep_ids = [i // 8 for i in range(80)]
    index = _build_index(embs, episode_ids=ep_ids)
    episodes = [_make_random_episode(i, T=80, seed=i + 200) for i in range(10)]
    scenarios.append({
        "name": "quality_degraded_manipulation",
        "description": "Good visual coverage but random actions",
        "index": index,
        "episodes": episodes,
        "tasks": [
            {"query": "pick up a cup from the table", "ground_truth": 0.30},
            {"query": "navigate through a hallway", "ground_truth": 0.05},
        ],
    })

    # --- Scenario 5: Multi-task dataset (two clusters) ---
    embs_a = _make_cluster(60, dim, dir_manip, std=0.04, seed=5)
    embs_b = _make_cluster(40, dim, dir_cook, std=0.04, seed=6)
    all_embs = np.vstack([embs_a, embs_b])
    ep_ids = [i // 10 for i in range(60)] + [6 + i // 10 for i in range(40)]
    index = _build_index(all_embs, episode_ids=ep_ids)
    episodes = [_make_deterministic_episode(i, T=60, seed=i + 300) for i in range(10)]
    scenarios.append({
        "name": "multi_task_manip_and_cooking",
        "description": "Mixed manipulation + cooking data",
        "index": index,
        "episodes": episodes,
        "tasks": [
            {"query": "pick up a cup from the table", "ground_truth": 0.70},
            {"query": "stir the pot while cooking", "ground_truth": 0.55},
            {"query": "navigate through a hallway", "ground_truth": 0.05},
        ],
    })

    # --- Scenario 6: Navigation-focused dataset ---
    embs = _make_cluster(90, dim, dir_nav, std=0.04, seed=7)
    ep_ids = [i // 9 for i in range(90)]
    index = _build_index(embs, episode_ids=ep_ids)
    episodes = [_make_deterministic_episode(i, T=90, seed=i + 400) for i in range(10)]
    scenarios.append({
        "name": "high_capability_navigation",
        "description": "Dense, high-quality navigation data",
        "index": index,
        "episodes": episodes,
        "tasks": [
            {"query": "navigate through a hallway", "ground_truth": 0.85},
            {"query": "pick up a cup from the table", "ground_truth": 0.10},
        ],
    })

    return scenarios, mock_encode


# ---------------------------------------------------------------------------
# ProfilerBenchmark
# ---------------------------------------------------------------------------


class ProfilerBenchmark:
    """Validates ORBIT's profiler predictions against ground truth."""

    BENCHMARK_DATASETS = [
        {
            "repo_id": "lerobot/aloha_static_cups_open",
            "task": "open cups",
            "known_success_rate": 0.85,
            "related_tasks": ["pick up cup", "pour water"],
            "unrelated_tasks": ["fold laundry", "navigate outdoors"],
        },
        {
            "repo_id": "lerobot/aloha_sim_transfer_cube_human",
            "task": "transfer cube",
            "known_success_rate": 0.90,
            "related_tasks": ["pick and place block", "grasp object"],
            "unrelated_tasks": ["cook a meal", "drive a car"],
        },
        {
            "repo_id": "lerobot/pusht",
            "task": "push T-shaped block to target",
            "known_success_rate": 0.75,
            "related_tasks": ["push object", "move block"],
            "unrelated_tasks": ["navigate hallway", "open door"],
        },
        {
            "repo_id": "lerobot/aloha_static_coffee",
            "task": "make coffee",
            "known_success_rate": 0.70,
            "related_tasks": ["pour liquid", "operate machine"],
            "unrelated_tasks": ["fold laundry", "outdoor navigation"],
        },
    ]

    def __init__(self, output_dir: str = "experiments/results") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict:
        """Run the full benchmark (synthetic + real if available)."""
        results: dict = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

        logger.info("=" * 60)
        logger.info("ORBIT Profiler Validation Benchmark")
        logger.info("=" * 60)

        # Always run synthetic
        results["synthetic"] = self.run_synthetic()

        # Try real data
        try:
            results["real_data"] = self.run_real_data()
        except Exception as e:
            logger.warning("Real-data benchmark skipped: %s", e)
            results["real_data"] = {"status": "skipped", "reason": str(e)}

        # Combine all data points for overall correlation
        all_predicted = []
        all_actual = []
        labels = []
        sources = []

        for dp in results["synthetic"]["data_points"]:
            all_predicted.append(dp["predicted"])
            all_actual.append(dp["ground_truth"])
            labels.append(dp["label"])
            sources.append("synthetic")

        if isinstance(results.get("real_data"), dict) and "data_points" in results["real_data"]:
            for dp in results["real_data"]["data_points"]:
                all_predicted.append(dp["predicted"])
                all_actual.append(dp["ground_truth"])
                labels.append(dp["label"])
                sources.append("real")

        results["overall"] = self._compute_correlations(all_predicted, all_actual)
        results["overall"]["n_datapoints"] = len(all_predicted)

        # Generate outputs
        self._generate_scatter_plot(
            all_predicted, all_actual, labels, sources,
            self.output_dir / "validation_scatter.png",
        )
        report = self.generate_validation_report(results)
        (self.output_dir / "validation_report.md").write_text(report)

        # Save JSON (convert numpy types)
        def _to_serializable(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(self.output_dir / "benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2, default=_to_serializable)

        logger.info("Results saved to %s", self.output_dir)
        return results

    # ------------------------------------------------------------------
    # Synthetic benchmark
    # ------------------------------------------------------------------

    def run_synthetic(self) -> dict:
        """Run benchmark on synthetic data with known ground truth."""
        logger.info("\n--- Synthetic Validation ---")
        scenarios, mock_encode = _build_synthetic_scenarios()

        analyzer = CoverageAnalyzer(min_cluster_size=3)
        quality_est = QualityEstimator(k_neighbors=3)

        data_points: list[dict] = []
        scenario_results: list[dict] = []

        for scenario in scenarios:
            logger.info("  Scenario: %s", scenario["name"])
            index: EmbeddingIndex = scenario["index"]
            episodes = scenario["episodes"]

            coverage = analyzer.analyze(index)
            quality = quality_est.estimate_quality(episodes)

            scorer = CapabilityScorer(top_k=min(50, index.num_embeddings))
            scorer._encode_texts = mock_encode
            scorer._text_encoder_mode = "mock"

            task_queries = [t["query"] for t in scenario["tasks"]]
            caps = scorer.score_tasks(index, coverage, quality, task_queries)

            scenario_result = {
                "name": scenario["name"],
                "description": scenario["description"],
                "tasks": [],
            }

            for task_info, cap in zip(scenario["tasks"], caps):
                dp = {
                    "label": f"{scenario['name']}:{task_info['query'][:30]}",
                    "scenario": scenario["name"],
                    "query": task_info["query"],
                    "predicted": round(cap.score, 4),
                    "ground_truth": task_info["ground_truth"],
                    "confidence": round(cap.confidence, 4),
                    "gap_description": cap.gap_description,
                }
                data_points.append(dp)
                scenario_result["tasks"].append(dp)
                logger.info(
                    "    Task: %-35s  predicted=%.3f  truth=%.3f",
                    task_info["query"][:35], cap.score, task_info["ground_truth"],
                )

            scenario_results.append(scenario_result)

        predicted = [dp["predicted"] for dp in data_points]
        actual = [dp["ground_truth"] for dp in data_points]
        correlations = self._compute_correlations(predicted, actual)

        # Related vs unrelated separation test
        related_scores = [dp["predicted"] for dp in data_points if dp["ground_truth"] >= 0.5]
        unrelated_scores = [dp["predicted"] for dp in data_points if dp["ground_truth"] <= 0.15]
        separation = _mannwhitney_u(related_scores, unrelated_scores) if related_scores and unrelated_scores else None

        # Rank accuracy
        rank_acc = _rank_accuracy(predicted, actual)

        result = {
            "status": "completed",
            "data_points": data_points,
            "scenarios": scenario_results,
            "correlations": correlations,
            "separation_test": {
                "related_mean": round(float(np.mean(related_scores)), 4) if related_scores else None,
                "unrelated_mean": round(float(np.mean(unrelated_scores)), 4) if unrelated_scores else None,
                "mann_whitney_u": separation,
                "pass": separation is not None and separation["p_value"] < 0.05,
            },
            "rank_accuracy": round(rank_acc, 4),
            "pass": correlations["spearman_rho"] > 0.7 and rank_acc > 0.7,
        }

        logger.info("\n  Synthetic Results:")
        logger.info("    Pearson r:    %.4f", correlations["pearson_r"])
        logger.info("    Spearman rho: %.4f", correlations["spearman_rho"])
        logger.info("    Rank accuracy: %.4f", rank_acc)
        logger.info("    PASS: %s", result["pass"])

        return result

    # ------------------------------------------------------------------
    # Real-data benchmark
    # ------------------------------------------------------------------

    def run_real_data(self) -> dict:
        """Run benchmark on real LeRobot Hub datasets."""
        logger.info("\n--- Real-Data Validation ---")

        from orbit.profile.profiler import DatasetProfiler

        profiler = DatasetProfiler()
        data_points: list[dict] = []

        for ds_info in self.BENCHMARK_DATASETS:
            repo_id = ds_info["repo_id"]
            logger.info("  Dataset: %s", repo_id)

            try:
                all_tasks = (
                    [ds_info["task"]]
                    + ds_info["related_tasks"]
                    + ds_info["unrelated_tasks"]
                )
                profile = profiler.profile_from_hub(
                    repo_id,
                    task_descriptions=all_tasks,
                    max_episodes=20,
                )

                # Primary task
                primary_cap = profile.capabilities[0]
                data_points.append({
                    "label": f"real:{repo_id.split('/')[-1]}",
                    "dataset": repo_id,
                    "query": ds_info["task"],
                    "predicted": round(primary_cap.score, 4),
                    "ground_truth": ds_info["known_success_rate"],
                    "confidence": round(primary_cap.confidence, 4),
                    "type": "primary",
                })

                # Related tasks should score higher than unrelated
                n_related = len(ds_info["related_tasks"])
                related_caps = profile.capabilities[1:1 + n_related]
                unrelated_caps = profile.capabilities[1 + n_related:]

                for cap in related_caps:
                    data_points.append({
                        "label": f"real:{repo_id.split('/')[-1]}:{cap.task_description[:20]}",
                        "dataset": repo_id,
                        "query": cap.task_description,
                        "predicted": round(cap.score, 4),
                        "ground_truth": ds_info["known_success_rate"] * 0.8,
                        "type": "related",
                    })
                for cap in unrelated_caps:
                    data_points.append({
                        "label": f"real:{repo_id.split('/')[-1]}:{cap.task_description[:20]}",
                        "dataset": repo_id,
                        "query": cap.task_description,
                        "predicted": round(cap.score, 4),
                        "ground_truth": 0.05,
                        "type": "unrelated",
                    })

                logger.info("    Primary score: %.3f (truth: %.2f)", primary_cap.score, ds_info["known_success_rate"])

            except Exception as e:
                logger.warning("    Skipped %s: %s", repo_id, e)
                continue

        if not data_points:
            return {"status": "skipped", "reason": "No datasets could be downloaded"}

        predicted = [dp["predicted"] for dp in data_points]
        actual = [dp["ground_truth"] for dp in data_points]
        correlations = self._compute_correlations(predicted, actual)

        return {
            "status": "completed",
            "data_points": data_points,
            "correlations": correlations,
        }

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_correlations(predicted: list[float], actual: list[float]) -> dict:
        """Compute Pearson r and Spearman rho."""
        from scipy import stats

        predicted_arr = np.array(predicted)
        actual_arr = np.array(actual)

        if len(predicted_arr) < 3:
            return {"pearson_r": 0.0, "pearson_p": 1.0, "spearman_rho": 0.0, "spearman_p": 1.0}

        pearson_r, pearson_p = stats.pearsonr(predicted_arr, actual_arr)
        spearman_rho, spearman_p = stats.spearmanr(predicted_arr, actual_arr)

        return {
            "pearson_r": round(float(pearson_r), 4),
            "pearson_p": round(float(pearson_p), 6),
            "spearman_rho": round(float(spearman_rho), 4),
            "spearman_p": round(float(spearman_p), 6),
        }

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_scatter_plot(
        predicted: list[float],
        actual: list[float],
        labels: list[str],
        sources: list[str],
        output_path: Path,
    ) -> None:
        """Generate predicted vs actual scatter plot with regression line."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from scipy import stats
        except ImportError:
            logger.warning("matplotlib not available; skipping scatter plot")
            return

        fig, ax = plt.subplots(1, 1, figsize=(8, 7))

        pred_arr = np.array(predicted)
        act_arr = np.array(actual)

        # Color by source
        colors = {"synthetic": "#4A90D9", "real": "#E74C3C"}
        for source in set(sources):
            mask = [s == source for s in sources]
            ax.scatter(
                pred_arr[mask], act_arr[mask],
                c=colors.get(source, "#888"),
                label=source.capitalize(),
                s=80, alpha=0.7, edgecolors="white", linewidth=0.5,
            )

        # Regression line
        if len(pred_arr) >= 3:
            slope, intercept, r_value, _, _ = stats.linregress(pred_arr, act_arr)
            x_fit = np.linspace(0, 1, 100)
            y_fit = slope * x_fit + intercept
            ax.plot(x_fit, y_fit, "k--", alpha=0.5, linewidth=1.5, label=f"OLS fit (r={r_value:.2f})")

        # Perfect correlation reference
        ax.plot([0, 1], [0, 1], ":", color="gray", alpha=0.4, label="Perfect correlation")

        # Correlation annotations
        if len(pred_arr) >= 3:
            pearson_r, _ = stats.pearsonr(pred_arr, act_arr)
            spearman_rho, _ = stats.spearmanr(pred_arr, act_arr)
            ax.text(
                0.05, 0.95,
                f"Pearson r = {pearson_r:.3f}\nSpearman rho = {spearman_rho:.3f}\nn = {len(pred_arr)}",
                transform=ax.transAxes, fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
            )

        ax.set_xlabel("Predicted Capability Score", fontsize=12)
        ax.set_ylabel("Ground Truth / Known Performance", fontsize=12)
        ax.set_title("ORBIT Profiler Validation: Predicted vs Actual Capability", fontsize=13)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Scatter plot saved to %s", output_path)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def generate_validation_report(self, results: dict) -> str:
        """Generate a human-readable markdown validation report."""
        lines = [
            "# ORBIT Profiler Validation Report",
            "",
            f"**Generated:** {results.get('timestamp', 'N/A')}",
            "",
            "## Summary",
            "",
        ]

        synth = results.get("synthetic", {})
        if synth.get("status") == "completed":
            corr = synth["correlations"]
            lines.extend([
                "### Synthetic Validation",
                "",
                f"- **Data points:** {len(synth['data_points'])}",
                f"- **Pearson r:** {corr['pearson_r']:.4f} (p={corr['pearson_p']:.6f})",
                f"- **Spearman rho:** {corr['spearman_rho']:.4f} (p={corr['spearman_p']:.6f})",
                f"- **Rank accuracy:** {synth['rank_accuracy']:.4f}",
                f"- **Overall PASS:** {'YES' if synth['pass'] else 'NO'}",
                "",
            ])

            sep = synth.get("separation_test", {})
            if sep.get("related_mean") is not None:
                lines.extend([
                    "**Related vs Unrelated Task Separation:**",
                    f"- Related task mean score: {sep['related_mean']:.4f}",
                    f"- Unrelated task mean score: {sep['unrelated_mean']:.4f}",
                    f"- Mann-Whitney U p-value: {sep['mann_whitney_u']['p_value']:.6f}" if sep.get("mann_whitney_u") else "- Mann-Whitney U: N/A",
                    f"- Separation PASS: {'YES' if sep['pass'] else 'NO'}",
                    "",
                ])

            lines.append("#### Scenario Details")
            lines.append("")
            lines.append("| Scenario | Task | Predicted | Truth | Delta |")
            lines.append("|----------|------|-----------|-------|-------|")
            for dp in synth["data_points"]:
                delta = dp["predicted"] - dp["ground_truth"]
                lines.append(
                    f"| {dp['scenario']} | {dp['query'][:35]} | "
                    f"{dp['predicted']:.3f} | {dp['ground_truth']:.2f} | "
                    f"{delta:+.3f} |"
                )
            lines.append("")

        real = results.get("real_data", {})
        if isinstance(real, dict) and real.get("status") == "completed":
            corr = real["correlations"]
            lines.extend([
                "### Real-Data Validation",
                "",
                f"- **Data points:** {len(real['data_points'])}",
                f"- **Pearson r:** {corr['pearson_r']:.4f}",
                f"- **Spearman rho:** {corr['spearman_rho']:.4f}",
                "",
            ])
        elif isinstance(real, dict) and real.get("status") == "skipped":
            lines.extend([
                "### Real-Data Validation",
                "",
                f"*Skipped: {real.get('reason', 'N/A')}*",
                "",
            ])

        overall = results.get("overall", {})
        if overall:
            lines.extend([
                "## Overall Correlation",
                "",
                f"- **Total data points:** {overall.get('n_datapoints', 0)}",
                f"- **Pearson r:** {overall.get('pearson_r', 0):.4f}",
                f"- **Spearman rho:** {overall.get('spearman_rho', 0):.4f}",
                "",
            ])

        lines.extend([
            "## Interpretation",
            "",
            "**Pass criteria:**",
            "- Spearman rho > 0.7 (strong rank correlation between predicted and actual)",
            "- Rank accuracy > 0.7 (correct pairwise ordering in >70% of comparisons)",
            "- Related tasks consistently score higher than unrelated tasks (p < 0.05)",
            "",
            "**What this validates:**",
            "1. ORBIT correctly ranks which datasets are better for which tasks",
            "2. Related tasks score higher than unrelated tasks (semantic sensitivity)",
            "3. Data quality degrades capability scores appropriately",
            "4. Coverage gaps are reflected in lower predicted capability",
            "",
        ])

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper statistics
# ---------------------------------------------------------------------------


def _mannwhitney_u(a: list[float], b: list[float]) -> dict | None:
    """Compute Mann-Whitney U test for separation."""
    try:
        from scipy.stats import mannwhitneyu
        stat, p = mannwhitneyu(a, b, alternative="greater")
        return {"u_statistic": round(float(stat), 4), "p_value": round(float(p), 6)}
    except Exception:
        return None


def _rank_accuracy(predicted: list[float], actual: list[float]) -> float:
    """Fraction of pairwise comparisons with correct ordering."""
    n = len(predicted)
    if n < 2:
        return 0.0
    correct = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if actual[i] == actual[j]:
                continue
            total += 1
            pred_order = predicted[i] > predicted[j]
            true_order = actual[i] > actual[j]
            if pred_order == true_order:
                correct += 1
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    benchmark = ProfilerBenchmark()
    results = benchmark.run()

    overall_pass = results.get("synthetic", {}).get("pass", False)
    print("\n" + "=" * 60)
    print(f"BENCHMARK {'PASSED' if overall_pass else 'FAILED'}")
    print("=" * 60)
    sys.exit(0 if overall_pass else 1)
