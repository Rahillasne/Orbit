"""Benchmark validation pipeline for ORBIT profiler predictions.

Correlates ORBIT capability profiler scores with known downstream task
performance from published papers to validate prediction quality.

Usage::

    python -m orbit.benchmarks.validate_profiler
    python -m orbit.benchmarks.validate_profiler --format json --output report.json
    python -m orbit.benchmarks.validate_profiler --dataset pusht_diffusion --max-episodes 10
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_GROUND_TRUTH_PATH = Path(__file__).parent / "ground_truth.json"
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "orbit" / "benchmark_profiles"
_BOOTSTRAP_ITERATIONS = 10_000
_BOOTSTRAP_CI = 0.95

# Reference tasks used alongside each target task so the CapabilityScorer
# activates relative (softmax) scoring instead of absolute similarities.
# These span diverse robotics domains to provide meaningful contrast.
_REFERENCE_TASKS = [
    "pick up an object from the table",
    "navigate through an obstacle course",
    "open a drawer and place an item inside",
    "pour liquid from a container into a cup",
    "fold a piece of cloth",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkEntry:
    """A single ground-truth benchmark record."""

    id: str
    repo_id: str
    task_description: str
    reported_success_rate: float
    metric_type: str
    source: str
    policy: str
    notes: str
    max_episodes: int | None = 50


@dataclass
class BenchmarkResult:
    """Result of profiling a single benchmark dataset."""

    entry: BenchmarkEntry
    orbit_score: float
    orbit_confidence: float
    ground_truth: float
    residual: float
    abs_error: float
    profiling_time_s: float
    error: str | None = None


@dataclass
class CorrelationResult:
    """Statistical correlation between predictions and ground truth."""

    spearman_rho: float
    spearman_p: float
    pearson_r: float
    pearson_p: float
    rank_accuracy: float
    n_samples: int


@dataclass
class BootstrapCI:
    """Bootstrap confidence interval for a statistic."""

    point_estimate: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    n_iterations: int


@dataclass
class ValidationReport:
    """Complete output of the benchmark validation pipeline."""

    results: list[BenchmarkResult]
    correlation: CorrelationResult | None
    bootstrap_spearman: BootstrapCI | None
    bootstrap_pearson: BootstrapCI | None
    failure_cases: list[BenchmarkResult]
    metadata: dict = field(default_factory=dict)

    # -- Serialization -------------------------------------------------------

    def to_dict(self) -> dict:
        """Convert the entire report to a JSON-safe dictionary."""

        def _entry_dict(r: BenchmarkResult) -> dict:
            return {
                "id": r.entry.id,
                "repo_id": r.entry.repo_id,
                "task": r.entry.task_description,
                "ground_truth": r.ground_truth,
                "orbit_score": round(r.orbit_score, 4),
                "orbit_confidence": round(r.orbit_confidence, 4),
                "residual": round(r.residual, 4),
                "abs_error": round(r.abs_error, 4),
                "profiling_time_s": round(r.profiling_time_s, 2),
                "source": r.entry.source,
                "policy": r.entry.policy,
                "error": r.error,
            }

        def _ci_dict(ci: BootstrapCI | None) -> dict | None:
            if ci is None:
                return None
            return asdict(ci)

        return {
            "results": [_entry_dict(r) for r in self.results],
            "correlation": asdict(self.correlation) if self.correlation else None,
            "bootstrap_spearman": _ci_dict(self.bootstrap_spearman),
            "bootstrap_pearson": _ci_dict(self.bootstrap_pearson),
            "failure_cases": [_entry_dict(r) for r in self.failure_cases],
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        lines: list[str] = []
        lines.append("# ORBIT Profiler Benchmark Validation\n")

        # -- Summary --
        lines.append("## Summary\n")
        n_success = sum(1 for r in self.results if r.error is None)
        n_failed = sum(1 for r in self.results if r.error is not None)
        lines.append(f"- **Datasets evaluated**: {n_success}")
        lines.append(f"- **Datasets failed to load**: {n_failed}")
        if self.correlation:
            c = self.correlation
            sig_spearman = "significant" if c.spearman_p < 0.05 else "not significant"
            sig_pearson = "significant" if c.pearson_p < 0.05 else "not significant"
            lines.append(
                f"- **Spearman rho**: {c.spearman_rho:.4f}"
                f" (p={c.spearman_p:.4f}, {sig_spearman})"
            )
            lines.append(f"- **Pearson r**: {c.pearson_r:.4f} (p={c.pearson_p:.4f}, {sig_pearson})")
            lines.append(f"- **Rank accuracy**: {c.rank_accuracy:.4f}")
        if self.bootstrap_spearman:
            bs = self.bootstrap_spearman
            lines.append(
                f"- **Spearman {bs.ci_level:.0%} CI**: [{bs.ci_lower:.4f}, {bs.ci_upper:.4f}]"
            )
        if self.bootstrap_pearson:
            bp = self.bootstrap_pearson
            lines.append(
                f"- **Pearson {bp.ci_level:.0%} CI**: [{bp.ci_lower:.4f}, {bp.ci_upper:.4f}]"
            )
        lines.append("")

        # -- Per-dataset results --
        lines.append("## Per-Dataset Results\n")
        lines.append("| Dataset | Ground Truth | ORBIT Score | Confidence | Residual | Source |")
        lines.append("|---------|-------------|-------------|------------|----------|--------|")
        for r in self.results:
            if r.error:
                lines.append(
                    f"| {r.entry.id} | {r.ground_truth:.2f} | ERROR | - | - | {r.entry.source} |"
                )
            else:
                lines.append(
                    f"| {r.entry.id} | {r.ground_truth:.2f} | {r.orbit_score:.4f} "
                    f"| {r.orbit_confidence:.4f} | {r.residual:+.4f} | {r.entry.source} |"
                )
        lines.append("")

        # -- Failure cases --
        if self.failure_cases:
            lines.append("## Failure Cases\n")
            lines.append(
                "Datasets where the profiler prediction diverged significantly "
                "from reported performance:\n"
            )
            for r in self.failure_cases:
                lines.append(f"### {r.entry.id}")
                lines.append(f"- **Ground truth**: {r.ground_truth:.2f} ({r.entry.policy})")
                lines.append(f"- **ORBIT score**: {r.orbit_score:.4f}")
                lines.append(f"- **Absolute error**: {r.abs_error:.4f}")
                direction = "overestimate" if r.residual > 0 else "underestimate"
                lines.append(f"- **Direction**: {direction}")
                lines.append(f"- **Notes**: {r.entry.notes}")
                lines.append("")

        # -- Statistical notes --
        lines.append("## Statistical Notes\n")
        lines.append(
            "- **Spearman rank correlation** measures whether ORBIT correctly "
            "orders datasets by quality (monotonic relationship)."
        )
        lines.append(
            "- **Pearson correlation** measures linear relationship between "
            "ORBIT scores and success rates."
        )
        lines.append(
            "- **Rank accuracy** is the fraction of all pairwise dataset "
            "comparisons where ORBIT predicts the correct relative ordering."
        )
        if self.bootstrap_spearman:
            lines.append(
                f"- **Bootstrap**: {self.bootstrap_spearman.n_iterations} iterations, "
                f"percentile method, seed=42."
            )
        lines.append(
            "- ORBIT scores are *not* calibrated to success rates — they reflect "
            "dataset capability (coverage, quality, relevance), not policy performance. "
            "Rank-order correlation is the primary validation metric."
        )
        lines.append(
            "- With small N, confidence intervals are wide. Expand ground_truth.json "
            "with more benchmarks to tighten estimates."
        )
        lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class BenchmarkValidator:
    """Correlates ORBIT profiler predictions with known benchmark results.

    Usage::

        validator = BenchmarkValidator()
        report = validator.run()
        print(report.to_markdown())
    """

    def __init__(
        self,
        ground_truth_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        max_episodes: int | None = None,
        embedding_model: str = "google/siglip-base-patch16-224",
        device: str = "cpu",
        failure_threshold: float = 0.25,
        bootstrap_iterations: int = _BOOTSTRAP_ITERATIONS,
        bootstrap_ci: float = _BOOTSTRAP_CI,
    ) -> None:
        self._ground_truth_path = (
            Path(ground_truth_path) if ground_truth_path else _GROUND_TRUTH_PATH
        )
        self._cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self._max_episodes = max_episodes
        self._embedding_model = embedding_model
        self._device = device
        self._failure_threshold = failure_threshold
        self._bootstrap_iterations = bootstrap_iterations
        self._bootstrap_ci = bootstrap_ci

    # -- Public API ----------------------------------------------------------

    def run(self, dataset_ids: list[str] | None = None, force: bool = False) -> ValidationReport:
        """Run the full validation pipeline.

        Parameters
        ----------
        dataset_ids:
            If provided, only validate these benchmark IDs.  ``None`` runs all.
        force:
            If ``True``, ignore cached scores and re-profile everything.

        Returns
        -------
        ValidationReport
            Complete validation results with correlations and analysis.
        """
        entries = self._load_ground_truth()
        if dataset_ids:
            id_set = set(dataset_ids)
            entries = [e for e in entries if e.id in id_set]
            missing = id_set - {e.id for e in entries}
            if missing:
                logger.warning("Unknown benchmark IDs (skipped): %s", missing)

        if not entries:
            logger.error("No benchmark entries to validate.")
            return ValidationReport(
                results=[],
                correlation=None,
                bootstrap_spearman=None,
                bootstrap_pearson=None,
                failure_cases=[],
                metadata={"error": "no entries"},
            )

        t0 = time.monotonic()
        results: list[BenchmarkResult] = []
        for entry in entries:
            logger.info("Profiling %s (%s) ...", entry.id, entry.repo_id)
            result = self._profile_single(entry, force=force)
            results.append(result)
            if result.error:
                logger.warning("  FAILED: %s", result.error)
            else:
                logger.info(
                    "  score=%.4f  ground_truth=%.2f  residual=%+.4f  (%.1fs)",
                    result.orbit_score,
                    result.ground_truth,
                    result.residual,
                    result.profiling_time_s,
                )

        successful = [r for r in results if r.error is None]
        correlation = self._compute_correlations(successful)
        bootstrap_spearman, bootstrap_pearson = self._bootstrap_correlation(successful)
        failure_cases = self._identify_failure_cases(successful)
        total_time = time.monotonic() - t0

        return ValidationReport(
            results=results,
            correlation=correlation,
            bootstrap_spearman=bootstrap_spearman,
            bootstrap_pearson=bootstrap_pearson,
            failure_cases=failure_cases,
            metadata={
                "total_time_s": round(total_time, 2),
                "embedding_model": self._embedding_model,
                "device": self._device,
                "n_entries": len(entries),
                "n_successful": len(successful),
                "n_failed": len(entries) - len(successful),
                "failure_threshold": self._failure_threshold,
                "bootstrap_iterations": self._bootstrap_iterations,
            },
        )

    # -- Ground truth loading ------------------------------------------------

    def _load_ground_truth(self) -> list[BenchmarkEntry]:
        with open(self._ground_truth_path) as f:
            data = json.load(f)

        entries: list[BenchmarkEntry] = []
        for b in data["benchmarks"]:
            entries.append(
                BenchmarkEntry(
                    id=b["id"],
                    repo_id=b["repo_id"],
                    task_description=b["task_description"],
                    reported_success_rate=b["reported_success_rate"],
                    metric_type=b["metric_type"],
                    source=b["source"],
                    policy=b["policy"],
                    notes=b.get("notes", ""),
                    max_episodes=b.get("max_episodes", 50),
                )
            )
        return entries

    # -- Single dataset profiling --------------------------------------------

    def _profile_single(self, entry: BenchmarkEntry, force: bool = False) -> BenchmarkResult:
        score_cache = self._cache_dir / "scores" / f"{entry.id}.json"
        if not force and score_cache.exists():
            try:
                cached = json.loads(score_cache.read_text())
                if cached.get("embedding_model") == self._embedding_model:
                    return BenchmarkResult(
                        entry=entry,
                        orbit_score=cached["orbit_score"],
                        orbit_confidence=cached["orbit_confidence"],
                        ground_truth=entry.reported_success_rate,
                        residual=cached["orbit_score"] - entry.reported_success_rate,
                        abs_error=abs(cached["orbit_score"] - entry.reported_success_rate),
                        profiling_time_s=0.0,
                    )
            except (json.JSONDecodeError, KeyError):
                logger.debug("Invalid cache for %s, re-profiling", entry.id)

        t0 = time.monotonic()
        try:
            from orbit.profile.profiler import DatasetProfiler

            profiler = DatasetProfiler(
                embedding_model=self._embedding_model,
                device=self._device,
            )
            max_ep = self._max_episodes if self._max_episodes is not None else entry.max_episodes
            dataset_cache = self._cache_dir / "datasets" / entry.id
            # Pass reference tasks alongside the target so the scorer uses
            # relative (softmax) scoring instead of absolute similarities,
            # which are near-zero for SigLIP on robotics images.
            all_tasks = [entry.task_description] + _REFERENCE_TASKS
            profile = profiler.profile_from_hub(
                repo_id=entry.repo_id,
                task_descriptions=all_tasks,
                max_episodes=max_ep,
                cache_dir=str(dataset_cache),
            )
            elapsed = time.monotonic() - t0

            if not profile.capabilities:
                return BenchmarkResult(
                    entry=entry,
                    orbit_score=0.0,
                    orbit_confidence=0.0,
                    ground_truth=entry.reported_success_rate,
                    residual=-entry.reported_success_rate,
                    abs_error=entry.reported_success_rate,
                    profiling_time_s=elapsed,
                    error="No capability scores returned",
                )

            # First capability corresponds to our target task
            cap = profile.capabilities[0]
            result = BenchmarkResult(
                entry=entry,
                orbit_score=cap.score,
                orbit_confidence=cap.confidence,
                ground_truth=entry.reported_success_rate,
                residual=cap.score - entry.reported_success_rate,
                abs_error=abs(cap.score - entry.reported_success_rate),
                profiling_time_s=elapsed,
            )

            # Cache the score
            score_cache.parent.mkdir(parents=True, exist_ok=True)
            score_cache.write_text(
                json.dumps(
                    {
                        "orbit_score": cap.score,
                        "orbit_confidence": cap.confidence,
                        "embedding_model": self._embedding_model,
                        "entry_id": entry.id,
                    },
                    indent=2,
                )
            )

            return result

        except Exception as exc:
            elapsed = time.monotonic() - t0
            return BenchmarkResult(
                entry=entry,
                orbit_score=0.0,
                orbit_confidence=0.0,
                ground_truth=entry.reported_success_rate,
                residual=0.0,
                abs_error=0.0,
                profiling_time_s=elapsed,
                error=str(exc),
            )

    # -- Statistical analysis ------------------------------------------------

    def _compute_correlations(self, results: list[BenchmarkResult]) -> CorrelationResult | None:
        if len(results) < 3:
            logger.warning(
                "Need at least 3 successful results for correlation; got %d",
                len(results),
            )
            return None

        from scipy import stats

        predicted = np.array([r.orbit_score for r in results])
        actual = np.array([r.ground_truth for r in results])

        spearman = stats.spearmanr(predicted, actual)
        pearson = stats.pearsonr(predicted, actual)
        rank_acc = self._rank_accuracy(predicted, actual)

        return CorrelationResult(
            spearman_rho=float(spearman.statistic),
            spearman_p=float(spearman.pvalue),
            pearson_r=float(pearson.statistic),
            pearson_p=float(pearson.pvalue),
            rank_accuracy=rank_acc,
            n_samples=len(results),
        )

    @staticmethod
    def _rank_accuracy(predicted: np.ndarray, actual: np.ndarray) -> float:
        """Fraction of pairwise comparisons where predicted ordering matches actual."""
        n = len(predicted)
        concordant = 0
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                if actual[i] == actual[j]:
                    continue
                total += 1
                if (predicted[i] - predicted[j]) * (actual[i] - actual[j]) > 0:
                    concordant += 1
        return concordant / total if total > 0 else 0.0

    def _bootstrap_correlation(
        self, results: list[BenchmarkResult]
    ) -> tuple[BootstrapCI | None, BootstrapCI | None]:
        if len(results) < 4:
            logger.warning("Need at least 4 results for bootstrap CI; got %d", len(results))
            return None, None

        from scipy import stats

        predicted = np.array([r.orbit_score for r in results])
        actual = np.array([r.ground_truth for r in results])
        n = len(results)
        rng = np.random.default_rng(42)

        spearman_boot = np.empty(self._bootstrap_iterations)
        pearson_boot = np.empty(self._bootstrap_iterations)

        for b in range(self._bootstrap_iterations):
            idx = rng.choice(n, size=n, replace=True)
            # Skip degenerate samples (need at least 3 unique indices)
            if len(set(idx)) < 3:
                spearman_boot[b] = np.nan
                pearson_boot[b] = np.nan
                continue
            p_boot = predicted[idx]
            a_boot = actual[idx]
            # Constant arrays cause errors in correlation
            if np.std(p_boot) == 0 or np.std(a_boot) == 0:
                spearman_boot[b] = np.nan
                pearson_boot[b] = np.nan
                continue
            spearman_boot[b] = stats.spearmanr(p_boot, a_boot).statistic
            pearson_boot[b] = stats.pearsonr(p_boot, a_boot).statistic

        corr = self._compute_correlations(results)
        alpha = 1 - self._bootstrap_ci

        def _ci(samples: np.ndarray, point: float) -> BootstrapCI:
            valid = samples[~np.isnan(samples)]
            if len(valid) == 0:
                return BootstrapCI(
                    point_estimate=point,
                    ci_lower=float("nan"),
                    ci_upper=float("nan"),
                    ci_level=self._bootstrap_ci,
                    n_iterations=0,
                )
            return BootstrapCI(
                point_estimate=point,
                ci_lower=float(np.percentile(valid, 100 * alpha / 2)),
                ci_upper=float(np.percentile(valid, 100 * (1 - alpha / 2))),
                ci_level=self._bootstrap_ci,
                n_iterations=int(len(valid)),
            )

        return (
            _ci(spearman_boot, corr.spearman_rho),
            _ci(pearson_boot, corr.pearson_r),
        )

    def _identify_failure_cases(self, results: list[BenchmarkResult]) -> list[BenchmarkResult]:
        failures = [r for r in results if r.abs_error > self._failure_threshold]
        failures.sort(key=lambda r: r.abs_error, reverse=True)
        return failures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli_main() -> None:
    import click

    @click.command()
    @click.option(
        "--cache-dir", default=None, type=click.Path(),
        help="Cache directory for profiles.",
    )
    @click.option(
        "--output", "-o", default=None, type=click.Path(),
        help="Output file path.",
    )
    @click.option(
        "--format", "fmt",
        type=click.Choice(["markdown", "json"]),
        default="markdown",
        help="Report format.",
    )
    @click.option(
        "--max-episodes", default=None, type=int,
        help="Override max episodes per dataset.",
    )
    @click.option(
        "--dataset", multiple=True,
        help="Specific dataset IDs to validate (default: all).",
    )
    @click.option("--force", is_flag=True, help="Ignore cached scores and re-profile.")
    @click.option("--device", default="cpu", help="Torch device (cpu or cuda).")
    @click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
    def main(
        cache_dir: str | None,
        output: str | None,
        fmt: str,
        max_episodes: int | None,
        dataset: tuple[str, ...],
        force: bool,
        device: str,
        verbose: bool,
    ) -> None:
        """Validate ORBIT profiler predictions against known benchmark results."""
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format="%(levelname)s %(name)s: %(message)s",
        )

        validator = BenchmarkValidator(
            cache_dir=cache_dir,
            max_episodes=max_episodes,
            device=device,
        )
        dataset_ids = list(dataset) if dataset else None
        report = validator.run(dataset_ids=dataset_ids, force=force)

        if fmt == "json":
            text = report.to_json()
        else:
            text = report.to_markdown()

        if output:
            Path(output).write_text(text)
            click.echo(f"Report saved to {output}")
        else:
            click.echo(text)

    main()


if __name__ == "__main__":
    _cli_main()
