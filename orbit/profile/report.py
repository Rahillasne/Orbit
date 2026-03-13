"""Structured report generation."""

from __future__ import annotations

import json
import logging

import numpy as np

from orbit.profile.types import DatasetProfile

logger = logging.getLogger(__name__)


class ProfileReporter:
    """Generate prescriptions, reports, and dashboard data from a DatasetProfile."""

    def __init__(self, low_score_threshold: float = 0.5) -> None:
        self.low_score_threshold = low_score_threshold

    # ------------------------------------------------------------------
    # Prescriptions
    # ------------------------------------------------------------------

    def generate_prescriptions(self, profile: DatasetProfile) -> list[dict]:
        """Generate ranked data collection prescriptions from the profile."""
        prescriptions: list[dict] = []

        for cap in profile.capabilities:
            if cap.score >= self.low_score_threshold:
                continue
            gap_score = 1.0 - cap.score
            estimated_demos = max(5, int(gap_score * 30))
            quality_weight = profile.quality.aggregate_score if profile.quality else 1.0
            raw_priority = gap_score * quality_weight

            gap_desc = cap.gap_description or f"Insufficient coverage for '{cap.task_description}'."
            instruction = (
                f"Collect {estimated_demos} demonstrations of '{cap.task_description}'. "
                f"Focus on: {gap_desc}"
            )

            prescriptions.append(
                {
                    "priority": 0,  # assigned after sort
                    "task": cap.task_description,
                    "instruction": instruction,
                    "estimated_demos": estimated_demos,
                    "gap_description": gap_desc,
                    "current_capability": round(cap.score, 3),
                    "target_capability": round(min(cap.score + 0.3, 1.0), 3),
                    "_sort_key": raw_priority,
                }
            )

        # Sort by gap severity (highest first)
        prescriptions.sort(key=lambda p: p["_sort_key"], reverse=True)
        for i, p in enumerate(prescriptions):
            p["priority"] = i + 1
            del p["_sort_key"]

        return prescriptions

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def generate_report(
        self,
        profile: DatasetProfile,
        format: str = "markdown",
    ) -> str | dict:
        """Generate a human-readable report.

        Parameters
        ----------
        profile:
            The dataset profile to report on.
        format:
            One of ``"markdown"``, ``"json"``, or ``"dict"``.
        """
        data = self._build_report_dict(profile)

        if format == "dict":
            return data
        if format == "json":
            return json.dumps(data, indent=2, default=str)
        return self._render_markdown(profile, data)

    def _build_report_dict(self, profile: DatasetProfile) -> dict:
        """Build a structured dict from the profile."""
        capabilities = []
        for cap in profile.capabilities:
            verdict = self._verdict(cap.score)
            entry: dict = {
                "task": cap.task_description,
                "score": round(cap.score, 3),
                "confidence": round(cap.confidence, 3),
                "supporting_episodes": cap.supporting_episodes,
                "verdict": verdict,
                "gap_description": cap.gap_description,
            }
            if cap.score_breakdown:
                entry["score_breakdown"] = {
                    "visual_relevance": round(cap.score_breakdown.visual_relevance, 3),
                    "data_quality": round(cap.score_breakdown.data_quality, 3),
                    "coverage_diversity": round(cap.score_breakdown.coverage_diversity, 3),
                    "volume": round(cap.score_breakdown.volume, 3),
                }
            capabilities.append(entry)

        quality_data: dict = {
            "aggregate_score": round(profile.quality.aggregate_score, 3),
            "mutual_information": round(profile.quality.mutual_information_estimate, 3),
            "low_quality_episodes": profile.quality.low_quality_episodes,
            "num_episodes_scored": len(profile.quality.episode_scores),
        }
        if profile.quality.signal_breakdown:
            sb = profile.quality.signal_breakdown
            quality_data["signal_breakdown"] = {
                "mutual_information": round(sb.mutual_information, 3),
                "action_smoothness": round(sb.action_smoothness, 3),
                "episode_completion": round(sb.episode_completion, 3),
                "observation_consistency": round(sb.observation_consistency, 3),
                "demonstration_quality": round(sb.demonstration_quality, 3),
            }

        # Top strengths and gaps
        sorted_caps = sorted(profile.capabilities, key=lambda c: c.score, reverse=True)
        strengths = [c.task_description for c in sorted_caps[:3] if c.score >= 0.5]
        gaps = [c.task_description for c in sorted_caps[-3:] if c.score < 0.5]

        return {
            "dataset_name": profile.dataset_name,
            "num_episodes": profile.num_episodes,
            "num_frames": profile.num_frames,
            "overall_coverage_score": round(profile.coverage.overall_coverage_score, 3),
            "strengths": strengths,
            "gaps": gaps,
            "capabilities": capabilities,
            "quality": quality_data,
            "coverage": {
                "num_dense_regions": len(profile.coverage.dense_regions),
                "num_sparse_regions": len(profile.coverage.sparse_regions),
                "overall_score": round(profile.coverage.overall_coverage_score, 3),
            },
            "prescriptions": profile.prescriptions,
            "timestamp": profile.timestamp,
        }

    def _render_markdown(self, profile: DatasetProfile, data: dict) -> str:
        """Render the report as markdown."""
        lines: list[str] = []

        # 1. Executive Summary
        lines.append(f"# Dataset Profile: {data['dataset_name']}")
        lines.append("")
        lines.append("## Executive Summary")
        lines.append("")
        lines.append(f"- **Episodes:** {data['num_episodes']}")
        lines.append(f"- **Frames:** {data['num_frames']}")
        lines.append(f"- **Overall Coverage:** {data['overall_coverage_score']:.3f}")
        lines.append(f"- **Quality (MI):** {data['quality']['mutual_information']:.3f}")
        lines.append("")
        if data["strengths"]:
            lines.append(f"**Top strengths:** {', '.join(data['strengths'])}")
        if data["gaps"]:
            lines.append(f"**Top gaps:** {', '.join(data['gaps'])}")
        lines.append("")

        # 2. Coverage Analysis
        lines.append("## Coverage Analysis")
        lines.append("")
        lines.append(f"- Dense regions: {data['coverage']['num_dense_regions']}")
        lines.append(f"- Sparse regions: {data['coverage']['num_sparse_regions']}")
        lines.append(f"- Overall coverage score: {data['coverage']['overall_score']:.3f}")
        lines.append("")

        # 3. Capability Breakdown
        lines.append("## Capability Breakdown")
        lines.append("")
        if data["capabilities"]:
            lines.append("| Task | Score | Confidence | Episodes | Verdict |")
            lines.append("|------|-------|------------|----------|---------|")
            for cap in data["capabilities"]:
                lines.append(
                    f"| {cap['task']} | {cap['score']:.3f} | "
                    f"{cap['confidence']:.3f} | {cap['supporting_episodes']} | "
                    f"{cap['verdict']} |"
                )
            lines.append("")

            # Score component breakdown
            has_breakdown = any("score_breakdown" in c for c in data["capabilities"])
            if has_breakdown:
                lines.append("### Score Component Breakdown")
                lines.append("")
                lines.append(
                    "| Task | Visual Relevance | Data Quality | Coverage/Diversity | Volume |"
                )
                lines.append(
                    "|------|-----------------|-------------|-------------------|--------|"
                )
                for cap in data["capabilities"]:
                    bd = cap.get("score_breakdown", {})
                    lines.append(
                        f"| {cap['task']} | {bd.get('visual_relevance', 0):.3f} | "
                        f"{bd.get('data_quality', 0):.3f} | "
                        f"{bd.get('coverage_diversity', 0):.3f} | "
                        f"{bd.get('volume', 0):.3f} |"
                    )
                lines.append("")
        else:
            lines.append("No tasks scored.")
        lines.append("")

        # 4. Quality Assessment
        lines.append("## Quality Assessment")
        lines.append("")
        lines.append(f"- Aggregate quality: {data['quality']['aggregate_score']:.3f}")
        lines.append(f"- Mutual information: {data['quality']['mutual_information']:.3f}")
        if data["quality"]["low_quality_episodes"]:
            lines.append(f"- Low quality episodes: {data['quality']['low_quality_episodes']}")
        lines.append("")

        # Quality signal breakdown
        if "signal_breakdown" in data["quality"]:
            sb = data["quality"]["signal_breakdown"]
            lines.append("### Quality Signal Breakdown")
            lines.append("")
            lines.append("| Signal | Score |")
            lines.append("|--------|-------|")
            lines.append(f"| Mutual Information | {sb['mutual_information']:.3f} |")
            lines.append(f"| Action Smoothness | {sb['action_smoothness']:.3f} |")
            lines.append(f"| Episode Completion | {sb['episode_completion']:.3f} |")
            lines.append(f"| Observation Consistency | {sb['observation_consistency']:.3f} |")
            lines.append(f"| Demonstration Quality | {sb['demonstration_quality']:.3f} |")
            lines.append("")

        # 5. Prescriptions
        lines.append("## Prescriptions")
        lines.append("")
        if data["prescriptions"]:
            for p in data["prescriptions"]:
                lines.append(
                    f"{p['priority']}. **[{p['task']}]** "
                    f"(current: {p['current_capability']:.3f}) — "
                    f"{p['instruction']}"
                )
        else:
            lines.append("No prescriptions — dataset looks good!")
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _verdict(score: float) -> str:
        if score >= 0.7:
            return "Strong"
        if score >= 0.5:
            return "Adequate"
        if score >= 0.2:
            return "Weak"
        return "No Coverage"

    # ------------------------------------------------------------------
    # Report card
    # ------------------------------------------------------------------

    def generate_report_card(self, profile: DatasetProfile):
        """Generate a graded report card from the profile.

        Returns a :class:`~orbit.profile.types.DatasetReportCard`.
        """
        from orbit.profile.report_card import ReportCardGenerator

        generator = ReportCardGenerator()
        return generator.generate(profile)

    # ------------------------------------------------------------------
    # Dashboard data
    # ------------------------------------------------------------------

    def generate_dashboard_data(self, profile: DatasetProfile) -> dict:
        """Generate data structure for Streamlit dashboard."""
        summary_stats = {
            "dataset_name": profile.dataset_name,
            "num_episodes": profile.num_episodes,
            "num_frames": profile.num_frames,
            "overall_coverage": round(profile.coverage.overall_coverage_score, 3),
            "aggregate_quality": round(profile.quality.aggregate_score, 3),
        }

        # Coverage UMAP
        coverage_umap = None
        if profile.coverage.umap_projection is not None:
            coverage_umap = profile.coverage.umap_projection.tolist()

        # Capability bars
        capability_bars = [
            {
                "task": c.task_description,
                "score": round(c.score, 3),
                "confidence": round(c.confidence, 3),
            }
            for c in profile.capabilities
        ]

        # Quality histogram
        scores = list(profile.quality.episode_scores.values())
        quality_histogram = {
            "scores": scores,
            "mean": round(float(np.mean(scores)) if scores else 0.0, 3),
            "std": round(float(np.std(scores)) if scores else 0.0, 3),
        }

        # Quality signal breakdown
        quality_signal_breakdown = None
        if profile.quality.signal_breakdown:
            sb = profile.quality.signal_breakdown
            quality_signal_breakdown = {
                "mutual_information": round(sb.mutual_information, 3),
                "action_smoothness": round(sb.action_smoothness, 3),
                "episode_completion": round(sb.episode_completion, 3),
                "observation_consistency": round(sb.observation_consistency, 3),
                "demonstration_quality": round(sb.demonstration_quality, 3),
            }

        # Prescription table
        prescription_table = profile.prescriptions

        # Report card (optional, may fail gracefully)
        report_card_data = None
        try:
            card = self.generate_report_card(profile)
            report_card_data = card.to_dict()
        except Exception:
            logger.debug("Report card generation failed — skipping in dashboard data")

        return {
            "summary_stats": summary_stats,
            "coverage_umap": coverage_umap,
            "capability_bars": capability_bars,
            "quality_histogram": quality_histogram,
            "quality_signal_breakdown": quality_signal_breakdown,
            "prescription_table": prescription_table,
            "report_card": report_card_data,
        }
