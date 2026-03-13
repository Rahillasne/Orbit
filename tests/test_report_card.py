"""Tests for the report card generation system."""

from __future__ import annotations

import json

import numpy as np

from orbit.profile.report_card import ReportCardGenerator, score_to_grade
from orbit.profile.types import (
    CapabilityScore,
    CoverageMap,
    DatasetGap,
    DatasetProfile,
    DatasetReportCard,
    EmbeddingIndex,
    Prescription,
    QualityMetrics,
    QualitySignalBreakdown,
    ScoreBreakdown,
    TaskAssessment,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_profile(
    num_episodes: int = 30,
    num_frames: int = 300,
    coverage_score: float = 0.65,
    quality_score: float = 0.72,
    capabilities: list[CapabilityScore] | None = None,
    mi: float = 0.55,
    smoothness: float = 0.70,
    completion: float = 0.60,
    consistency: float = 0.85,
    demo_quality: float = 0.65,
) -> DatasetProfile:
    """Create a synthetic DatasetProfile for testing."""
    rng = np.random.default_rng(42)

    DIM = 64
    n_embs = min(num_frames, 50)
    embs = rng.standard_normal((n_embs, DIM)).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.maximum(norms, 1e-8)

    import faiss

    index = faiss.IndexFlatIP(DIM)
    index.add(embs)

    embedding_index = EmbeddingIndex(
        index=index,
        episode_ids=list(range(n_embs)),
        frame_indices=list(range(n_embs)),
        dimension=DIM,
        num_embeddings=n_embs,
    )

    coverage = CoverageMap(
        dense_regions=[{"center": rng.standard_normal(DIM).astype(np.float32), "density": 0.8}],
        sparse_regions=[
            {
                "center": rng.standard_normal(DIM).astype(np.float32),
                "density": 0.1,
                "description": "peripheral region",
            }
        ]
        if coverage_score < 0.7
        else [],
        overall_coverage_score=coverage_score,
        umap_projection=None,
    )

    if capabilities is None:
        capabilities = [
            CapabilityScore(
                task_description="pick up cube",
                score=0.75,
                confidence=0.8,
                supporting_episodes=15,
                action_diversity=0.6,
                environment_diversity=0.5,
                gap_description=None,
                score_breakdown=ScoreBreakdown(
                    visual_relevance=0.7,
                    data_quality=0.8,
                    coverage_diversity=0.6,
                    volume=0.5,
                ),
            ),
            CapabilityScore(
                task_description="place on shelf",
                score=0.35,
                confidence=0.4,
                supporting_episodes=5,
                action_diversity=0.3,
                environment_diversity=0.2,
                gap_description="Low visual similarity to task 'place on shelf'",
                score_breakdown=ScoreBreakdown(
                    visual_relevance=0.3,
                    data_quality=0.7,
                    coverage_diversity=0.3,
                    volume=0.2,
                ),
            ),
        ]

    quality = QualityMetrics(
        episode_scores={i: quality_score + rng.uniform(-0.1, 0.1) for i in range(num_episodes)},
        aggregate_score=quality_score,
        low_quality_episodes=[],
        mutual_information_estimate=mi,
        signal_breakdown=QualitySignalBreakdown(
            mutual_information=mi,
            action_smoothness=smoothness,
            episode_completion=completion,
            observation_consistency=consistency,
            demonstration_quality=demo_quality,
        ),
    )

    return DatasetProfile(
        dataset_name="test_dataset",
        num_episodes=num_episodes,
        num_frames=num_frames,
        embedding_index=embedding_index,
        coverage=coverage,
        capabilities=capabilities,
        quality=quality,
        timestamp="2026-01-01T00:00:00",
    )


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


class TestScoreToGrade:
    def test_grade_A(self):
        assert score_to_grade(0.90) == "A"
        assert score_to_grade(0.85) == "A"

    def test_grade_B(self):
        assert score_to_grade(0.70) == "B"
        assert score_to_grade(0.84) == "B"

    def test_grade_C(self):
        assert score_to_grade(0.50) == "C"
        assert score_to_grade(0.69) == "C"

    def test_grade_D(self):
        assert score_to_grade(0.30) == "D"
        assert score_to_grade(0.49) == "D"

    def test_grade_F(self):
        assert score_to_grade(0.29) == "F"
        assert score_to_grade(0.0) == "F"

    def test_boundary_exactly(self):
        """Test exact boundary values."""
        assert score_to_grade(0.85) == "A"
        assert score_to_grade(0.70) == "B"
        assert score_to_grade(0.50) == "C"
        assert score_to_grade(0.30) == "D"

    def test_perfect_score(self):
        assert score_to_grade(1.0) == "A"


# ---------------------------------------------------------------------------
# Volume score
# ---------------------------------------------------------------------------


class TestVolumeScore:
    def test_volume_100_plus(self):
        gen = ReportCardGenerator()
        profile = _make_profile(num_episodes=150)
        score = gen._compute_volume_score(profile)
        assert score >= 0.85

    def test_volume_50_to_100(self):
        gen = ReportCardGenerator()
        profile = _make_profile(num_episodes=75)
        score = gen._compute_volume_score(profile)
        assert 0.65 <= score <= 0.85

    def test_volume_25_to_50(self):
        gen = ReportCardGenerator()
        profile = _make_profile(num_episodes=35)
        score = gen._compute_volume_score(profile)
        assert 0.45 <= score <= 0.65

    def test_volume_10_to_25(self):
        gen = ReportCardGenerator()
        profile = _make_profile(num_episodes=15)
        score = gen._compute_volume_score(profile)
        assert 0.25 <= score <= 0.45

    def test_volume_under_10(self):
        gen = ReportCardGenerator()
        profile = _make_profile(num_episodes=5)
        score = gen._compute_volume_score(profile)
        assert score < 0.25


# ---------------------------------------------------------------------------
# Strength / weakness detection
# ---------------------------------------------------------------------------


class TestStrengthDetection:
    def test_high_coverage_detected(self):
        gen = ReportCardGenerator()
        profile = _make_profile(coverage_score=0.85)
        strengths = gen._detect_strengths(profile, 0.85, 0.70, 0.50, 0.70)
        assert any("scene variety" in s.lower() or "coverage" in s.lower() for s in strengths)

    def test_high_quality_detected(self):
        gen = ReportCardGenerator()
        profile = _make_profile(quality_score=0.85)
        strengths = gen._detect_strengths(profile, 0.50, 0.85, 0.50, 0.70)
        assert any("quality" in s.lower() or "demonstration" in s.lower() for s in strengths)

    def test_high_smoothness_detected(self):
        gen = ReportCardGenerator()
        profile = _make_profile(smoothness=0.90)
        strengths = gen._detect_strengths(profile, 0.50, 0.70, 0.50, 0.70)
        assert any("smooth" in s.lower() or "trajectories" in s.lower() for s in strengths)

    def test_always_at_least_one_strength(self):
        gen = ReportCardGenerator()
        profile = _make_profile(
            coverage_score=0.10,
            quality_score=0.10,
            mi=0.1,
            smoothness=0.1,
            completion=0.1,
            consistency=0.1,
            demo_quality=0.1,
        )
        profile.capabilities = []
        strengths = gen._detect_strengths(profile, 0.10, 0.10, 0.10, 0.10)
        assert len(strengths) >= 1


class TestWeaknessDetection:
    def test_low_coverage_detected(self):
        gen = ReportCardGenerator()
        profile = _make_profile(coverage_score=0.20)
        weaknesses = gen._detect_weaknesses(profile, 0.20, 0.70, 0.50, 0.70)
        assert any("coverage" in w.lower() or "sparse" in w.lower() for w in weaknesses)

    def test_low_volume_detected(self):
        gen = ReportCardGenerator()
        profile = _make_profile(num_episodes=5)
        weaknesses = gen._detect_weaknesses(profile, 0.50, 0.70, 0.50, 0.10)
        assert any("episode" in w.lower() or "insufficient" in w.lower() for w in weaknesses)

    def test_redundant_dataset_detected(self):
        gen = ReportCardGenerator()
        profile = _make_profile(num_episodes=150, coverage_score=0.30)
        weaknesses = gen._detect_weaknesses(profile, 0.30, 0.70, 0.50, 0.90)
        assert any("redundant" in w.lower() for w in weaknesses)

    def test_high_quality_low_diversity_detected(self):
        gen = ReportCardGenerator()
        profile = _make_profile(quality_score=0.85)
        weaknesses = gen._detect_weaknesses(profile, 0.50, 0.85, 0.20, 0.70)
        assert any("limited" in w.lower() and "variation" in w.lower() for w in weaknesses)


# ---------------------------------------------------------------------------
# Full report card generation
# ---------------------------------------------------------------------------


class TestReportCardGeneration:
    def test_generates_valid_report_card(self):
        gen = ReportCardGenerator()
        profile = _make_profile()
        card = gen.generate(profile)

        assert isinstance(card, DatasetReportCard)
        assert card.overall_grade in ("A", "B", "C", "D", "F")
        assert 0.0 <= card.overall_score <= 1.0
        assert card.dataset_name == "test_dataset"

    def test_all_grades_present(self):
        gen = ReportCardGenerator()
        profile = _make_profile()
        card = gen.generate(profile)

        for grade_attr in ("coverage_grade", "quality_grade", "diversity_grade", "volume_grade"):
            assert getattr(card, grade_attr) in ("A", "B", "C", "D", "F")

    def test_task_assessments_match_capabilities(self):
        gen = ReportCardGenerator()
        profile = _make_profile()
        card = gen.generate(profile)

        assert len(card.task_assessments) == len(profile.capabilities)
        for ta in card.task_assessments:
            assert isinstance(ta, TaskAssessment)
            assert ta.grade in ("A", "B", "C", "D", "F")

    def test_strengths_and_weaknesses_populated(self):
        gen = ReportCardGenerator()
        profile = _make_profile()
        card = gen.generate(profile)

        assert isinstance(card.strengths, list)
        assert len(card.strengths) >= 1  # always at least one

    def test_gaps_are_dataset_gaps(self):
        gen = ReportCardGenerator()
        profile = _make_profile(coverage_score=0.40)
        card = gen.generate(profile)

        for gap in card.gaps:
            assert isinstance(gap, DatasetGap)
            assert gap.severity in ("Critical", "Major", "Minor")

    def test_prescriptions_are_prescriptions(self):
        gen = ReportCardGenerator()
        profile = _make_profile(coverage_score=0.40)
        card = gen.generate(profile)

        for p in card.prescriptions:
            assert isinstance(p, Prescription)
            assert p.priority >= 1
            assert p.estimated_demos > 0


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


class TestReportCardSerialization:
    def test_to_dict_is_serializable(self):
        gen = ReportCardGenerator()
        profile = _make_profile()
        card = gen.generate(profile)

        d = card.to_dict()
        # Should not raise
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_to_json_roundtrip(self):
        gen = ReportCardGenerator()
        profile = _make_profile()
        card = gen.generate(profile)

        json_str = card.to_json()
        parsed = json.loads(json_str)

        assert parsed["dataset_name"] == "test_dataset"
        assert parsed["overall_grade"] in ("A", "B", "C", "D", "F")
        assert "grades" in parsed
        assert "coverage" in parsed["grades"]

    def test_dict_has_all_fields(self):
        gen = ReportCardGenerator()
        profile = _make_profile()
        card = gen.generate(profile)

        d = card.to_dict()
        expected_keys = {
            "dataset_name",
            "overall_grade",
            "overall_score",
            "grades",
            "strengths",
            "weaknesses",
            "gaps",
            "prescriptions",
            "task_assessments",
            "timestamp",
        }
        assert expected_keys.issubset(set(d.keys()))


# ---------------------------------------------------------------------------
# CLI rendering
# ---------------------------------------------------------------------------


class TestCLIRendering:
    def test_render_produces_output(self):
        gen = ReportCardGenerator()
        profile = _make_profile()
        card = gen.generate(profile)

        output = gen.render_cli(card)
        assert isinstance(output, str)
        assert len(output) > 100
        assert "ORBIT" in output
        assert card.overall_grade in output

    def test_render_includes_dataset_name(self):
        gen = ReportCardGenerator()
        profile = _make_profile()
        card = gen.generate(profile)

        output = gen.render_cli(card)
        assert "test_dataset" in output

    def test_render_includes_task_grades(self):
        gen = ReportCardGenerator()
        profile = _make_profile()
        card = gen.generate(profile)

        output = gen.render_cli(card)
        assert "pick up cube" in output


# ---------------------------------------------------------------------------
# ProfileReporter integration
# ---------------------------------------------------------------------------


class TestProfileReporterIntegration:
    def test_generate_report_card_method(self):
        from orbit.profile.report import ProfileReporter

        reporter = ProfileReporter()
        profile = _make_profile()
        card = reporter.generate_report_card(profile)

        assert isinstance(card, DatasetReportCard)
        assert card.overall_grade in ("A", "B", "C", "D", "F")
