"""Tests for orbit.detector module."""

from orbit.detector.legacy import DetectorConfig, HeuristicDetector


class TestHeuristicDetector:
    """Tests for HeuristicDetector."""

    def test_detect_success_episode(self, sample_episode):
        detector = HeuristicDetector()
        result = detector.detect(sample_episode)
        # A successful episode with decent reward should not be flagged
        # (unless it triggers length checks, etc.)
        assert result.episode_id == sample_episode.episode_id

    def test_detect_low_reward(self, failure_episode):
        detector = HeuristicDetector(DetectorConfig(reward_threshold=0.0))
        result = detector.detect(failure_episode)
        assert result.is_failure is True
        assert any("reward" in r.lower() for r in result.failure_reasons)

    def test_detect_stuck_robot(self, failure_episode):
        detector = HeuristicDetector(DetectorConfig(action_variance_threshold=0.01))
        result = detector.detect(failure_episode)
        assert result.is_failure is True
        assert any("variance" in r.lower() for r in result.failure_reasons)

    def test_detect_consecutive_failures(self, failure_episode):
        config = DetectorConfig(
            consecutive_failure_steps=3,
            min_reward_per_step=-0.1,
        )
        detector = HeuristicDetector(config)
        result = detector.detect(failure_episode)
        assert result.is_failure is True
        assert any("consecutive" in r.lower() for r in result.failure_reasons)

    def test_detect_short_episode(self, failure_episode):
        config = DetectorConfig(min_episode_length=20)
        detector = HeuristicDetector(config)
        result = detector.detect(failure_episode)
        assert any("short" in r.lower() for r in result.failure_reasons)

    def test_detect_batch(self, sample_episode, failure_episode):
        detector = HeuristicDetector()
        results = detector.detect_batch([sample_episode, failure_episode])
        assert len(results) == 2
        assert results[0].episode_id == sample_episode.episode_id
        assert results[1].episode_id == failure_episode.episode_id

    def test_detection_result_has_reasons(self, failure_episode):
        detector = HeuristicDetector()
        result = detector.detect(failure_episode)
        assert result.is_failure is True
        assert len(result.failure_reasons) > 0

    def test_custom_thresholds(self, sample_episode):
        # With very strict thresholds, even a "successful" episode can fail
        config = DetectorConfig(
            reward_threshold=100.0,  # impossibly high
            max_episode_length=5,  # very short max
        )
        detector = HeuristicDetector(config)
        result = detector.detect(sample_episode)
        assert result.is_failure is True

    def test_success_flag_check(self, failure_episode):
        detector = HeuristicDetector()
        result = detector.detect(failure_episode)
        assert any("success=False" in r for r in result.failure_reasons)

    def test_confidence_bounded(self, failure_episode):
        detector = HeuristicDetector()
        result = detector.detect(failure_episode)
        assert 0.0 <= result.confidence <= 1.0
