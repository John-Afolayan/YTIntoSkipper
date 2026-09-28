"""Submission and learning regressions with all external services mocked."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from audio_fingerprint import AudioAnalysisError
from automator import IntroSkipperAutomator
from models import IntroSegment
from video_db import VideoDB


class PipelineSafetyTests(unittest.TestCase):
    def app(self, confidence=.9, adjustment=0, workers=1):
        app = IntroSkipperAutomator.__new__(IntroSkipperAutomator)
        app.dry_run = False
        app.channel_id = "channel"
        app.manual_approval = False
        app.clipboard = False
        app.workers = workers
        app.downloader = Mock()
        app.downloader.extract_video_id.return_value = "video-id"
        app.downloader.download_audio.return_value = ("nonexistent-test-audio.wav", "video-id")
        app.sponsorblock = Mock()
        app.sponsorblock.get_segments.return_value = []
        app.sponsorblock.submit_segment.return_value = True
        app.db = Mock()
        app.db.is_processed.return_value = False
        app.feedback_store = Mock()
        app.adaptive_engine = Mock()
        app.adaptive_engine.apply_corrections.return_value = SimpleNamespace(
            corrected_start=0, corrected_end=10, should_reject=False,
            flags=[], confidence_adjustment=adjustment)
        app.find_intro_in_video = Mock(return_value=IntroSegment(0, 10, confidence, video_duration=100))
        app._get_user_approval_with_feedback = Mock(return_value=False)
        app._print_manual_prompt = Mock()
        return app

    def test_adaptive_downgrade_to_low_cannot_auto_submit(self):
        app = self.app(confidence=.50, adjustment=-.15)
        app.process_video("test-url")
        app.sponsorblock.submit_segment.assert_not_called()
        app._get_user_approval_with_feedback.assert_called_once()

    def test_positive_adaptive_bias_cannot_promote_medium(self):
        app = self.app(confidence=.79, adjustment=.10)
        app.process_video("test-url")
        app.sponsorblock.submit_segment.assert_not_called()
        app._get_user_approval_with_feedback.assert_called_once()

    def test_forced_review_survives_high_confidence_and_parallel_workers(self):
        app = self.app(workers=2)
        app.find_intro_in_video.return_value.force_review = True
        self.assertEqual(app.process_video("test-url")[1], "needs_review")
        app.sponsorblock.submit_segment.assert_not_called()
        app._get_user_approval_with_feedback.assert_not_called()

    def test_learned_boundary_shift_requires_review(self):
        app = self.app()
        app.adaptive_engine.apply_corrections.return_value.corrected_start = .8
        app.process_video("test-url")
        app.sponsorblock.submit_segment.assert_not_called()
        app._get_user_approval_with_feedback.assert_called_once()

    def test_invalid_learned_boundaries_never_submitted(self):
        app = self.app()
        app.adaptive_engine.apply_corrections.return_value.corrected_end = -1
        self.assertEqual(app.process_video("test-url")[1], "needs_review")
        app.sponsorblock.submit_segment.assert_not_called()

    def test_analysis_failure_is_retryable_not_no_intro(self):
        app = self.app()
        app.find_intro_in_video.side_effect = AudioAnalysisError("bad decode")
        self.assertEqual(app.process_video("test-url")[1], "failed")
        app.db.record.assert_called_once_with("video-id", "error_analysis")
        app.sponsorblock.submit_segment.assert_not_called()

    def test_no_intro_is_not_a_batch_failure(self):
        app = self.app()
        app.find_intro_in_video.return_value = None
        self.assertEqual(app.process_video("test-url")[1], "no_intro")
        app.sponsorblock.submit_segment.assert_not_called()

    def test_dry_run_does_not_change_database_or_submit(self):
        for detected in [IntroSegment(0, 10, .95), None]:
            app = self.app()
            app.dry_run = True
            app.find_intro_in_video.return_value = detected
            app.process_video("test-url")
            app.db.record.assert_not_called()
            app.sponsorblock.submit_segment.assert_not_called()

    def test_previous_dry_run_does_not_block_real_processing(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = VideoDB(str(Path(tmp) / "test.db"))
            try:
                db.record("video", "dry_run")
                self.assertFalse(db.is_processed("video"))
            finally:
                db.close()

    def test_corrected_feedback_preserves_original_detected_times(self):
        app = self.app()
        segment = IntroSegment(1, 11, .9, video_duration=100)
        with patch("builtins.input", side_effect=["c", "2", "12"]):
            approved = IntroSkipperAutomator._get_user_approval_with_feedback(
                app, "url", "id", segment)
        self.assertTrue(approved)
        feedback = app.feedback_store.record_feedback.call_args.kwargs
        self.assertEqual((feedback["detected_start"], feedback["detected_end"]), (1, 11))
        self.assertEqual((feedback["correct_start"], feedback["correct_end"]), (2, 12))
        self.assertEqual((segment.start_time, segment.end_time), (2, 12))

    def test_high_confidence_verified_result_still_auto_submits(self):
        app = self.app()
        self.assertEqual(app.process_video("test-url")[1], "success")
        app.sponsorblock.submit_segment.assert_called_once()


if __name__ == "__main__":
    unittest.main()
