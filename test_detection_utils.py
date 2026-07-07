import unittest

from detection_utils import (
    select_best_candidate,
    select_start_candidate,
    resolve_ambiguity,
)


def make_verify(coverage_by_time):
    """Stub verify_fn returning canned coverage per start time."""
    calls = []

    def verify(t):
        calls.append(t)
        cov = coverage_by_time.get(round(t, 2))
        if cov is None:
            return None
        return {"coverage": cov, "mean_similarity": cov, "frames": 400}

    verify.calls = calls
    return verify


class CandidateSelectionTests(unittest.TestCase):
    def test_delayed_high_confidence_peak_can_override_weak_zero_bonus(self):
        candidates = [
            {"time": 0.0, "raw": 0.751, "weighted": 1.151, "frame": 0},
            {"time": 24.08, "raw": 0.991, "weighted": 1.111, "frame": 1037},
        ]

        selected = select_best_candidate(candidates, score_at_zero=0.751)

        self.assertEqual(selected["time"], 24.08)

    def test_zero_is_kept_when_delayed_peak_is_only_moderately_better(self):
        candidates = [
            {"time": 0.0, "raw": 0.810, "weighted": 1.210, "frame": 0},
            {"time": 35.29, "raw": 0.893, "weighted": 0.962, "frame": 1520},
        ]

        selected = select_best_candidate(candidates, score_at_zero=0.810)

        self.assertEqual(selected["time"], 0.0)

    def test_zero_is_kept_when_zero_evidence_is_strong(self):
        candidates = [
            {"time": 0.0, "raw": 0.866, "weighted": 1.266, "frame": 0},
            {"time": 54.59, "raw": 0.960, "weighted": 0.986, "frame": 2351},
        ]

        selected = select_best_candidate(candidates, score_at_zero=0.866)

        self.assertEqual(selected["time"], 0.0)

    def test_weighted_winner_is_unchanged_when_it_is_already_delayed(self):
        candidates = [
            {"time": 0.0, "raw": 0.420, "weighted": 0.820, "frame": 0},
            {"time": 12.03, "raw": 0.882, "weighted": 1.101, "frame": 518},
        ]

        selected = select_best_candidate(candidates, score_at_zero=0.420)

        self.assertEqual(selected["time"], 12.03)


class VerificationArbitrationTests(unittest.TestCase):
    """The delayed-intro failure mode: a mediocre 0s correlation wins on
    position weighting, but frame-wise verification shows the real intro
    is at the delayed position."""

    def _delayed_intro_candidates(self):
        # 0s scores 0.80 raw (harmonically similar music at video start) —
        # too strong for the raw-score heuristic override to fire — while the
        # true intro at 60s scores 0.94 raw but gets almost no position bonus.
        return [
            {"time": 0.0, "raw": 0.80, "weighted": 1.20, "frame": 0},
            {"time": 60.0, "raw": 0.94, "weighted": 0.96, "frame": 2584},
        ]

    def test_verification_switches_to_delayed_when_zero_collapses(self):
        verify = make_verify({0.0: 0.30, 60.0: 0.92})
        best = select_start_candidate(
            self._delayed_intro_candidates(), score_at_zero=0.72, verify_fn=verify,
        )
        self.assertEqual(best["time"], 60.0)

    def test_strongly_verified_zero_is_never_overridden(self):
        verify = make_verify({0.0: 0.90, 60.0: 0.95})
        best = select_start_candidate(
            self._delayed_intro_candidates(), score_at_zero=0.72, verify_fn=verify,
        )
        self.assertEqual(best["time"], 0.0)
        # Short-circuit: delayed position should not even be verified
        self.assertNotIn(60.0, verify.calls)

    def test_indecisive_verification_keeps_heuristic_winner(self):
        verify = make_verify({0.0: 0.55, 60.0: 0.65})  # lead < 0.20
        best = select_start_candidate(
            self._delayed_intro_candidates(), score_at_zero=0.72, verify_fn=verify,
        )
        self.assertEqual(best["time"], 0.0)

    def test_verification_failure_falls_back_to_heuristics(self):
        verify = make_verify({})  # always None
        best = select_start_candidate(
            self._delayed_intro_candidates(), score_at_zero=0.72, verify_fn=verify,
        )
        self.assertEqual(best["time"], 0.0)

    def test_no_verify_fn_behaves_like_select_best_candidate(self):
        candidates = self._delayed_intro_candidates()
        self.assertEqual(
            select_start_candidate(candidates, score_at_zero=0.72),
            select_best_candidate(candidates, score_at_zero=0.72),
        )

    def test_weak_delayed_peak_does_not_trigger_verification(self):
        candidates = [
            {"time": 0.0, "raw": 0.82, "weighted": 1.22, "frame": 0},
            {"time": 45.0, "raw": 0.55, "weighted": 0.59, "frame": 1938},
        ]
        verify = make_verify({0.0: 0.20, 45.0: 0.99})
        best = select_start_candidate(candidates, score_at_zero=0.82, verify_fn=verify)
        self.assertEqual(best["time"], 0.0)
        self.assertEqual(verify.calls, [])  # no I/O for the common case

    def test_heuristic_override_is_reverted_when_delayed_fails_verification(self):
        # select_best_candidate's raw-score override picks 24s, but the
        # audio evidence says the intro really is at 0s.
        candidates = [
            {"time": 0.0, "raw": 0.751, "weighted": 1.151, "frame": 0},
            {"time": 24.08, "raw": 0.991, "weighted": 1.111, "frame": 1037},
        ]
        verify = make_verify({0.0: 0.88, 24.08: 0.35})
        best = select_start_candidate(candidates, score_at_zero=0.751, verify_fn=verify)
        self.assertEqual(best["time"], 0.0)

    def test_heuristic_override_is_kept_when_delayed_verifies(self):
        candidates = [
            {"time": 0.0, "raw": 0.751, "weighted": 1.151, "frame": 0},
            {"time": 24.08, "raw": 0.991, "weighted": 1.111, "frame": 1037},
        ]
        verify = make_verify({0.0: 0.40, 24.08: 0.95})
        best = select_start_candidate(candidates, score_at_zero=0.751, verify_fn=verify)
        self.assertEqual(best["time"], 24.08)
        # Delayed verified fine on the first call — 0s never checked
        self.assertNotIn(0.0, verify.calls)


class AmbiguityResolutionTests(unittest.TestCase):
    def test_resolves_toward_delayed_match(self):
        verify = make_verify({0.0: 0.30, 40.0: 0.85})
        self.assertEqual(resolve_ambiguity(40.0, 0.0, verify), "best")

    def test_resolves_toward_zero(self):
        verify = make_verify({0.0: 0.90, 40.0: 0.50})
        self.assertEqual(resolve_ambiguity(40.0, 0.0, verify), "zero")

    def test_skips_when_both_verify_similarly(self):
        verify = make_verify({0.0: 0.75, 40.0: 0.80})
        self.assertEqual(resolve_ambiguity(40.0, 0.0, verify), "skip")

    def test_skips_when_verification_unavailable(self):
        verify = make_verify({})
        self.assertEqual(resolve_ambiguity(40.0, 0.0, verify), "skip")


if __name__ == "__main__":
    unittest.main()
