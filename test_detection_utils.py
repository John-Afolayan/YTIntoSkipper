import unittest
from detection_utils import select_start_candidate


def candidate(time, raw=0.9):
    return {"time": time, "raw": raw, "weighted": raw + (0.4 if time == 0 else 0)}


def evidence(score=0.95, verified=True):
    return {"verified": verified, "quality": score}


class CandidateSelectionTests(unittest.TestCase):
    def test_delayed_intro_beats_false_zero_even_with_strong_chroma(self):
        candidates = [candidate(0, 0.96), candidate(60, 0.90)]
        self.assertEqual(select_start_candidate(candidates, verify_fn=lambda t:
            evidence(0.99, t == 60))["time"], 60)

    def test_every_candidate_is_verified_including_single_peak(self):
        for candidates in [[candidate(0)], [candidate(0), candidate(12), candidate(40)]]:
            calls = []
            def verify(t):
                calls.append(t)
                return evidence()
            select_start_candidate(candidates, verify_fn=verify)
            self.assertEqual(calls, [c["time"] for c in candidates])

    def test_no_verifier_or_missing_evidence_rejects(self):
        self.assertIsNone(select_start_candidate([candidate(0)]))
        self.assertIsNone(select_start_candidate([candidate(0)], verify_fn=lambda t: None))

    def test_chroma_coverage_alone_cannot_authorize_a_match(self):
        self.assertIsNone(select_start_candidate([candidate(0)], verify_fn=lambda t:
            {"coverage": 1.0, "mean_similarity": 1.0}))

    def test_all_failed_verification_rejects(self):
        self.assertIsNone(select_start_candidate([candidate(0), candidate(60)],
            verify_fn=lambda t: evidence(0.99, False)))

    def test_third_peak_can_win(self):
        cs = [candidate(0, .99), candidate(20, .98), candidate(60, .85)]
        self.assertEqual(select_start_candidate(cs, verify_fn=lambda t:
            evidence(.9, t == 60))["time"], 60)

    def test_genuine_repeats_prefer_first_occurrence(self):
        self.assertEqual(select_start_candidate([candidate(40), candidate(0)],
            verify_fn=lambda t: evidence(.98 if t else .86))["time"], 0)

    def test_quality_beats_position_bonus(self):
        self.assertEqual(select_start_candidate([candidate(0), candidate(12)],
            verify_fn=lambda t: evidence(.95 if t else .65))["time"], 12)

    def test_subsecond_delay_is_preserved(self):
        self.assertEqual(select_start_candidate([candidate(.8)],
            verify_fn=lambda t: evidence())["time"], .8)

    def test_nonfinite_inputs_rejected(self):
        for c in [candidate(float("nan")), candidate(0, float("inf")), candidate(-1)]:
            self.assertIsNone(select_start_candidate([c], verify_fn=lambda t: evidence()))
        self.assertIsNone(select_start_candidate([candidate(0)],
            verify_fn=lambda t: evidence(float("nan"))))


if __name__ == "__main__":
    unittest.main()
