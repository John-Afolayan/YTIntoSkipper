import unittest

from detection_utils import select_best_candidate


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


if __name__ == "__main__":
    unittest.main()
