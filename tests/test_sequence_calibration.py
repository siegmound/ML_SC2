"""S10 probability audit tests: Brier, calibration bins, and paired uncertainty."""
from __future__ import annotations

import unittest

import numpy as np

from sc2proj.sequential.calibration import (
    calibration_diagnostics,
    paired_brier_bootstrap,
    reliability_bins,
    validate_binary_probabilities,
)


class CalibrationDiagnosticsTests(unittest.TestCase):
    def test_perfect_binary_probabilities_are_exactly_reliable(self):
        labels = np.array([0, 1, 0, 1])
        probs = np.array([0., 1., 0., 1.])
        d = calibration_diagnostics(labels, probs, bins=10)
        self.assertEqual(d["expected_calibration_error"], 0.0)
        self.assertEqual(d["brier_score"], 0.0)
        self.assertEqual(d["calibration_bias_p1"], 0.0)
        self.assertEqual(d["reliability_bins"][-1]["n"], 2)
        self.assertAlmostEqual(d["brier_skill_vs_climatology"], 1.0)

    def test_constant_half_predictions_are_calibrated_at_balanced_prevalence(self):
        d = calibration_diagnostics([0, 1, 0, 1], [0.5] * 4, bins=10)
        self.assertAlmostEqual(d["expected_calibration_error"], 0.0)
        self.assertAlmostEqual(d["brier_score"], 0.25)
        self.assertAlmostEqual(d["log_loss"], float(np.log(2)), places=7)
        self.assertAlmostEqual(d["brier_skill_vs_climatology"], 0.0)
        self.assertAlmostEqual(d["probability_variance"], 0.0)

    def test_reliability_bins_include_endpoints_and_empty_bins(self):
        b = reliability_bins([0, 1], [0, 1], bins=5)
        self.assertEqual([r["n"] for r in b], [1, 0, 0, 0, 1])
        self.assertIsNone(b[1]["observed_p1"])
        self.assertEqual(b[0]["lower"], 0.0)
        self.assertEqual(b[-1]["upper"], 1.0)

    def test_invalid_labels_probabilities_and_bin_count(self):
        for y, p in (
            ([], []), ([0], [0.2, 0.3]), ([1, 2], [0.2, 0.8]),
            ([1], [float("nan")]), ([1], [1.01]), ([0], [-0.01]),
        ):
            with self.subTest(y=y, p=p), self.assertRaises(ValueError):
                validate_binary_probabilities(y, p)
        for n in (-1, 0, 1, 101):
            with self.subTest(n=n), self.assertRaises(ValueError):
                reliability_bins([0, 1], [0.2, 0.8], bins=n)

    def test_paired_brier_bootstrap_zero_delta_for_identical_models(self):
        y = np.array([0, 1, 1, 0] * 10)
        p = np.array([0.3, 0.7, 0.8, 0.2] * 10)
        d = paired_brier_bootstrap(y, p, p, reps=120, seed=7)
        self.assertAlmostEqual(d["delta_brier_prefix_minus_last9"], 0.0)
        self.assertEqual(d["delta_brier_ci95"], [0.0, 0.0])
        self.assertEqual(d["replay_groups"], 40)

    def test_paired_brier_gain_is_negative_with_better_probabilities(self):
        y = np.array([0, 1] * 30)
        base = np.full(len(y), 0.5)
        good = np.where(y == 1, 0.8, 0.2)
        d = paired_brier_bootstrap(y, base, good, reps=100, seed=11)
        self.assertAlmostEqual(d["delta_brier_prefix_minus_last9"], -0.21)
        self.assertTrue(all(x < 0 for x in d["delta_brier_ci95"]))
        self.assertIsNone(paired_brier_bootstrap(y, base, good, reps=0)["delta_brier_ci95"])


if __name__ == "__main__":
    unittest.main()
