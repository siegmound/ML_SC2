"""S11 contracts: exact validation checkpoint alignment and no silent subset drift."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from sc2proj.sequential.stage_compare import (
    align_validation_model, validate_stage_cohort,
)


def cohort() -> pd.DataFrame:
    rows = []
    for replay, label in (("A", 0), ("B", 1)):
        for i, p in enumerate((25, 50, 75, 90)):
            rows.append({
                "replay_id": replay, "time_sec": (i + 9) * 15,
                "progress_percent": p, "p1_wins": label,
                "v31_original64_prob": 0.8 if label else 0.2,
                "v31_original64_pred": label,
            })
    return pd.DataFrame(rows)


def predictions():
    a = cohort()
    return pd.DataFrame({
        "replay_id": a.replay_id,
        "time_sec": a.time_sec,
        "y_true": a.p1_wins,
        "y_prob": a.v31_original64_prob,
        "y_pred": a.v31_original64_pred,
    })


class StageModelComparisonTests(unittest.TestCase):
    def test_exact_cohort_and_reference_columns_preserved(self):
        stage = cohort()
        self.assertEqual(validate_stage_cohort(stage), 2)
        matched = align_validation_model(stage, predictions(), "s8_gru", require_exact=True)
        self.assertEqual(len(matched), 8)
        self.assertIn("v31_original64_prob", matched.columns)
        self.assertTrue(np.array_equal(matched.p1_wins, matched.y_true))

    def test_archived_extra_endpoints_can_match_but_not_exact_gru(self):
        pred = predictions()
        pred.loc[len(pred)] = ["A", 999, 0, 0.5, 1]
        self.assertEqual(len(align_validation_model(cohort(), pred, "rf", 
                              require_exact=False, enforce_half_threshold=False)), 8)
        with self.assertRaisesRegex(ValueError, "expected only selected"):
            align_validation_model(cohort(), pred, "s9", require_exact=True)

    def test_missing_endpoint_or_incorrect_winner_rejected(self):
        pred = predictions()
        with self.assertRaisesRegex(ValueError, "missing frozen"):
            align_validation_model(cohort(), pred.iloc[1:], "xgb")
        bad = pred.copy()
        bad.loc[0, "y_true"] = 1
        with self.assertRaisesRegex(ValueError, "winner label mismatch"):
            align_validation_model(cohort(), bad, "xgb")

    def test_duplicate_endpoints_and_incomplete_progress_rejected(self):
        pred = pd.concat([predictions(), predictions().iloc[[0]]])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            align_validation_model(cohort(), pred, "mlp")
        with self.assertRaisesRegex(ValueError, "same replay IDs at every stage"):
            validate_stage_cohort(cohort().iloc[:-1])

    def test_nonfinite_and_threshold_guards(self):
        bad = predictions()
        bad.loc[0, "y_prob"] = float("nan")
        with self.assertRaisesRegex(ValueError, "missing"):
            align_validation_model(cohort(), bad, "xgb")
        bad = predictions()
        bad.loc[0, "y_prob"] = 1.1
        with self.assertRaisesRegex(ValueError, "invalid probabilities"):
            align_validation_model(cohort(), bad, "rf")
        bad = predictions()
        bad.loc[0, "y_pred"] = 1
        with self.assertRaisesRegex(ValueError, "0.5 decision threshold"):
            align_validation_model(cohort(), bad, "s8")
        # Archived classifier may legitimately have a previously chosen threshold.
        self.assertEqual(len(align_validation_model(
            cohort(), bad, "rf", enforce_half_threshold=False,
        )), 8)


if __name__ == "__main__":
    unittest.main()
