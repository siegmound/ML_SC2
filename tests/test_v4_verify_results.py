"""Post-training v4 synthetic metric/lineage verification tests."""
from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    import sc2reader
except ImportError:
    sys.modules["sc2reader"] = types.ModuleType("sc2reader")


def auditor():
    spec = importlib.util.spec_from_file_location(
        "v4_results_audit", ROOT / "scripts/92_v4_verify_results.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class V4ResultsVerificationTests(unittest.TestCase):
    def setUp(self):
        self.a = auditor()
        self.expected = pd.DataFrame([
            ("a", 135, 0), ("a", 150, 0),
            ("b", 135, 1), ("b", 150, 1),
        ], columns=["replay_id", "time_sec", "p1_wins"])
        self.predictions = pd.DataFrame([
            ("a", 135, 0, .20, 0), ("a", 150, 0, .25, 0),
            ("b", 135, 1, .90, 1), ("b", 150, 1, .80, 1),
        ], columns=["replay_id", "time_sec", "y_true", "y_prob", "y_pred"])

    def test_verifier_confirms_predictions_and_both_evaluation_units(self):
        metrics = self.a.metric_values(
            self.predictions.y_true.to_numpy(),
            self.predictions.y_prob.to_numpy(),
        )
        report = self.a.verify_predictions(
            self.predictions, self.expected, metrics
        )
        self.assertEqual(report["n_test_snapshots"], 4)
        self.assertEqual(report["n_test_replays"], 2)
        self.assertEqual(report["snapshot_metrics"]["accuracy"], 1.0)
        self.assertEqual(report["replay_mean_probability_metrics"]["accuracy"], 1.0)

    def test_rejects_wrong_test_outcomes_and_overlap_duplicates(self):
        bad = self.predictions.copy()
        bad.loc[0, "y_true"] = 1
        with self.assertRaisesRegex(ValueError, "Test label disagrees"):
            self.a.verify_predictions(bad, self.expected, {})
        bad = self.predictions.copy()
        bad.loc[1, "time_sec"] = 135
        with self.assertRaisesRegex(ValueError, "Duplicate test prediction"):
            self.a.verify_predictions(bad, self.expected, {})

    def test_rejects_nonprobabilities_threshold_inconsistency_and_metric_fraud(self):
        bad = self.predictions.copy()
        bad.loc[0, "y_prob"] = 1.2
        with self.assertRaisesRegex(ValueError, "out of"):
            self.a.verify_predictions(bad, self.expected, {})
        bad = self.predictions.copy()
        bad.loc[0, "y_pred"] = 1
        with self.assertRaisesRegex(ValueError, "threshold"):
            self.a.verify_predictions(bad, self.expected, {})
        with self.assertRaisesRegex(ValueError, "Recomputed accuracy"):
            self.a.verify_predictions(
                self.predictions, self.expected, {"accuracy": .5}
            )


if __name__ == "__main__":
    unittest.main()
