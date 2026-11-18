"""S6: GRU comparison validation and replay-bootstrap generalized from S5."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from sc2proj.sequential.comparison import (
    load_verified_gru, model_metrics, paired_replay_bootstrap,
)
from sc2proj.sequential.variants import V32_SHA256


class S6ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.labels = np.array([0, 0, 1, 1], dtype=np.int8)
        self.pred = pd.DataFrame({
            "replay_id": ["A", "A", "B", "B"],
            "time_sec": [120, 135, 120, 135],
            "y_true": self.labels,
            "y_prob": [0.1, 0.2, 0.7, 0.8],
            "y_pred": [0, 0, 1, 1],
        })
        self.expected = self.pred.rename(columns={"y_true": "p1_wins"}).loc[
            :, ["replay_id", "time_sec", "p1_wins"]].copy()
        self.index = SimpleNamespace(
            features=tuple(f"feature{i}" for i in range(73)),
            manifest={"source_zip_sha256": V32_SHA256, "window_length": 9},
        )
        self.pred.to_csv(self.folder / "validation_predictions.csv", index=False)
        m = model_metrics(self.labels, self.pred.y_prob.to_numpy(),
                          self.pred.y_pred.to_numpy())
        self.summary = {
            "status": "S6_VALIDATION_PASS", "seed": 42, "smoke": False,
            "window_length": 9, "feature_count": 73, "source_zip_sha256": V32_SHA256,
            "dataset": "v3_2_combatfix",
            "test_evaluation_policy": "not_accessed_during_training",
            "checkpoint_selection": "validation_bce_only",
            "validation_windows": 4,
            "validation_metrics": {"n_endpoints": 4, "n_replays": 2,
                                    "accuracy": m["accuracy"],
                                    "balanced_accuracy": m["balanced_accuracy"],
                                    "roc_auc": m["roc_auc"], "log_loss": m["log_loss"]},
        }
        self._write()

    def _write(self):
        (self.folder / "summary.json").write_text(json.dumps(self.summary),
                                                  encoding="utf-8")

    def test_accept_combatfix_and_reject_wrong_status(self):
        got = load_verified_gru(
            self.folder, 42, self.index, self.expected,
            dataset_variant="v3_2_combatfix", expected_status="S6_VALIDATION_PASS")
        self.assertEqual(len(got), 4)
        with self.assertRaisesRegex(ValueError, "status mismatch"):
            load_verified_gru(self.folder, 42, self.index, self.expected)

    def test_reject_mislabeled_variant(self):
        self.summary["dataset"] = "v3_1_fixed"
        self._write()
        with self.assertRaisesRegex(ValueError, "dataset variant mismatch"):
            load_verified_gru(self.folder, 42, self.index, self.expected,
                              dataset_variant="v3_2_combatfix",
                              expected_status="S6_VALIDATION_PASS")

    def test_bootstrap_v31_paired_contract(self):
        paired = self.expected.copy()
        paired["gru_prob"] = [0.1, 0.2, 0.7, 0.8]
        paired["gru_pred"] = [0, 0, 1, 1]
        paired["v31_prob"] = [0.6, 0.2, 0.4, 0.7]
        paired["v31_pred"] = [1, 0, 0, 1]
        a = paired_replay_bootstrap(paired, "v31", reps=50, seed=123)
        b = paired_replay_bootstrap(paired, "v31", reps=50, seed=123)
        self.assertEqual(a, b)
        self.assertAlmostEqual(a["accuracy_delta"], 0.5)
        self.assertLess(a["log_loss_delta"], 0)
        self.assertEqual(a["replay_groups"], 2)


if __name__ == "__main__":
    unittest.main()
