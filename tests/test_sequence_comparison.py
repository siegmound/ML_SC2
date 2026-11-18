"""S5: strict endpoint and label parity, paired replay-unit bootstrap, no test access."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pandas as pd

from sc2proj.sequential.baselines import archived_prediction_path
from sc2proj.sequential.comparison import (
    load_verified_gru, paired_replay_bootstrap, compare_one_seed, summarize_multiseed,
)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "gru"
        self.run.mkdir()
        records = []
        for replay, label in (("val-A", 0), ("val-B", 1)):
            for t in (120, 135, 150):
                records.append({"replay_id": replay, "time_sec": t, "p1_wins": label})
        self.expected = pd.DataFrame(records)
        self.index = SimpleNamespace(features=tuple("abc"), manifest={
            "window_length": 9, "source_zip_sha256": "correct-hash"}, windows=self.expected)
        self.split = SimpleNamespace(seed=42, window_positions={"val": np.arange(6)},
                                     groups={"val": frozenset(("val-A", "val-B"))})
        self.gru = self.expected.rename(columns={"p1_wins": "y_true"}).copy()
        self.gru["y_prob"] = [0.1, 0.2, 0.4, 0.7, 0.8, 0.9]
        self.gru["y_pred"] = (self.gru.y_prob >= 0.5).astype(int)
        self._save()

    def _save(self, *, summary_override=None):
        self.gru.to_csv(self.run / "validation_predictions.csv", index=False)
        summary = {"status": "S4_VALIDATION_PASS", "seed": 42, "smoke": False,
                   "window_length": 9, "feature_count": 3,
                   "source_zip_sha256": "correct-hash", "dataset": "v3_1_fixed",
                   "test_evaluation_policy": "not_accessed_during_training",
                   "checkpoint_selection": "validation_bce_only", "validation_windows": len(self.gru),
                   "config": {"hidden_size": 8, "batch_size": 2}, "selected_epoch": 1,
                   "validation_metrics": {
                       "n_endpoints": len(self.gru), "n_replays": int(self.gru.replay_id.nunique()),
                       "accuracy": 1.0, "balanced_accuracy": 1.0,
                       "roc_auc": 1.0,
                       "log_loss": float(-np.log([0.9, 0.8, 0.6, 0.7, 0.8, 0.9]).mean()),
                   }}
        summary.update(summary_override or {})
        (self.run / "summary.json").write_text(json.dumps(summary))

    def test_valid_gru_endpoint_schema_and_metrics(self):
        got = load_verified_gru(self.run, 42, self.index, self.expected)
        self.assertEqual(len(got), 6)
        self.assertEqual(got.y_pred.tolist(), [0, 0, 0, 1, 1, 1])

    def test_missing_extra_duplicate_endpoint_fails_closed(self):
        self.gru = self.gru.iloc[:-1]
        self._save()
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            load_verified_gru(self.run, 42, self.index, self.expected)
        self.gru = pd.concat([self.gru, self.gru.iloc[[0]]], ignore_index=True)
        self._save()
        with self.assertRaisesRegex(ValueError, "Duplicate GRU endpoint"):
            load_verified_gru(self.run, 42, self.index, self.expected)

    def test_bad_label_probability_and_prediction_rejected(self):
        self.gru.loc[0, "y_true"] = 1
        self._save()
        with self.assertRaisesRegex(ValueError, "labels disagree"):
            load_verified_gru(self.run, 42, self.index, self.expected)
        self.gru.loc[0, "y_true"] = 0
        self.gru.loc[0, "y_prob"] = 2.0
        self._save()
        with self.assertRaisesRegex(ValueError, "probabilities"):
            load_verified_gru(self.run, 42, self.index, self.expected)
        self.gru.loc[0, "y_prob"] = 0.1
        self.gru.loc[0, "y_pred"] = 1
        self._save()
        with self.assertRaisesRegex(ValueError, "threshold"):
            load_verified_gru(self.run, 42, self.index, self.expected)

    def test_wrong_dataset_or_heldout_policy_rejected(self):
        self._save(summary_override={"source_zip_sha256": "different-hash"})
        with self.assertRaisesRegex(ValueError, "source_zip_sha256"):
            load_verified_gru(self.run, 42, self.index, self.expected)
        self._save(summary_override={"test_evaluation_policy": "test_used"})
        with self.assertRaisesRegex(ValueError, "test_evaluation_policy"):
            load_verified_gru(self.run, 42, self.index, self.expected)

    def test_replay_bootstrap_reproducibility_and_exact_deltas(self):
        data = self.expected.copy()
        data["gru_pred"] = [0, 0, 0, 1, 1, 1]
        data["xgb_pred"] = [1, 0, 1, 0, 1, 0]
        data["gru_prob"] = [0.1, 0.2, 0.4, 0.7, 0.8, 0.9]
        data["xgb_prob"] = [0.8, 0.2, 0.6, 0.4, 0.8, 0.4]
        a = paired_replay_bootstrap(data, "xgb", reps=70, seed=4)
        b = paired_replay_bootstrap(data, "xgb", reps=70, seed=4)
        self.assertEqual(a, b)
        self.assertEqual(a["replay_groups"], 2)
        self.assertAlmostEqual(a["accuracy_delta"], 4 / 6)
        self.assertGreater(a["accuracy_delta_ci95"][0], 0)
        self.assertLess(a["log_loss_delta"], 0)
        self.assertIsNone(paired_replay_bootstrap(data, "xgb", reps=0)["accuracy_delta_ci95"])

    def test_baseline_and_gru_combined_integration(self):
        for model in ("rf", "xgb", "mlp"):
            path = archived_prediction_path(self.root / "results", model, 42, "val")
            path.parent.mkdir(parents=True)
            hist = self.gru.copy()
            hist["y_prob"] = [0.2, 0.3, 0.45, 0.55, 0.65, 0.8]
            hist["y_pred"] = (hist.y_prob >= 0.5).astype(int)
            hist.to_csv(path, index=False)
        # Only 6 synthetic endpoints, whereas complete FAIR S5 requires 112312.
        with self.assertRaisesRegex(ValueError, "S3 endpoint count"):
            compare_one_seed(self.index, self.split, self.root / "results", self.run, reps=0)

    def test_complete_synthetic_seed_comparison(self):
        for model in ("rf", "xgb", "mlp"):
            path = archived_prediction_path(self.root / "results", model, 42, "val")
            path.parent.mkdir(parents=True)
            hist = self.gru.copy()
            hist["y_prob"] = [0.2, 0.3, 0.45, 0.55, 0.65, 0.8]
            hist["y_pred"] = (hist.y_prob >= 0.5).astype(int)
            hist.to_csv(path, index=False)
        with patch("sc2proj.sequential.comparison.VALIDATION_ENDPOINTS", {42: 6}):
            result, combined = compare_one_seed(self.index, self.split, self.root / "results", self.run,
                                                reps=15, bootstrap_seed=7)
        self.assertEqual(result["endpoints"], 6)
        self.assertEqual(len(combined), 6)
        self.assertIn("gru", result["models"])
        self.assertEqual(result["gru_selected_epoch"], 1)
        self.assertEqual(result["gru_minus_baseline"]["rf"]["paired_bootstrap"]["replay_groups"], 2)

    def test_cross_seed_config_mismatch_rejected(self):
        base = {"source_zip_sha256": "hash", "gru_config": {"hidden_size": 64},
                "models": {}, "gru_minus_baseline": {}}
        entries = [{**base, "seed": seed} for seed in (42, 43, 44)]
        entries[2] = {**entries[2], "gru_config": {"hidden_size": 128}}
        with self.assertRaisesRegex(ValueError, "hyperparameters differ"):
            summarize_multiseed(entries, reps=0, bootstrap_seed=2026)

    def test_summary_requires_three_ordered_seeds(self):
        with self.assertRaisesRegex(ValueError, "42, 43, 44"):
            summarize_multiseed([], reps=0, bootstrap_seed=2026)


if __name__ == "__main__":
    unittest.main()
