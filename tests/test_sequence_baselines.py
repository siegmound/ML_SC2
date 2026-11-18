"""S3 tests: strict replay/time joins and frozen prediction integrity."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from sc2proj.sequential.baselines import (
    MODELS, archived_prediction_path, compare_archived_baselines,
)


class EndpointBaselineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        groups = {p: frozenset([p + "-A"]) for p in ("train", "val", "test")}
        records = []
        for part in ("train", "val", "test"):
            for i, t in enumerate((165, 180, 195, 210)):
                records.append({"replay_id": part + "-A", "time_sec": t, "p1_wins": i % 2})
        self.index = SimpleNamespace(windows=pd.DataFrame(records), manifest={"window_length": 9})
        self.partition = SimpleNamespace(
            groups=groups, seed=42,
            window_positions={p: np.array([i for i, row in enumerate(records)
                                           if row["replay_id"] == p + "-A"])
                              for p in ("train", "val", "test")},
        )
        for model in MODELS:
            for part in ("val", "test"):
                path = archived_prediction_path(self.root, model, 42, part)
                path.parent.mkdir(parents=True, exist_ok=True)
                observations = [
                    {"replay_id": part + "-A", "time_sec": float(t),
                     "y_true": (i - 2) % 2 if i >= 2 else 0,
                     "y_prob": 0.1 if i % 2 == 0 else 0.8,
                     "y_pred": i % 2}
                    for i, t in enumerate((135, 150, 165, 180, 195, 210))
                ]
                pd.DataFrame(observations).to_csv(path, index=False)

    def test_all_models_match_exact_sequence_endpoints(self):
        report, matched = compare_archived_baselines(self.index, self.partition, "val", self.root)
        self.assertEqual(report["status"], "S3_MATCHED_ENDPOINTS_PASS")
        self.assertEqual(report["endpoints"], 4)
        self.assertEqual(len(matched), 4)
        self.assertEqual(list(matched.columns), [
            "replay_id", "time_sec", "p1_wins",
            "rf_prob", "rf_pred", "xgb_prob", "xgb_pred", "mlp_prob", "mlp_pred",
        ])
        self.assertEqual(report["models"]["rf"]["accuracy"], 1.0)

    def test_common_endpoints_of_different_windows(self):
        longer = SimpleNamespace(
            windows=self.index.windows.iloc[[2, 3, 6, 7, 10, 11]].copy(),
            manifest={"window_length": 17},
        )
        report, matched = compare_archived_baselines(
            self.index, self.partition, "val", self.root, common_indexes=[longer],
        )
        self.assertEqual(len(matched), 2)
        self.assertEqual(report["common_window_lengths"], [9, 17])

    def test_missing_historical_endpoint_rejected(self):
        path = archived_prediction_path(self.root, "rf", 42, "val")
        frame = pd.read_csv(path)
        frame.loc[frame.time_sec != 165].to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "missing archived predictions"):
            compare_archived_baselines(self.index, self.partition, "val", self.root)

    def test_inconsistent_labels_rejected(self):
        path = archived_prediction_path(self.root, "rf", 42, "val")
        frame = pd.read_csv(path)
        frame.loc[frame.time_sec == 165, "y_true"] = 1
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "labels disagree"):
            compare_archived_baselines(self.index, self.partition, "val", self.root)

    def test_duplicate_wrong_split_and_nonintegral_time_rejected(self):
        path = archived_prediction_path(self.root, "rf", 42, "val")
        frame = pd.read_csv(path)
        duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        duplicate.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            compare_archived_baselines(self.index, self.partition, "val", self.root)
        frame.loc[0, "replay_id"] = "test-A"
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "outside selected split"):
            compare_archived_baselines(self.index, self.partition, "val", self.root)
        frame.loc[0, "replay_id"] = "val-A"
        frame.loc[0, "time_sec"] = 135.5
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "timestamp"):
            compare_archived_baselines(self.index, self.partition, "val", self.root)


if __name__ == "__main__":
    unittest.main()
