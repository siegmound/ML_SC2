"""S4 GRU contract tests: shape, validation-only checkpointing, endpoint parity."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from sc2proj.sequential.data import (
    IndexedSequences, PartitionedSequences, PreparedSequences, SequenceWindowDataset,
)
from sc2proj.sequential.gru import (
    GRUConfig, GRUWinner, fit_gru, validation_predictions,
)


class _ForbiddenTestDataset(SequenceWindowDataset):
    def __getitem__(self, i):
        raise AssertionError("The test partition must not be iterated during S4 training")


def make_synthetic():
    rng = np.random.default_rng(8)
    features = np.ascontiguousarray(rng.standard_normal((18, 3)).astype(np.float32))
    ordered_rows = []
    windows = []
    groups = {"train": frozenset({"train-A"}),
              "val": frozenset({"val-B"}),
              "test": frozenset({"test-C"})}
    masks = {}
    positions = {}
    datasets = {}
    for part_i, part in enumerate(("train", "val", "test")):
        replay = next(iter(groups[part]))
        start = part_i * 6
        masks[part] = np.zeros(18, dtype=bool)
        masks[part][start:start + 6] = True
        positions[part] = np.arange(part_i * 4, (part_i + 1) * 4, dtype=np.int64)
        for i in range(6):
            ordered_rows.append({"replay_id": replay, "time_sec": i * 15,
                                 "p1_wins": part_i % 2, "source_row": start + i})
        for i in range(4):
            windows.append({"start_row": start + i, "stop_row": start + i + 3,
                            "replay_id": replay, "time_sec": (i + 2) * 15,
                            "p1_wins": part_i % 2})
    ordered = pd.DataFrame(ordered_rows)
    win = pd.DataFrame(windows)
    partition = PartitionedSequences(groups, masks, positions, seed=42)
    for part in ("train", "val", "test"):
        cls = _ForbiddenTestDataset if part == "test" else SequenceWindowDataset
        datasets[part] = cls(features, win, positions[part])
    prepared = PreparedSequences(features, StandardScaler().fit(features[:6]),
                                 partition, datasets)
    index = IndexedSequences(ordered, win, ("f0", "f1", "f2"),
                             {"window_length": 3, "source_zip_sha256": "synthetic"})
    return index, partition, prepared


class GRUS4Tests(unittest.TestCase):
    def test_gru_logits_shape_and_input_validation(self):
        model = GRUWinner(feature_count=3, hidden_size=8, num_layers=2)
        self.assertEqual(tuple(model(torch.zeros(4, 9, 3)).shape), (4,))
        self.assertEqual(tuple(model(torch.zeros(1, 3, 3)).shape), (1,))
        with self.assertRaises(ValueError):
            model(torch.zeros(4, 3))
        with self.assertRaises(ValueError):
            model(torch.zeros(4, 9, 2))

    def test_invalid_configs_rejected(self):
        with self.assertRaises(ValueError):
            GRUConfig(patience=0)
        with self.assertRaises(ValueError):
            GRUConfig(dropout=1.0)
        with self.assertRaises(ValueError):
            GRUConfig(batch_size=-1)
        with self.assertRaises(ValueError):
            GRUWinner(feature_count=0)

    def test_smoke_uses_train_and_val_but_never_test(self):
        _, _, prepared = make_synthetic()
        config = GRUConfig(hidden_size=8, batch_size=2, max_epochs=2, patience=2)
        model, report = fit_gru(prepared, config, torch.device("cpu"), smoke=True)
        self.assertEqual(report["status"], "S4_SMOKE_PASS")
        self.assertEqual(len(report["history"]), 2)
        self.assertIn(report["selected_epoch"], (1, 2))
        self.assertTrue(np.isfinite(report["best_validation_log_loss"]))
        self.assertEqual(report["test_evaluation_policy"], "not_accessed_during_training")
        self.assertIsInstance(model, GRUWinner)

    def test_validation_predictions_match_all_s1_s3_endpoints(self):
        index, split, prepared = make_synthetic()
        config = GRUConfig(hidden_size=8, batch_size=2, max_epochs=2, patience=2)
        model, _ = fit_gru(prepared, config, torch.device("cpu"), smoke=True)
        frame, metrics = validation_predictions(
            model, prepared, index, split, torch.device("cpu"), batch_size=2
        )
        self.assertEqual(list(frame.columns),
                         ["replay_id", "time_sec", "y_true", "y_prob", "y_pred"])
        self.assertEqual(frame["time_sec"].tolist(), [30, 45, 60, 75])
        self.assertEqual(metrics["n_endpoints"], 4)
        self.assertEqual(metrics["n_replays"], 1)
        self.assertTrue(frame["y_prob"].between(0.0, 1.0).all())
        self.assertIsNone(metrics["roc_auc"])

    def test_mismatched_endpoint_is_rejected(self):
        index, split, prepared = make_synthetic()
        model = GRUWinner(feature_count=3, hidden_size=8)
        corrupted = index.windows.copy()
        corrupted.loc[4, "time_sec"] = 90
        index = IndexedSequences(index.ordered_rows, corrupted, index.features, index.manifest)
        with self.assertRaisesRegex(ValueError, "do not match"):
            validation_predictions(model, prepared, index, split, torch.device("cpu"), 2)


if __name__ == "__main__":
    unittest.main()
