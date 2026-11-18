"""S9 contracts: exact prefix indices, padding, matched cohorts, and no-future fit."""
from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from sc2proj.sequential.data import IndexedSequences
from sc2proj.sequential.gru import GRUConfig
from sc2proj.sequential.prefix import (
    CheckpointSequenceDataset, LengthBucketBatchSampler,
    PackedGRUWinner, collate_checkpoints, compare_modes,
    fit_checkpoint_gru, predict_checkpoint_gru,
)
from sc2proj.sequential.progress import select_exact_checkpoints
from sc2proj.sequential.windows import build_sequence_index


def make_toy():
    rec = []
    for replay, label, part in (
        ("train-A", 0, "train"), ("train-B", 1, "train"),
        ("val-A", 0, "val"), ("val-B", 1, "val"),
        ("test-C", 0, "test"),
    ):
        for t in range(40):
            rec.append({
                "replay_id": replay, "time_sec": 15 * t, "p1_wins": label,
                "source_row": len(rec),
            })
    ix = build_sequence_index(pd.DataFrame(rec), ("feature_a", "feature_b"), 9, 1, 15)
    idx = IndexedSequences(ix.ordered_rows, ix.windows, ix.feature_names,
                           {"window_length": 9, "stride": 1})
    replay_names = idx.ordered_rows.replay_id
    window_names = idx.windows.replay_id
    split = SimpleNamespace(
        groups={
            "train": frozenset(("train-A", "train-B")),
            "val": frozenset(("val-A", "val-B")),
            "test": frozenset(("test-C",)),
        },
        window_positions={
            p: np.flatnonzero(window_names.isin(ids).to_numpy(dtype=bool))
            for p, ids in {
                "train": ("train-A", "train-B"),
                "val": ("val-A", "val-B"),
                "test": ("test-C",),
            }.items()
        },
    )
    values = np.ascontiguousarray(
        np.arange(len(idx.ordered_rows) * 2, dtype=np.float32).reshape(-1, 2) / 100,
        dtype=np.float32,
    )
    train, _ = select_exact_checkpoints(idx, split, partition="train")
    val, _ = select_exact_checkpoints(idx, split, partition="val")
    return idx, split, values, train, val


class S9PrefixTests(unittest.TestCase):
    def setUp(self):
        self.idx, self.split, self.values, self.train, self.val = make_toy()

    def test_same_checkpoint_pairs_different_context_lengths(self):
        short = CheckpointSequenceDataset(self.values, self.idx, self.val, "last9")
        long = CheckpointSequenceDataset(self.values, self.idx, self.val, "prefix")
        self.assertEqual(len(short), 8)
        self.assertEqual(len(long), 8)
        self.assertTrue(np.all(short.lengths == 9))
        self.assertEqual(long.lengths[:4].tolist(), [10, 20, 30, 36])
        self.assertEqual(short.rows.time_sec.tolist(), long.rows.time_sec.tolist())
        self.assertEqual(short.rows.p1_wins.tolist(), long.rows.p1_wins.tolist())
        for i in range(8):
            self.assertTrue(np.array_equal(
                short[i][0].numpy(), long[i][0].numpy()[-9:]
            ))

    def test_future_snapshot_values_not_exposed_at_25_percent(self):
        ds = CheckpointSequenceDataset(self.values, self.idx, self.val, "prefix")
        first = ds[0][0].clone()
        after_end = ds.ends[0]
        self.values[after_end:] = 12345.0
        self.assertTrue(torch.equal(ds[0][0], first))

    def test_collate_pads_and_packed_model_respects_lengths(self):
        ds = CheckpointSequenceDataset(self.values, self.idx, self.val, "prefix")
        sample = [ds[0], ds[1], ds[2]]
        x, lengths, labels, positions = collate_checkpoints(sample)
        self.assertEqual(tuple(x.shape), (3, 30, 2))
        self.assertEqual(lengths.tolist(), [10, 20, 30])
        self.assertTrue(torch.equal(x[0, 10:], torch.zeros(20, 2)))
        model = PackedGRUWinner(2, GRUConfig(hidden_size=8))
        model.eval()
        out = model(x, lengths)
        self.assertEqual(tuple(out.shape), (3,))
        # Changing only padding does not affect valid prefix predictions.
        x[0, 10:] = 1e6
        with torch.no_grad():
            other = model(x, lengths)
        self.assertTrue(torch.allclose(out, other, atol=1e-6))
        with self.assertRaisesRegex(ValueError, "lengths"):
            model(x, torch.tensor([40, 20, 30]))

    def test_full_train_epoch_samples_each_checkpoint_once(self):
        ds = CheckpointSequenceDataset(self.values, self.idx, self.train, "prefix")
        sampler = LengthBucketBatchSampler(ds.lengths, 3, seed=42, shuffle=True)
        seen = [i for batch in sampler for i in batch]
        self.assertEqual(sorted(seen), list(range(len(ds))))
        self.assertEqual(len(set(seen)), len(ds))

    def test_wrong_labels_and_unknown_mode_rejected(self):
        bad = self.val.copy()
        bad.loc[0, "p1_wins"] = 1 - bad.loc[0, "p1_wins"]
        with self.assertRaisesRegex(ValueError, "target mismatch"):
            CheckpointSequenceDataset(self.values, self.idx, bad, "prefix")
        with self.assertRaisesRegex(ValueError, "Invalid checkpoint sequence mode"):
            CheckpointSequenceDataset(self.values, self.idx, self.val, "future")

    def test_cpu_smoke_and_val_predictions_never_iterates_test(self):
        cfg = GRUConfig(hidden_size=8, batch_size=4, max_epochs=2, patience=2)
        for mode in ("last9", "prefix"):
            train = CheckpointSequenceDataset(self.values, self.idx, self.train, mode)
            valid = CheckpointSequenceDataset(self.values, self.idx, self.val, mode)
            model, report = fit_checkpoint_gru(
                train, valid, cfg, torch.device("cpu"), seed=42, smoke=True)
            self.assertEqual(report["status"], "S9_SMOKE_PASS")
            self.assertEqual(report["test_evaluation_policy"], "not_accessed_during_training")
            pred = predict_checkpoint_gru(model, valid, cfg, torch.device("cpu"))
            self.assertEqual(len(pred), 8)
            self.assertEqual(set(pred.columns), {
                "replay_id", "time_sec", "progress_percent",
                "y_true", "y_prob", "y_pred",
            })
            self.assertTrue(pred.y_prob.between(0.0, 1.0).all())

    def test_matched_paired_bootstrap_rejects_endpoint_mismatch(self):
        cfg = GRUConfig(hidden_size=8, batch_size=4, max_epochs=2, patience=2)
        preds = {}
        for mode in ("last9", "prefix"):
            valid = CheckpointSequenceDataset(self.values, self.idx, self.val, mode)
            model = PackedGRUWinner(2, cfg)
            preds[mode] = predict_checkpoint_gru(
                model, valid, cfg, torch.device("cpu"))
        matched = compare_modes(preds, reps=50, seed=81)
        self.assertEqual(set(matched), {"25", "50", "75", "90"})
        self.assertEqual(matched["25"]["n_replays"], 2)
        altered = preds["prefix"].copy()
        altered.loc[0, "time_sec"] += 15
        with self.assertRaisesRegex(ValueError, "mismatched paired checkpoint"):
            compare_modes({"last9": preds["last9"], "prefix": altered},
                          reps=10, seed=81)


if __name__ == "__main__":
    unittest.main()
