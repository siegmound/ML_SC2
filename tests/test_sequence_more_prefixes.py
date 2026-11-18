"""S12: train-only extra-cutoff selection and matched-variable-length GRU contracts."""
from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch

from test_sequence_prefix import make_toy
from sc2proj.sequential.extra_prefixes import expand_train_checkpoint_examples
from sc2proj.sequential.prefix import (
    CheckpointSequenceDataset, fit_checkpoint_gru, predict_checkpoint_gru, compare_modes,
)
from sc2proj.sequential.gru import GRUConfig


class MorePrefixesTests(unittest.TestCase):
    def setUp(self):
        self.index, self.partition, self.values, self.train, self.val = make_toy()

    def test_exact_original_anchor_replay_cohort_and_new_cutoffs(self):
        expanded, coverage = expand_train_checkpoint_examples(
            self.index, self.partition, self.train, extras_per_replay=12,
        )
        self.assertEqual(coverage["training_replays"], 2)
        self.assertEqual(coverage["s9_anchor_examples"], 8)
        self.assertEqual(coverage["extra_examples"], 24)
        self.assertEqual(coverage["expanded_train_examples"], 32)
        self.assertEqual(coverage["extras_per_replay_min"], 12)
        self.assertEqual(coverage["extras_per_replay_max"], 12)
        self.assertEqual(set(expanded.replay_id), {"train-A", "train-B"})
        self.assertTrue(expanded.duplicated(["replay_id", "time_sec"]).eq(False).all())
        self.assertEqual((expanded.progress_percent == -1).sum(), 24)
        self.assertEqual(set(expanded.loc[expanded.progress_percent != -1, "progress_percent"]),
                         {25, 50, 75, 90})
        self.assertEqual(len(expanded.loc[expanded.progress_percent != -1]), len(self.train))
        # Original S9 validation cohort is untouched.
        self.assertEqual(len(self.val), 8)
        self.assertEqual(set(self.val.replay_id), {"val-A", "val-B"})

    def test_last9_and_prefix_receive_identical_new_endpoint_labels(self):
        expanded, _ = expand_train_checkpoint_examples(
            self.index, self.partition, self.train, extras_per_replay=12,
        )
        a = CheckpointSequenceDataset(self.values, self.index, expanded, "last9")
        b = CheckpointSequenceDataset(self.values, self.index, expanded, "prefix")
        self.assertEqual(len(a), 32)
        self.assertEqual(len(b), 32)
        self.assertTrue(np.all(a.lengths == 9))
        self.assertGreater(b.lengths.max(), 9)
        self.assertEqual(a.rows.time_sec.tolist(), b.rows.time_sec.tolist())
        self.assertEqual(a.rows.p1_wins.tolist(), b.rows.p1_wins.tolist())
        for i in range(len(a)):
            self.assertTrue(torch.equal(a[i][0], b[i][0][-9:]))

    def test_zero_extras_reconstruct_original_four_checkpoint_training(self):
        expanded, coverage = expand_train_checkpoint_examples(
            self.index, self.partition, self.train, extras_per_replay=0,
        )
        self.assertEqual(coverage["extra_examples"], 0)
        self.assertEqual(coverage["expanded_train_examples"], len(self.train))
        self.assertTrue(expanded.sort_values(
            ["replay_id", "progress_percent"]
        ).reset_index(drop=True).equals(
            self.train[list(expanded.columns)].sort_values(
                ["replay_id", "progress_percent"]
            ).reset_index(drop=True)
        ))

    def test_selection_deterministic_and_bounded(self):
        a, _ = expand_train_checkpoint_examples(
            self.index, self.partition, self.train, extras_per_replay=8,
        )
        b, _ = expand_train_checkpoint_examples(
            self.index, self.partition, self.train, extras_per_replay=8,
        )
        self.assertTrue(a.equals(b))
        self.assertEqual(len(a), 8 + 16)
        self.assertTrue(a.observed_snapshot_count.ge(9).all())
        self.assertTrue(a.observed_snapshot_count.le(a.replay_snapshot_count).all())
        self.assertTrue(a.observed_fraction.between(0, 1).all())

    def test_rejects_heldout_group_overlap_and_invalid_anchor(self):
        bad_split = SimpleNamespace(
            groups={**self.partition.groups, "train": frozenset(
                (*self.partition.groups["train"], "val-A")
            )},
            window_positions=self.partition.window_positions,
        )
        with self.assertRaisesRegex(ValueError, "Train/validation replay leakage"):
            expand_train_checkpoint_examples(self.index, bad_split, self.train)
        wrong = self.train.copy()
        wrong.loc[0, "p1_wins"] = 1 - wrong.loc[0, "p1_wins"]
        with self.assertRaisesRegex(ValueError, "contradictory"):
            expand_train_checkpoint_examples(self.index, self.partition, wrong)
        with self.assertRaisesRegex(ValueError, "integer"):
            expand_train_checkpoint_examples(self.index, self.partition, self.train,
                                            extras_per_replay=41)

    def test_cpu_smoke_trains_both_modes_and_predicts_on_same_s9_validation(self):
        expanded, _ = expand_train_checkpoint_examples(
            self.index, self.partition, self.train, extras_per_replay=12,
        )
        cfg = GRUConfig(hidden_size=8, batch_size=4, max_epochs=2, patience=2)
        pred = {}
        for mode in ("last9", "prefix"):
            train = CheckpointSequenceDataset(self.values, self.index, expanded, mode)
            valid = CheckpointSequenceDataset(self.values, self.index, self.val, mode)
            model, report = fit_checkpoint_gru(
                train, valid, cfg, torch.device("cpu"), 42, smoke=True,
            )
            self.assertEqual(report["test_evaluation_policy"], "not_accessed_during_training")
            pred[mode] = predict_checkpoint_gru(
                model, valid, cfg, torch.device("cpu"),
            )
        self.assertEqual(len(pred["last9"]), 8)
        self.assertEqual(len(pred["prefix"]), 8)
        result = compare_modes(pred, reps=10, seed=42)
        self.assertEqual(set(result), {"25", "50", "75", "90"})


if __name__ == "__main__":
    unittest.main()
