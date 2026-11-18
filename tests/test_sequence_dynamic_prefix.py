"""S13 dynamic train-cutoff sampling: deterministic, bounded, replay-disjoint."""
from __future__ import annotations

from math import ceil
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch

from test_sequence_prefix import make_toy
from sc2proj.sequential.dynamic_prefix import (
    DynamicReplayCutoffSampler, fit_dynamic_gru,
)
from sc2proj.sequential.extra_prefixes import expand_train_checkpoint_examples
from sc2proj.sequential.gru import GRUConfig
from sc2proj.sequential.prefix import (
    CheckpointSequenceDataset, compare_modes, predict_checkpoint_gru,
)


class DynamicPrefixTests(unittest.TestCase):
    def setUp(self):
        self.index, self.part, self.values, self.anchors, self.valid = make_toy()
        self.extra, self.expansion = expand_train_checkpoint_examples(
            self.index, self.part, self.anchors, extras_per_replay=12,
        )

    def dataset(self, mode="prefix"):
        return CheckpointSequenceDataset(self.values, self.index, self.extra, mode)

    def test_exact_four_per_replay_per_epoch_and_same_s9_budget(self):
        dataset = self.dataset()
        sampler = DynamicReplayCutoffSampler(dataset, batch_size=3, seed=42)
        self.assertEqual(len(dataset), 32)
        self.assertEqual(sampler.n_replays, 2)
        self.assertEqual(sampler.samples_per_epoch, len(self.anchors))
        self.assertEqual(len(sampler), ceil(8 / 3))
        for epoch in (1, 2, 3):
            sampler.set_epoch(epoch)
            chosen = sampler.selected_indices()
            self.assertEqual(len(chosen), 8)
            self.assertEqual(len(np.unique(chosen)), 8)
            self.assertEqual(
                dataset.rows.iloc[chosen].groupby("replay_id").size().to_dict(),
                {"train-A": 4, "train-B": 4},
            )
            batches = list(iter(sampler))
            self.assertEqual(sorted(i for b in batches for i in b), sorted(chosen.tolist()))
            self.assertEqual(len(batches), len(sampler))

    def test_deterministic_across_runs_and_modes_but_changes_over_epochs(self):
        a = DynamicReplayCutoffSampler(self.dataset("last9"), 4, seed=42)
        b = DynamicReplayCutoffSampler(self.dataset("prefix"), 4, seed=42)
        all_selections = []
        for epoch in (1, 2, 3, 4):
            a.set_epoch(epoch)
            b.set_epoch(epoch)
            left = a.selected_indices()
            right = b.selected_indices()
            self.assertTrue(np.array_equal(left, right))
            self.assertTrue(np.array_equal(left, a.selected_indices()))
            all_selections.append(tuple(left.tolist()))
        self.assertGreater(len(set(all_selections)), 1)

    def test_exact_four_anchor_strata_keep_game_time_coverage(self):
        dataset = self.dataset("prefix")
        sampler = DynamicReplayCutoffSampler(dataset, 4, seed=42)
        sampler.set_epoch(3)
        chosen = set(sampler.selected_indices().tolist())
        self.assertEqual(len(sampler.strata), 2)
        for four in sampler.strata:
            self.assertEqual(len(four), 4)
            self.assertEqual(sum(sum(int(i in chosen) for i in pool) for pool in four), 4)
            self.assertTrue(all(sum(int(i in chosen) for i in pool) == 1 for pool in four))
        self.assertTrue(dataset.rows.iloc[list(chosen)].observed_snapshot_count.ge(9).all())

    def test_reject_incomplete_anchors_or_unrecognized_train_checkpoint(self):
        dropped = self.extra.loc[~(
            (self.extra.replay_id == "train-A")
            & (self.extra.progress_percent == 25)
        )].reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "four original"):
            DynamicReplayCutoffSampler(
                CheckpointSequenceDataset(self.values, self.index, dropped, "prefix"),
                4, seed=42,
            )
        altered = self.extra.copy()
        altered.loc[altered.progress_percent == -1, "progress_percent"] = 12
        with self.assertRaisesRegex(ValueError, "unknown checkpoint"):
            DynamicReplayCutoffSampler(
                CheckpointSequenceDataset(self.values, self.index, altered, "last9"),
                4, seed=42,
            )

    def test_only_train_replays_and_no_future_snapshot_input(self):
        self.assertEqual(set(self.extra.replay_id), {"train-A", "train-B"})
        self.assertTrue(set(self.extra.replay_id).isdisjoint(set(self.valid.replay_id)))
        ds = self.dataset("prefix")
        sampler = DynamicReplayCutoffSampler(ds, 4, seed=42)
        sampler.set_epoch(1)
        i = int(sampler.selected_indices()[0])
        sequence = ds[i][0].clone()
        end = int(ds.ends[i])
        old = self.values[end:].copy()
        try:
            self.values[end:] = 888888.0
            self.assertTrue(torch.equal(ds[i][0], sequence))
        finally:
            self.values[end:] = old

    def test_cpu_two_arm_smoke_and_unchanged_val_checkpoints(self):
        config = GRUConfig(hidden_size=8, batch_size=4, max_epochs=2, patience=2)
        pred = {}
        for mode in ("last9", "prefix"):
            train = self.dataset(mode)
            val = CheckpointSequenceDataset(self.values, self.index, self.valid, mode)
            model, report = fit_dynamic_gru(
                train, val, config, torch.device("cpu"), seed=42, smoke=True,
            )
            self.assertEqual(report["train_pool_examples"], 32)
            self.assertEqual(report["train_examples_per_epoch"], 8)
            self.assertEqual(report["train_batches_per_epoch"], 2)
            self.assertEqual(report["status"], "S13_SMOKE_FIT_PASS")
            self.assertEqual(
                report["test_evaluation_policy"],
                "not_accessed_during_training",
            )
            self.assertTrue(1 <= report["selected_epoch"] <= 2)
            pred[mode] = predict_checkpoint_gru(
                model, val, config, torch.device("cpu"),
            )
            self.assertEqual(len(pred[mode]), 8)
            self.assertEqual(
                set(pred[mode].replay_id), {"val-A", "val-B"},
            )
        compared = compare_modes(pred, reps=10, seed=42)
        self.assertEqual(set(compared), {"25", "50", "75", "90"})


if __name__ == "__main__":
    unittest.main()
