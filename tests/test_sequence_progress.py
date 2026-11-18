"""S8: exact percent-of-game checkpoint selection and paired validation tests."""
from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd

from sc2proj.sequential.data import IndexedSequences
from sc2proj.sequential.progress import (
    CHECKPOINTS, join_checked_predictions, score_progress, select_exact_checkpoints,
)
from sc2proj.sequential.windows import build_sequence_index


def synthetic_index():
    records = []
    for rid, n, label in (("A", 40, 0), ("B", 40, 1), ("short", 20, 0)):
        for i in range(n):
            records.append({"replay_id": rid, "time_sec": i * 15,
                            "p1_wins": label, "source_row": len(records)})
    ix = build_sequence_index(pd.DataFrame(records), ("f1", "f2"), 9, 1, 15)
    index = IndexedSequences(
        ix.ordered_rows, ix.windows, ix.feature_names,
        {"window_length": 9, "stride": 1},
    )
    partition = SimpleNamespace(
        groups={"val": frozenset(("A", "B", "short"))},
        window_positions={"val": np.arange(len(index.windows), dtype=np.int64)},
    )
    return index, partition


class ProgressCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.index, self.partition = synthetic_index()
        self.selected, self.coverage = select_exact_checkpoints(
            self.index, self.partition
        )

    def test_fractions_use_full_replay_snapshot_count_without_future(self):
        self.assertEqual(self.coverage["validation_replays_total"], 3)
        self.assertEqual(self.coverage["balanced_replays"], 2)
        self.assertEqual(self.coverage["replays_excluded"], 1)
        self.assertEqual(len(self.selected), 8)
        self.assertEqual(set(self.selected["replay_id"].tolist()), {"A", "B"})
        self.assertEqual(self.coverage["raw_eligible_replays_by_checkpoint"]["25"], 2)
        a = self.selected[self.selected["replay_id"] == "A"]
        self.assertEqual(a["observed_snapshot_count"].tolist(), [10, 20, 30, 36])
        self.assertEqual(a["time_sec"].tolist(), [135, 285, 435, 525])
        self.assertEqual(a["progress_percent"].tolist(), list(CHECKPOINTS))
        self.assertTrue((a["observed_fraction"] <= 0.90).all())

    def test_ineligible_checkpoint_excludes_entire_replay(self):
        ix = self.index
        remove_key = (ix.windows.replay_id == "A") & (ix.windows.time_sec == 135)
        windows = ix.windows.loc[~remove_key].reset_index(drop=True)
        changed = replace(ix, windows=windows)
        part = SimpleNamespace(
            groups=self.partition.groups,
            window_positions={"val": np.arange(len(windows), dtype=np.int64)},
        )
        selected, coverage = select_exact_checkpoints(changed, part)
        self.assertEqual(coverage["balanced_replays"], 1)
        self.assertEqual(set(selected.replay_id), {"B"})
        self.assertEqual(coverage["raw_eligible_replays_by_checkpoint"]["25"], 1)

    def test_no_nearest_future_endpoint_substitution(self):
        ix = self.index
        altered = ix.windows.loc[
            ~((ix.windows.replay_id == "B") & (ix.windows.time_sec == 135))
        ].reset_index(drop=True)
        new = replace(ix, windows=altered)
        partition = SimpleNamespace(
            groups=self.partition.groups,
            window_positions={"val": np.arange(len(altered), dtype=np.int64)},
        )
        selected, _ = select_exact_checkpoints(new, partition)
        self.assertNotIn("B", set(selected.replay_id))

    def test_wrong_percent_configuration_rejected(self):
        for percents in ((0, 50), (25, 25), (25, 101), (), (25.5, 50)):
            with self.subTest(percents=percents), self.assertRaises(ValueError):
                select_exact_checkpoints(self.index, self.partition, percents)

    def test_join_predictions_enforces_binary_label_parity_and_coverage(self):
        base = self.selected[["replay_id", "time_sec", "p1_wins"]].copy()
        predictions = base.rename(columns={"p1_wins": "y_true"})
        predictions["y_prob"] = np.where(predictions.y_true == 1, 0.8, 0.2)
        predictions["y_pred"] = predictions["y_true"]
        full = join_checked_predictions(self.selected, {"v31_original64": predictions})
        self.assertIn("v31_original64_prob", full.columns)
        bad = predictions.copy()
        bad.loc[0, "y_true"] = 1 - bad.loc[0, "y_true"]
        with self.assertRaisesRegex(ValueError, "winner label mismatch"):
            join_checked_predictions(self.selected, {"v31_original64": bad})
        with self.assertRaisesRegex(ValueError, "missing selected checkpoint"):
            join_checked_predictions(self.selected, {"v31_original64": predictions.iloc[:-1]})

    def test_progression_uses_identical_replays_and_whole_replay_bootstrap(self):
        sample = self.selected.copy()
        # Always wrong at 25%, always correct at 90%.
        sample["v31_original64_pred"] = np.where(
            sample.progress_percent == 25, 1 - sample.p1_wins, sample.p1_wins
        ).astype(np.int8)
        sample["v31_original64_prob"] = np.where(
            sample.v31_original64_pred == 1, 0.8, 0.2
        )
        metrics, gains = score_progress(
            sample, arms=("v31_original64",), reps=100, rng_seed=11
        )
        self.assertEqual(metrics["25"]["models"]["v31_original64"]["accuracy"], 0.0)
        self.assertEqual(metrics["90"]["models"]["v31_original64"]["accuracy"], 1.0)
        self.assertAlmostEqual(gains["v31_original64"]["accuracy_delta"], 1.0)
        self.assertEqual(gains["v31_original64"]["paired_replay_bootstrap"][
            "accuracy_delta_ci95"], [1.0, 1.0])
        self.assertLess(gains["v31_original64"]["log_loss_delta"], 0)
        self.assertEqual(gains["v31_original64"]["paired_replay_bootstrap"][
            "replay_groups"], 2)

    def test_missing_one_checkpoint_in_score_fails(self):
        sample = self.selected.copy()
        sample["v31_original64_pred"] = sample.p1_wins
        sample["v31_original64_prob"] = np.where(sample.p1_wins == 1, 0.8, 0.2)
        sample = sample.loc[
            ~((sample.replay_id == "A") & (sample.progress_percent == 90))
        ]
        with self.assertRaisesRegex(ValueError, "different replay populations"):
            score_progress(sample, arms=("v31_original64",), reps=0)


if __name__ == "__main__":
    unittest.main()
