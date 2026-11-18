"""S9 multi-seed aggregation: fail-closed prediction provenance checks."""
from __future__ import annotations

import runpy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
aggregate = runpy.run_path(str(ROOT / "scripts/86_aggregate_prefix_vs_last9.py"))
read_predictions = aggregate["read_predictions"]


def synthetic_frame() -> pd.DataFrame:
    rows = []
    for replay, true in (("A", 0), ("B", 1)):
        for p, t in zip((25, 50, 75, 90), (9, 19, 29, 35)):
            probability = (0.8 if true else 0.2)
            rows.append({
                "replay_id": replay, "time_sec": t * 15, "progress_percent": p,
                "y_true": true, "y_prob": probability, "y_pred": true,
            })
    return pd.DataFrame(rows)


class S9AggregateValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "pred.csv"
        self.frame = synthetic_frame()

    def tearDown(self):
        self.temp.cleanup()

    def _read(self, frame: pd.DataFrame):
        frame.to_csv(self.path, index=False)
        return read_predictions(self.path)

    def test_valid_balanced_checkpoints(self):
        pred = self._read(self.frame.sample(frac=1, random_state=7))
        self.assertEqual(len(pred), 8)
        self.assertEqual(set(pred.progress_percent), {25, 50, 75, 90})
        self.assertEqual(pred.groupby("replay_id").size().to_dict(), {"A": 4, "B": 4})

    def test_duplicate_replay_time_fails(self):
        bad = pd.concat([self.frame, self.frame.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self._read(bad)

    def test_missing_checkpoint_fails(self):
        with self.assertRaisesRegex(ValueError, "Unbalanced"):
            self._read(self.frame.iloc[:-1])

    def test_bad_probability_and_threshold_fail(self):
        bad = self.frame.copy()
        bad.loc[0, "y_prob"] = 1.4
        with self.assertRaisesRegex(ValueError, "Invalid prediction"):
            self._read(bad)
        bad = self.frame.copy()
        bad.loc[0, "y_pred"] = 1
        with self.assertRaisesRegex(ValueError, "threshold"):
            self._read(bad)

    def test_inconsistent_winner_label_fails(self):
        bad = self.frame.copy()
        bad.loc[0, "y_true"] = 1
        with self.assertRaisesRegex(ValueError, "Replay label changes"):
            self._read(bad)


if __name__ == "__main__":
    unittest.main()
