"""S1 regression tests: replay, time, label, and schema safety."""
from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from sc2proj.sequential import build_sequence_index, csv_schema_from_zip, read_snapshot_metadata, validate_schema


class SequenceWindowTests(unittest.TestCase):
    def setUp(self):
        # Intentionally unsorted across and within replay groups.
        a_times = [135, 150, 165, 180, 195, 225, 240, 255]
        b_times = [135, 150, 165, 180]
        records = ([{"replay_id": "A", "time_sec": t, "p1_wins": 1} for t in a_times]
                   + [{"replay_id": "B", "time_sec": t, "p1_wins": 0} for t in b_times])
        self.raw = pd.DataFrame(records).sample(frac=1.0, random_state=8).reset_index(drop=True)
        self.features = ("f1", "f2")

    def test_gap_never_crossed_and_sorted_positions_recover_source_rows(self):
        ix = build_sequence_index(self.raw, self.features, window_length=3)
        self.assertEqual(ix.stats()["windows"], 6)  # A: 3+1, B: 2
        self.assertEqual(ix.stats()["windowed_replays"], 2)
        self.assertEqual(ix.stats()["feature_count"], 2)
        self.assertEqual(ix.windows.columns.tolist(), ["start_row", "stop_row", "replay_id", "time_sec", "p1_wins"])
        for row in ix.windows.itertuples(index=False):
            sub = ix.ordered_rows.iloc[row.start_row:row.stop_row]
            self.assertEqual(len(sub), 3)
            self.assertEqual(sub.replay_id.nunique(), 1)
            self.assertEqual(sub.replay_id.iloc[-1], row.replay_id)
            self.assertEqual(sub.time_sec.iloc[-1], row.time_sec)
            self.assertEqual(sub.p1_wins.iloc[-1], row.p1_wins)
            np.testing.assert_array_equal(np.diff(sub.time_sec.to_numpy()), [15, 15])
            for ordered_row in sub.itertuples(index=False):
                source = self.raw.iloc[ordered_row.source_row]
                self.assertEqual(source.time_sec, ordered_row.time_sec)
                self.assertEqual(source.replay_id, ordered_row.replay_id)

    def test_stride_two_and_unwindowable_replay(self):
        ix = build_sequence_index(self.raw, self.features, window_length=3, stride=2)
        self.assertEqual(len(ix.windows), 4)  # A: two in first run + one in second; B: one
        ix_big = build_sequence_index(self.raw, self.features, window_length=9)
        self.assertEqual(ix_big.stats()["windows"], 0)

    def test_invalid_input_is_rejected(self):
        cases = []
        duplicate = pd.concat([self.raw, self.raw.iloc[[0]]], ignore_index=True)
        cases.append(duplicate)
        conflicting = self.raw.copy()
        conflicting.loc[conflicting.replay_id == "A", "p1_wins"] = [0, 1, 0, 1, 0, 1, 0, 1]
        cases.append(conflicting)
        missing = self.raw.copy()
        missing.loc[0, "replay_id"] = None
        cases.append(missing)
        invalid_target = self.raw.copy()
        invalid_target.loc[0, "p1_wins"] = 2
        cases.append(invalid_target)
        invalid_timestamp = self.raw.copy()
        invalid_timestamp.loc[0, "time_sec"] = 151
        cases.append(invalid_timestamp)
        for bad in cases:
            with self.subTest(case=len(cases)):
                with self.assertRaises(ValueError):
                    build_sequence_index(bad, self.features, window_length=3)
        with self.assertRaises(ValueError):
            build_sequence_index(self.raw, self.features, window_length=0)
        with self.assertRaises(ValueError):
            build_sequence_index(self.raw, ("replay_id",), window_length=3)

    def test_zip_schema_order_and_metadata_only_load(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            zpath = root / "tiny.zip"
            source = self.raw.assign(f1=1.0, f2=2.0)[["replay_id", "time_sec", "f1", "f2", "p1_wins"]]
            with zipfile.ZipFile(zpath, "w") as archive:
                archive.writestr("tiny.csv", source.to_csv(index=False))
            manifest = root / "schema.json"
            manifest.write_text(json.dumps({"v3_1_predictors": ["f1", "f2"]}))
            cols = csv_schema_from_zip(zpath)
            self.assertEqual(validate_schema(cols, manifest), self.features)
            actual = read_snapshot_metadata(zpath)
            self.assertEqual(actual.columns.tolist(), ["replay_id", "time_sec", "p1_wins", "source_row"])
            self.assertEqual(len(actual), len(self.raw))
            subset = actual[actual.replay_id == "A"].copy()
            subset_index = build_sequence_index(subset, self.features, window_length=3)
            for original in subset_index.ordered_rows.itertuples(index=False):
                self.assertEqual(original.replay_id, self.raw.iloc[original.source_row].replay_id)
                self.assertEqual(original.time_sec, self.raw.iloc[original.source_row].time_sec)
            manifest.write_text(json.dumps({"v3_1_predictors": ["f2", "f1"]}))
            with self.assertRaisesRegex(ValueError, "schema mismatch"):
                validate_schema(cols, manifest)
            pointer = root / "pointer.zip"
            pointer.write_text("version https://git-lfs.github.com/spec/v1\n")
            with self.assertRaisesRegex(ValueError, "Not a valid ZIP"):
                csv_schema_from_zip(pointer)


if __name__ == "__main__":
    unittest.main()
