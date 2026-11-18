"""S2: replay isolation, train-only scaling, time safety, CSV source offsets."""
from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from sc2proj.sequential.data import (
    IndexedSequences, load_features_in_sorted_order, load_index,
    load_replay_partitions, make_loaders, prepare_sequential_data, sha256_file,
)


class SequentialLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        records = []
        for replay, offset, winner in (("train-A", 1, 1), ("val-B", 100, 0), ("test-C", -100, 1)):
            for k, t in enumerate((135, 150, 165, 180)):
                records.append({"replay_id": replay, "time_sec": t,
                                "f1": float(offset + k), "f2": float(2 * (offset + k)),
                                "p1_wins": winner})
        self.raw = pd.DataFrame(records).sample(frac=1.0, random_state=12).reset_index(drop=True)
        self.zip_path = self.root / "tiny.zip"
        with zipfile.ZipFile(self.zip_path, "w", compression=zipfile.ZIP_DEFLATED) as out:
            out.writestr("tiny.csv", self.raw.to_csv(index=False))
        self.ordered = self.raw[["replay_id", "time_sec", "p1_wins"]].copy()
        self.ordered["source_row"] = np.arange(len(self.ordered), dtype=np.int64)
        self.ordered = self.ordered.sort_values(["replay_id", "time_sec"]).reset_index(drop=True)
        windows = []
        for base in (0, 4, 8):
            for start in (base, base + 1):
                end = start + 3
                last = self.ordered.iloc[end - 1]
                windows.append({"start_row": start, "stop_row": end,
                                "replay_id": last.replay_id, "time_sec": int(last.time_sec),
                                "p1_wins": int(last.p1_wins)})
        self.windows = pd.DataFrame(windows)
        self.manifest = {
            "schema": "sc2.sequential.window_index.v1", "features": ["f1", "f2"],
            "feature_count": 2, "rows": 12, "windows": 6, "window_length": 3,
            "step_sec": 15, "subset_max_replays": None,
            "source_zip_sha256": sha256_file(self.zip_path),
        }
        self.idx_dir = self.root / "s1"
        self.idx_dir.mkdir()
        (self.idx_dir / "manifest.json").write_text(json.dumps(self.manifest))
        self.index = IndexedSequences(self.ordered, self.windows, ("f1", "f2"), self.manifest)
        self.split_path = self.root / "split.json"
        self.split = {"seed": 42, "train_groups": ["train-A"],
                      "val_groups": ["val-B"], "test_groups": ["test-C"],
                      "n_train_groups": 1, "n_val_groups": 1, "n_test_groups": 1}
        self.split_path.write_text(json.dumps(self.split))

    def test_csv_source_row_permutation_and_train_only_scaler(self):
        split = load_replay_partitions(self.split_path, self.index)
        p = prepare_sequential_data(self.index, split, self.zip_path)
        self.assertEqual([len(p.datasets[x]) for x in ("train", "val", "test")], [2, 2, 2])
        # Only original train-A rows (1,2,3,4), not the overlapping windows.
        np.testing.assert_allclose(p.scaler.mean_, [2.5, 5.0])
        self.assertEqual(int(p.scaler.n_samples_seen_), 4)
        seq, y = p.datasets["train"][0]
        self.assertEqual(tuple(seq.shape), (3, 2))
        self.assertEqual(float(y), 1.0)
        np.testing.assert_allclose(seq[:, 0].numpy(),
                                   [(i - 2.5) / np.std([1, 2, 3, 4]) for i in (1, 2, 3)], atol=1e-6)
        # Large validation values must not affect scaler fitting.
        self.assertGreater(float(p.datasets["val"][0][0][0, 0]), 40)
        self.assertEqual(int(p.features_sorted.shape[0]), 12)
        self.assertTrue(np.isfinite(p.features_sorted).all())
        loaders = make_loaders(p, batch_size=2)
        x, labels = next(iter(loaders["train"]))
        self.assertEqual(tuple(x.shape), (2, 3, 2))
        self.assertEqual(tuple(labels.shape), (2,))
        self.assertEqual(labels.dtype, __import__("torch").float32)

    def test_split_rejects_overlap_unassigned_or_inaccurate_counts(self):
        bad_specs = (
            {**self.split, "val_groups": ["train-A", "val-B"], "n_val_groups": 2},
            {**self.split, "test_groups": ["unknown"], "n_test_groups": 1},
            {**self.split, "n_train_groups": 99},
            {**self.split, "train_groups": ["train-A", "train-A"], "n_train_groups": 2},
        )
        for value in bad_specs:
            self.split_path.write_text(json.dumps(value))
            with self.subTest(value=value), self.assertRaises(ValueError):
                load_replay_partitions(self.split_path, self.index)

    def test_index_rejects_corrupted_windows(self):
        with patch("pandas.read_parquet", side_effect=[self.ordered, self.windows]):
            clean = load_index(self.idx_dir)
            self.assertEqual(len(clean.windows), 6)
        crossed = self.windows.copy()
        crossed.loc[1, "stop_row"] = 5  # crosses into validation replay (and wrong length)
        with patch("pandas.read_parquet", side_effect=[self.ordered, crossed]):
            with self.assertRaises(ValueError):
                load_index(self.idx_dir)
        # Construct a full-length window crossing the split between the replays.
        crossed = self.windows.copy()
        crossed.loc[1, "start_row"] = 2
        crossed.loc[1, "stop_row"] = 5
        crossed.loc[1, ["replay_id", "time_sec", "p1_wins"]] = self.ordered.loc[4, ["replay_id", "time_sec", "p1_wins"]].to_list()
        with patch("pandas.read_parquet", side_effect=[self.ordered, crossed]):
            with self.assertRaisesRegex(ValueError, "crosses"):
                load_index(self.idx_dir)

    def test_source_integrity_nan_and_smoke_subset(self):
        loaded = load_features_in_sorted_order(self.zip_path, self.index)
        self.assertEqual(tuple(loaded.shape), (12, 2))
        tampered = IndexedSequences(self.ordered, self.windows, ("f1", "f2"),
                                    {**self.manifest, "source_zip_sha256": "0" * 64})
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            load_features_in_sorted_order(self.zip_path, tampered)
        # Smoke indexes are allowed to cover a subset of the historical split groups.
        subset_rows = self.ordered.iloc[4:8].copy().reset_index(drop=True)
        subset_wins = self.windows.iloc[2:4].copy().reset_index(drop=True)
        subset_wins["start_row"] -= 4
        subset_wins["stop_row"] -= 4
        subset = IndexedSequences(subset_rows, subset_wins, ("f1", "f2"),
                                  {**self.manifest, "rows": 4, "windows": 2,
                                   "subset_max_replays": 1})
        sp = load_replay_partitions(self.split_path, subset)
        self.assertEqual(len(sp.window_positions["train"]), 2)
        self.assertEqual(len(sp.window_positions["val"]), 0)


if __name__ == "__main__":
    unittest.main()
