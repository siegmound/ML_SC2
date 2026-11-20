"""Synthetic tests for B3 sampled historical raw-to-frozen parity (FAIR runtime)."""
from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def auditor():
    spec = importlib.util.spec_from_file_location(
        "b3_historical_raw_parity", ROOT / "scripts/79_compare_b3_historical_raw_parity.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class B3HistoricalParityTests(unittest.TestCase):
    def test_replay_parity_and_numeric_tampering(self):
        tool = auditor()
        frozen = pd.DataFrame({
            "replay_id": ["r.SC2Replay"] * 3,
            "time_sec": [135, 150, 165],
            "p1_wins": [1, 1, 1],
            "p1_workers": [12, 15, 17],
            "diff_workers": [2, 4, 5],
            "p1_sq": [0.1, 0.2, 0.3],
        })
        reconstructed = frozen.copy()
        result, cols, diff = tool.compare_replay(
            "r.SC2Replay", frozen, reconstructed, atol=1e-6, rtol=1e-6
        )
        self.assertEqual(result["status"], "SAMPLED_PARITY")
        self.assertEqual(result["matched_rows"], 3)
        self.assertEqual(result["feature_comparisons"], 9)
        self.assertFalse(diff)
        self.assertEqual(len(cols), 3)
        reconstructed.loc[1, "p1_workers"] = 19
        result, _, diff = tool.compare_replay(
            "r.SC2Replay", frozen, reconstructed, atol=1e-6, rtol=1e-6
        )
        self.assertEqual(result["status"], "RECONSTRUCTION_MISMATCH")
        self.assertEqual(result["feature_mismatches"], 1)
        self.assertEqual(diff[0]["feature"], "p1_workers")
        reconstructed = frozen.drop(index=0).copy()
        result, _, _ = tool.compare_replay(
            "r.SC2Replay", frozen, reconstructed, atol=1e-6, rtol=1e-6
        )
        self.assertEqual(result["time_keys_only_frozen"], 1)
        self.assertEqual(result["status"], "RECONSTRUCTION_MISMATCH")
        reconstructed = frozen.copy()
        reconstructed.loc[1, "p1_wins"] = 0
        result, _, _ = tool.compare_replay(
            "r.SC2Replay", frozen, reconstructed, atol=1e-6, rtol=1e-6
        )
        self.assertEqual(result["label_mismatches"], 1)
        self.assertEqual(result["status"], "RECONSTRUCTION_MISMATCH")

    def test_deterministic_sampling_of_existing_sources(self):
        tool = auditor()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            for i in range(5):
                (raw / f"r{i}.SC2Replay").write_bytes(b"MPQ\x1bSYNTHETIC")
            inventory = root / "identities.csv"
            with inventory.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "replay_id", "source_entries", "in_frozen_v3_1", "classification"
                ])
                writer.writeheader()
                for i in range(5):
                    writer.writerow({
                        "replay_id": f"r{i}.SC2Replay",
                        "source_entries": "1", "in_frozen_v3_1": "True",
                        "classification": "SOURCE_PRESENT_FROZEN",
                    })
            selected = tool.select_present_replays(inventory, raw, 3, seed=42)
            self.assertEqual(selected, tool.select_present_replays(inventory, raw, 3, seed=42))
            self.assertEqual(len(selected), 3)
            (raw / selected[0][0]).write_bytes(b"\x00\x05\x16\x07APPLEDOUBLE")
            with self.assertRaisesRegex(ValueError, "Not a plausible MPQ replay"):
                tool.select_present_replays(inventory, raw, 5, seed=42)

    def test_historical_source_sha_checks_fail_closed(self):
        tool = auditor()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bad_source.zip"
            with zipfile.ZipFile(source, "w") as archive:
                for name in tool.SCRIPT_NAMES:
                    archive.writestr(name, b"tampered")
            destination = root / "work"
            destination.mkdir()
            with self.assertRaisesRegex(ValueError, "Historical source SHA mismatch"):
                tool.load_historical_source(source, destination)
            self.assertFalse((destination / "sc2_bridge_v3_counter120.py").exists())

    def test_frozen_sample_key_identity(self):
        tool = auditor()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("frozen.csv",
                                 "replay_id,time_sec,p1_wins,p1_workers\n"
                                 "r1.SC2Replay,135,1,12\n"
                                 "r1.SC2Replay,150,1,14\n"
                                 "r2.SC2Replay,135,0,10\n")
            frame = tool.frozen_selected(path, {"r1.SC2Replay"})
            self.assertEqual(len(frame), 2)
            self.assertEqual(set(frame.replay_id), {"r1.SC2Replay"})
            with self.assertRaisesRegex(ValueError, "Missing selected replays"):
                tool.frozen_selected(path, {"r1.SC2Replay", "absent.SC2Replay"})


if __name__ == "__main__":
    unittest.main()
