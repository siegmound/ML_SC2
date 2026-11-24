"""Synthetic provenance-preserving 16-survivor V4 parser preflight guards."""
from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def probe_module():
    spec = importlib.util.spec_from_file_location(
        "b3_v4_survivor_probe", ROOT / "scripts/96_b3_probe_v4_parser_survivors.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeBridge:
    early_cut_sec = 120
    snapshot_step_sec = 15

    def __init__(self, invalid=False):
        self.invalid = invalid

    def process_replay(self, path):
        from sc2proj.bridge import ParseResult
        name = Path(path).name
        if self.invalid:
            return ParseResult(name, "failed", "missing_start_locations", None, {})
        return ParseResult(
            name, "ok", "",
            pd.DataFrame({
                "replay_id": [name, name], "time_sec": [135, 150],
                "p1_wins": [0, 0], "diff_army": [3.0, 2.0],
            }), {
                "match_duration_sec": 800, "player_count": 2,
                "winner_known": True,
            }
        )


class RawSurvivorV4PreflightTests(unittest.TestCase):
    def test_selects_exactly_16_real_candidates_and_controls(self):
        probe = probe_module()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "diagnostic.csv"
            fields = [
                "replay_id", "source_membership",
                "current_metadata_conditions", "source_zip_member",
                "member_sha256",
            ]
            rows = [
                {
                    "replay_id": f"raw_{i}.SC2Replay",
                    "source_membership": "RAW_NOT_IN_FROZEN",
                    "current_metadata_conditions": (
                        "none_of_tested_conditions" if i < 16
                        else "duration_above_max"
                    ),
                    "source_zip_member": f"replays/raw_{i}.SC2Replay",
                    "member_sha256": "a" * 64,
                }
                for i in range(816)
            ] + [
                {
                    "replay_id": f"control_{i}.SC2Replay",
                    "source_membership": "FROZEN_MATCHED_CONTROL",
                    "current_metadata_conditions": "none_of_tested_conditions",
                    "source_zip_member": f"replays/control_{i}.SC2Replay",
                    "member_sha256": "b" * 64,
                }
                for i in range(100)
            ]
            with path.open("w", newline="") as out:
                writer = csv.DictWriter(out, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            first = probe.select_metadata_survivors(path, controls=15, seed=42)
            second = probe.select_metadata_survivors(path, controls=15, seed=42)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 31)
            self.assertEqual(sum(
                r["source_membership"] == "RAW_NOT_IN_FROZEN"
                for r in first
            ), 16)
            self.assertEqual(len({r["replay_id"] for r in first}), 31)

    def test_temporary_source_provenance_and_feature_validation(self):
        probe = probe_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "synthetic.zip"
            payload = b"MPQ\x1b test-replay"
            with zipfile.ZipFile(source, "w") as f:
                f.writestr("replays/game.SC2Replay", payload)
            with zipfile.ZipFile(source) as f:
                member = f.getinfo("replays/game.SC2Replay")
                import hashlib
                sha = hashlib.sha256(payload).hexdigest()
                result = probe.probe_selected_member(
                    f, member, sha, FakeBridge(), tmpdir=root
                )
                self.assertEqual(result["v4_status"], "ok")
                self.assertEqual(result["v4_snapshots"], 2)
                self.assertEqual(result["v4_feature_count"], 1)
                self.assertEqual(result["v4_match_duration_sec"], 800)
                self.assertFalse((root / "game.SC2Replay").exists())
                failed = probe.probe_selected_member(
                    f, member, sha, FakeBridge(invalid=True), tmpdir=root
                )
                self.assertEqual(failed["v4_reason"], "missing_start_locations")
                self.assertEqual(failed["v4_status"], "failed")

    def test_refuses_mismatched_member_digest(self):
        probe = probe_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "synthetic.zip"
            with zipfile.ZipFile(source, "w") as f:
                f.writestr("replays/game.SC2Replay", b"MPQ\x1b test-replay")
            with zipfile.ZipFile(source) as f:
                with self.assertRaisesRegex(ValueError, "bytes changed"):
                    probe.probe_selected_member(
                        f, f.getinfo("replays/game.SC2Replay"),
                        "0"*64, FakeBridge(), tmpdir=root
                    )


if __name__ == "__main__":
    unittest.main()
