"""Synthetic guards for prospective 816-source replay diagnostic."""
from __future__ import annotations

import importlib.util
import tempfile
import types
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/95_b3_probe_raw_only_replays.py"


def diagnostic():
    spec = importlib.util.spec_from_file_location("b3_raw_only_diagnostic", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replay(duration=230, players=2, winner=1):
    return types.SimpleNamespace(
        length=types.SimpleNamespace(seconds=duration),
        players=[types.SimpleNamespace(pid=i+1) for i in range(players)],
        winner=(types.SimpleNamespace(
            players=[types.SimpleNamespace(pid=winner)]
        ) if winner is not None else None),
    )


class B3RawOnlyDiagnosticTests(unittest.TestCase):
    def test_metadata_conditions_are_observations_not_historical_reasons(self):
        mod = diagnostic()
        good = mod.classify_replay_metadata(replay())
        self.assertEqual(good["current_metadata_conditions"], "none_of_tested_conditions")
        self.assertTrue(good["all_tested_metadata_conditions_pass"])
        short = mod.classify_replay_metadata(replay(duration=170, players=3, winner=None))
        self.assertEqual(
            short["current_metadata_conditions"],
            "duration_below_min;not_one_vs_one;winner_not_single_known_player",
        )
        long = mod.classify_replay_metadata(replay(duration=1801))
        self.assertEqual(long["current_metadata_conditions"], "duration_above_max")
        self.assertFalse(long["all_tested_metadata_conditions_pass"])

    def test_members_are_crc_checked_and_modern_loader_is_isolated(self):
        mod = diagnostic()
        with tempfile.TemporaryDirectory() as tmp:
            z = Path(tmp) / "source.zip"
            with zipfile.ZipFile(z, "w") as f:
                f.writestr("replays/valid.SC2Replay", b"MPQ\x1b valid container stub")
                f.writestr("replays/._metadata.SC2Replay",
                           b"\x00\x05\x16\x07 apple metadata")
            calls = []

            def fake_loader(path, load_level):
                self.assertTrue(Path(path).read_bytes().startswith(b"MPQ\x1b"))
                calls.append((path, load_level))
                return replay(duration=220, winner=2)

            with zipfile.ZipFile(z) as f:
                info = f.getinfo("replays/valid.SC2Replay")
                row = mod.inspect_member(
                    f, info, min_sec=180, max_sec=1800,
                    loader=fake_loader
                )
                self.assertEqual(row["header"], "MPQ_CANDIDATE")
                self.assertEqual(row["observed_duration_sec"], 220)
                self.assertEqual(row["current_metadata_conditions"],
                                 "none_of_tested_conditions")
                self.assertEqual(calls[0][1], 4)
                metadata = mod.inspect_member(
                    f, f.getinfo("replays/._metadata.SC2Replay"),
                    min_sec=180, max_sec=1800, loader=fake_loader
                )
                self.assertEqual(metadata["current_metadata_conditions"],
                                 "non_game_appledouble")
                self.assertEqual(len(calls), 1)

    def test_deterministic_disjoint_controls_and_smoke_sample(self):
        mod = diagnostic()
        raw = [f"bad_{i}.SC2Replay" for i in range(816)]
        good = [f"good_{i}.SC2Replay" for i in range(23241)]
        picked = mod.choose_sample(raw, good, max_raw_only=30,
                                   controls=15, seed=42)
        again = mod.choose_sample(raw, good, max_raw_only=30,
                                  controls=15, seed=42)
        self.assertEqual(picked, again)
        self.assertEqual(len(picked), 45)
        self.assertEqual(len(set(p[0] for p in picked)), 45)
        self.assertEqual(sum(p[1] == "FROZEN_MATCHED_CONTROL" for p in picked), 15)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            mod.choose_sample(raw, raw, max_raw_only=10, controls=5, seed=42)


if __name__ == "__main__":
    unittest.main()
