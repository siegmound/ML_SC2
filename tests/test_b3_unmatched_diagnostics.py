"""Regression tests for conservative B3 unmatched replay diagnostics."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def script():
    spec = importlib.util.spec_from_file_location(
        "b3_unmatched_probe", ROOT / "scripts/78_diagnose_b3_unmatched_replays.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class B3UnmatchedDiagnosticsTests(unittest.TestCase):
    def test_candidate_selection_ignores_frozen_missing_from_partial_sources(self):
        tool = script()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unresolved_ids.csv"
            path.write_text(
                "replay_id,classification\n"
                "r9.SC2Replay,SOURCE_PRESENT_FROZEN_ABSENT_REASON_UNKNOWN\n"
                "r10.SC2Replay,FROZEN_ID_MISSING_FROM_SCANNED_SOURCES\n"
                "r8.SC2Replay,SOURCE_PRESENT_FROZEN_ABSENT_REASON_UNKNOWN\n"
            )
            self.assertEqual(tool.candidate_ids(path), ["r8.SC2Replay", "r9.SC2Replay"])
            raw = Path(directory) / "raw"
            raw.mkdir()
            for name in ("r8.SC2Replay", "r9.SC2Replay"):
                (raw / name).write_bytes(b"MPQ placeholder for basename test")
            paths = tool.original_replay_files(raw, tool.candidate_ids(path))
            self.assertEqual(set(paths), {"r8.SC2Replay", "r9.SC2Replay"})
            (raw / "second").mkdir()
            (raw / "second/r8.SC2Replay").write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "Expected exactly one real source"):
                tool.original_replay_files(raw, tool.candidate_ids(path))

    def test_duration_diagnostics_are_not_historical_exclusion_proof(self):
        tool = script()
        replay = Path("example.SC2Replay")
        def good(seconds):
            return lambda path, load_level: SimpleNamespace(
                length=SimpleNamespace(total_seconds=lambda: seconds)
            )
        self.assertEqual(tool.probe_header(replay, good(179))["duration_category"],
                         "BELOW_HISTORICAL_MIN_180")
        self.assertEqual(tool.probe_header(replay, good(180))["duration_category"],
                         "WITHIN_HISTORICAL_DURATION_BAND")
        self.assertEqual(tool.probe_header(replay, good(1800))["duration_category"],
                         "WITHIN_HISTORICAL_DURATION_BAND")
        self.assertEqual(tool.probe_header(replay, good(1801))["duration_category"],
                         "ABOVE_HISTORICAL_MAX_1800")
        def broken(path, load_level):
            raise RuntimeError("cannot parse")
        result = tool.probe_header(replay, broken)
        self.assertEqual(result["header_load_status"], "ERROR")
        self.assertEqual(result["header_error_type"], "RuntimeError")
        self.assertIsNone(result["duration_seconds"])

    def test_rejects_duplicate_candidate_ids_and_path_traversal(self):
        tool = script()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unresolved_ids.csv"
            path.write_text(
                "replay_id,classification\n"
                "x.SC2Replay,SOURCE_PRESENT_FROZEN_ABSENT_REASON_UNKNOWN\n"
                "x.SC2Replay,SOURCE_PRESENT_FROZEN_ABSENT_REASON_UNKNOWN\n"
            )
            with self.assertRaisesRegex(ValueError, "Duplicate unmatched replay IDs"):
                tool.candidate_ids(path)
            path.write_text(
                "replay_id,classification\n"
                "../x.SC2Replay,SOURCE_PRESENT_FROZEN_ABSENT_REASON_UNKNOWN\n"
            )
            with self.assertRaisesRegex(ValueError, "basename-only"):
                tool.candidate_ids(path)


if __name__ == "__main__":
    unittest.main()
