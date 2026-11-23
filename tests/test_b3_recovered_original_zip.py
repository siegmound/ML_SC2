"""Synthetic ZIP-only regressions; no real replay payload downloaded."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/93_b3_reconcile_original_replays_zip.py"


def script():
    spec = importlib.util.spec_from_file_location("b3_recovered_zip", SOURCE)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class RawZipReconciliationTests(unittest.TestCase):
    def make_zip(self, dest: Path):
        with zipfile.ZipFile(dest, "w") as z:
            z.writestr("replays/a.SC2Replay", b"MPQ\x1bgame A")
            z.writestr("replays/b.SC2Replay", b"MPQ\x1bgame B")
            z.writestr("replays/c.SC2Replay", b"MPQ\x1bgame C")
            z.writestr("replays/2017_tournament.zip", b"nested but not traversed")
            z.writestr("replays/details.json", json.dumps({"packs": 71, "origin": "test"}))

    def test_census_preserves_only_direct_raw_members_and_json_metadata(self):
        a = script()
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "raw.zip"
            self.make_zip(f)
            summary, source, other = a.inventory_raw_zip(f)
            self.assertEqual(summary["direct_raw_replay_entries"], 3)
            self.assertEqual(summary["all_file_entries"], 5)
            self.assertEqual(summary["nested_zip_entries_not_recursed"], 1)
            self.assertEqual(summary["json_entries"], 1)
            self.assertEqual(set(source), {"a.SC2Replay", "b.SC2Replay", "c.SC2Replay"})
            self.assertEqual(
                next(x["json_top_level_keys"] for x in other if x["suffix"] == ".json"),
                "origin;packs"
            )
            report, only_raw, only_final = a.compare_identities(
                source, {"a.SC2Replay", "b.SC2Replay"}
            )
            self.assertEqual(report["raw_ids_also_present_frozen"], 2)
            self.assertEqual(only_raw, ["c.SC2Replay"])
            self.assertEqual(only_final, [])
            self.assertFalse(report["upstream_exclusion_causes_attested"])

    def test_frozen_ids_read_only_without_loading_every_feature(self):
        a = script()
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "frozen.zip"
            with zipfile.ZipFile(f, "w") as z:
                z.writestr("v3.csv",
                           "replay_id,time_sec,p1_wins,p1_combat\n"
                           "a.SC2Replay,135,0,0.5\n"
                           "a.SC2Replay,150,0,0.6\n"
                           "b.SC2Replay,135,1,0.8\n")
            ids, n, member = a.frozen_replay_ids(f)
            self.assertEqual(ids, {"a.SC2Replay", "b.SC2Replay"})
            self.assertEqual(n, 3)
            self.assertEqual(member, "v3.csv")

    def test_rejects_duplicate_replay_basename_and_zip_slip(self):
        a = script()
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "duplicate.zip"
            with zipfile.ZipFile(f, "w") as z:
                z.writestr("pack1/match.SC2Replay", b"MPQ\x1b1")
                z.writestr("pack2/match.SC2Replay", b"MPQ\x1b2")
            with self.assertRaisesRegex(ValueError, "Ambiguous repeated replay"):
                a.inventory_raw_zip(f)
            g = Path(d) / "unsafe.zip"
            with zipfile.ZipFile(g, "w") as z:
                z.writestr("../replays/danger.SC2Replay", b"MPQ\x1bbad")
            with self.assertRaisesRegex(ValueError, "Unsafe ZIP"):
                a.inventory_raw_zip(g)

    def test_frozen_id_absent_raw_is_reported_not_silenced(self):
        a = script()
        raw = {"a.SC2Replay": {}, "b.SC2Replay": {}}
        result, only_raw, only_final = a.compare_identities(
            raw, {"b.SC2Replay", "c.SC2Replay"}
        )
        self.assertEqual(only_raw, ["a.SC2Replay"])
        self.assertEqual(only_final, ["c.SC2Replay"])
        self.assertFalse(result["source_minus_final_is_exact_original_816"])


if __name__ == "__main__":
    unittest.main()
