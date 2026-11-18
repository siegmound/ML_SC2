"""S6 contracts: variant schema, frozen hashes, exact replay/window parity."""
from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from sc2proj.sequential.data import IndexedSequences
from sc2proj.sequential.variants import (
    V31_SHA256, V32_SHA256, verify_combatfix_parity,
)
from sc2proj.sequential.windows import validate_schema


def toy_indexes() -> tuple[IndexedSequences, IndexedSequences]:
    rows = pd.DataFrame({
        "replay_id": ["A"] * 10 + ["B"] * 10,
        "time_sec": list(range(0, 150, 15)) * 2,
        "p1_wins": [1] * 10 + [0] * 10,
        "source_row": list(range(20)),
    })
    windows = pd.DataFrame({
        "start_row": [0, 1, 10, 11],
        "stop_row": [9, 10, 19, 20],
        "replay_id": ["A", "A", "B", "B"],
        "time_sec": [120, 135, 120, 135],
        "p1_wins": [1, 1, 0, 0],
    })
    v31 = IndexedSequences(rows, windows, tuple(f"f{i}" for i in range(64)), {
        "source_zip_sha256": V31_SHA256, "window_length": 9, "step_sec": 15,
        "stride": 1, "subset_max_replays": 2,
    })
    v32 = IndexedSequences(rows.copy(), windows.copy(),
        tuple(f"f{i}" for i in range(73)), {
        "source_zip_sha256": V32_SHA256, "window_length": 9, "step_sec": 15,
        "stride": 1, "subset_max_replays": 2,
        "dataset_variant": "v3_2_combatfix",
    })
    return v31, v32


class CombatfixContractTests(unittest.TestCase):
    def test_both_feature_schemas_are_versioned_and_ordered(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = Path(td) / "features.json"
            manifest.write_text(json.dumps({
                "v3_1_predictors": ["f1", "f2"],
                "v3_2_predictors": ["f1", "added", "f2"],
            }))
            metadata = ("replay_id", "time_sec", "p1_wins")
            self.assertEqual(validate_schema(metadata + ("f1", "f2"), manifest), ("f1", "f2"))
            self.assertEqual(validate_schema(metadata + ("f1", "added", "f2"), manifest,
                                             "v3_2_combatfix"), ("f1", "added", "f2"))
            with self.assertRaisesRegex(ValueError, "schema mismatch"):
                validate_schema(metadata + ("f1", "f2", "added"), manifest,
                                "v3_2_combatfix")
            with self.assertRaisesRegex(ValueError, "Unknown dataset variant"):
                validate_schema(metadata + ("f1",), manifest, "unlisted")

    def test_parity_passes_for_identical_timestamps_and_labels(self):
        a, b = toy_indexes()
        result = verify_combatfix_parity(a, b, require_full=False)
        self.assertEqual(result["status"], "S6_DATASET_PARITY_PASS")
        self.assertEqual((result["rows"], result["windows"]), (20, 4))

    def test_frozen_checksums_rejected(self):
        a, b = toy_indexes()
        b.manifest["source_zip_sha256"] = "bad"
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            verify_combatfix_parity(a, b, require_full=False)

    def test_missing_or_wrong_windows_rejected(self):
        a, b = toy_indexes()
        b.windows.loc[0, "time_sec"] = 135
        with self.assertRaisesRegex(ValueError, "time_sec mismatch"):
            verify_combatfix_parity(a, b, require_full=False)
        b.windows.loc[0, "time_sec"] = 120
        b = replace(b, windows=b.windows.iloc[:-1].copy())
        with self.assertRaisesRegex(ValueError, "unequal number"):
            verify_combatfix_parity(a, b, require_full=False)

    def test_row_label_or_source_offset_change_rejected(self):
        a, b = toy_indexes()
        b.ordered_rows.loc[3, "p1_wins"] = 0
        with self.assertRaisesRegex(ValueError, "p1_wins mismatch"):
            verify_combatfix_parity(a, b, require_full=False)
        b.ordered_rows.loc[3, "p1_wins"] = 1
        b.ordered_rows.loc[3, "source_row"] = 999
        with self.assertRaisesRegex(ValueError, "source_row mismatch"):
            verify_combatfix_parity(a, b, require_full=False)

    def test_rejects_wrong_added_features(self):
        a, b = toy_indexes()
        b = replace(b, features=tuple(f"g{i}" for i in range(73)))
        with self.assertRaisesRegex(ValueError, "preserve all 64"):
            verify_combatfix_parity(a, b, require_full=False)

    def test_full_gate_does_not_accept_smoke_indices(self):
        a, b = toy_indexes()
        with self.assertRaisesRegex(ValueError, "subset"):
            verify_combatfix_parity(a, b, require_full=True)

    def test_wrong_window_stride_rejected(self):
        a, b = toy_indexes()
        b.manifest["stride"] = 2
        with self.assertRaisesRegex(ValueError, "stride"):
            verify_combatfix_parity(a, b, require_full=False)


if __name__ == "__main__":
    unittest.main()
