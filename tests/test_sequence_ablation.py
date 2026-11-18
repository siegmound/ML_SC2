"""S7 regression tests: corrected common64 view must preserve provenance and keys."""
from __future__ import annotations

import unittest
from dataclasses import replace

from sc2proj.sequential.variants import (
    V31_SHA256, V32_SHA256, combatfix_common64_view,
    verify_combatfix_parity,
)
from sc2proj.sequential.data import IndexedSequences
import pandas as pd


def make_indexes():
    common = tuple(f"f{i}" for i in range(64))
    # The added columns can be interspersed, as in the actual v3.2 CSV.
    extended = common[:4] + ("extra_a", "extra_b", "extra_c") + common[4:32] + (
        "extra_d", "extra_e", "extra_f", "extra_g", "extra_h", "extra_i"
    ) + common[32:]
    rows = pd.DataFrame({
        "replay_id": ["A"] * 9 + ["B"] * 9,
        "time_sec": [15 * i for i in range(9)] * 2,
        "p1_wins": [0] * 9 + [1] * 9,
        "source_row": list(range(18)),
    })
    windows = pd.DataFrame({
        "start_row": [0, 9], "stop_row": [9, 18],
        "replay_id": ["A", "B"], "time_sec": [120, 120],
        "p1_wins": [0, 1],
    })
    ref = IndexedSequences(rows, windows, common, {
        "source_zip_sha256": V31_SHA256, "window_length": 9,
        "step_sec": 15, "stride": 1, "features": list(common),
        "feature_count": 64, "subset_max_replays": 2,
    })
    cand = IndexedSequences(rows.copy(), windows.copy(), extended, {
        "source_zip_sha256": V32_SHA256, "window_length": 9,
        "step_sec": 15, "stride": 1, "features": list(extended),
        "feature_count": 73, "subset_max_replays": 2,
        "dataset_variant": "v3_2_combatfix",
    })
    return ref, cand


class Common64AblationTests(unittest.TestCase):
    def test_corrected_view_keeps_exact_64_feature_order_and_source_hash(self):
        ref, cand = make_indexes()
        self.assertEqual(verify_combatfix_parity(ref, cand, require_full=False)["status"],
                         "S6_DATASET_PARITY_PASS")
        view = combatfix_common64_view(ref, cand)
        self.assertEqual(view.features, ref.features)
        self.assertEqual(view.manifest["features"], list(ref.features))
        self.assertEqual(view.manifest["feature_count"], 64)
        self.assertEqual(view.manifest["source_zip_sha256"], V32_SHA256)
        self.assertEqual(view.manifest["feature_view"], "v3_2_common64")
        self.assertIs(view.windows, cand.windows)
        self.assertIs(view.ordered_rows, cand.ordered_rows)
        # View creation does not mutate the original v3.2 S1 manifest.
        self.assertEqual(cand.manifest["feature_count"], 73)
        self.assertEqual(len(cand.features), 73)

    def test_only_common_columns_and_exactly_nine_extra_are_allowed(self):
        ref, cand = make_indexes()
        wrong = replace(cand, features=("missing",) + cand.features[1:])
        with self.assertRaisesRegex(ValueError, "nine extra"):
            combatfix_common64_view(ref, wrong)
        wrong2 = replace(cand, features=cand.features[:-1])
        with self.assertRaisesRegex(ValueError, "64 reference and 73"):
            combatfix_common64_view(ref, wrong2)

    def test_view_is_shallow_and_never_rewrites_original_dataset(self):
        ref, cand = make_indexes()
        view = combatfix_common64_view(ref, cand)
        view.manifest["feature_count"] = 99
        self.assertEqual(cand.manifest["feature_count"], 73)
        self.assertEqual(ref.manifest["feature_count"], 64)

    def test_refuse_comparison_if_windows_differ(self):
        ref, cand = make_indexes()
        wrong = replace(cand, windows=cand.windows.copy())
        wrong.windows.loc[0, "time_sec"] = 135
        with self.assertRaisesRegex(ValueError, "time_sec mismatch"):
            verify_combatfix_parity(ref, wrong, require_full=False)


if __name__ == "__main__":
    unittest.main()
