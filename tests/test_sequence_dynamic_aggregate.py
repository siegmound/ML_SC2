"""S13 read-only aggregation guards: no silent checkpoint drift or score corruption."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from sc2proj.sequential.comparison import METRICS, model_metrics
from sc2proj.sequential.variants import V31_SHA256

TARGET = Path(__file__).resolve().parents[1] / "scripts/91_aggregate_dynamic_prefixes.py"
SPEC = importlib.util.spec_from_file_location("s13_aggregate_audit", TARGET)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def small_predictions(more_confident: bool) -> pd.DataFrame:
    rows = []
    for replay, target in (("train-v-0", 0), ("train-v-1", 1)):
        p1 = (0.1 if not target else 0.9) if more_confident else (0.3 if not target else 0.7)
        for j, stage in enumerate((25, 50, 75, 90)):
            rows.append({
                "replay_id": replay, "time_sec": (j + 9) * 15,
                "progress_percent": stage, "y_true": target,
                "y_prob": p1, "y_pred": int(p1 >= 0.5),
            })
    return pd.DataFrame(rows)


class DynamicAggregateTests(unittest.TestCase):
    def setUp(self):
        self.old = small_predictions(False)
        self.new = small_predictions(True)

    def test_valid_read_and_reject_unsafe_probability_or_replay(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "validation_predictions.csv"
            self.old.to_csv(path, index=False)
            self.assertEqual(len(audit.load_predictions(path, 2)), 8)
            bad = self.old.copy()
            bad.loc[0, "y_prob"] = 1.02
            bad.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Invalid"):
                audit.load_predictions(path, 2)
            self.old.iloc[1:].to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                audit.load_predictions(path, 2)

    def _fixture(self, root: Path):
        config = {"hidden_size": 64, "batch_size": 64}
        p9, p13, change = {}, {}, {}
        for stage in (25, 50, 75, 90):
            a = self.old.loc[self.old.progress_percent == stage]
            b = self.new.loc[self.new.progress_percent == stage]
            y = a.y_true.to_numpy(dtype=np.int8)
            former = model_metrics(y, a.y_prob.to_numpy(dtype=float),
                                   a.y_pred.to_numpy(dtype=np.int8))
            current = model_metrics(y, b.y_prob.to_numpy(dtype=float),
                                    b.y_pred.to_numpy(dtype=np.int8))
            p9[str(stage)] = {"models": {"last9": former, "prefix": former}}
            change[str(stage)] = {
                "n_replays": 2, "reference": former, "s13": current,
                "delta_s13_minus_reference": {
                    k: current[k] - former[k] for k in METRICS
                },
                "paired_replay_bootstrap": {
                    "replay_groups": 2, "replicates": 1000,
                    "accuracy_delta": current["accuracy"] - former["accuracy"],
                    "log_loss_delta": current["log_loss"] - former["log_loss"],
                    "accuracy_delta_ci95": [-0.2, 0.2],
                    "log_loss_delta_ci95": [-0.3, 0.3],
                },
            }
        s9 = {
            "schema": "sc2.sequential.prefix_vs_last9.v1",
            "status": "S9_MATCHED_VALIDATION_PASS",
            "source_zip_sha256": V31_SHA256, "seed": 42,
            "train_examples": 8, "validation_examples": 8, "settings": config,
            "validation_coverage": {"balanced_replays": 2},
            "arms": {mode: {"config": config} for mode in ("last9", "prefix")},
            "paired_checkpoint_metrics": p9,
        }
        hist = {
            "status": "S13_FIT_PASS", "seed": 42, "config": config,
            "test_evaluation_policy": "not_accessed_during_training",
            "checkpoint_selection": "validation_bce_only",
            "train_examples_per_epoch": 8, "train_pool_examples": 32,
            "validation_examples": 8, "selected_epoch": 1,
            "best_validation_log_loss": 0.6,
            "history": [{"epoch": 1, "epoch_train_examples": 8,
                         "train_log_loss": 0.61, "validation_log_loss": 0.6}],
        }
        s13 = {
            "schema": "sc2.sequential.dynamic_cutoff_training.v1",
            "status": "S13_DYNAMIC_VALIDATION_PASS",
            "seed": 42, "source_zip_sha256": V31_SHA256,
            "test_evaluation_policy": "not_accessed_during_training",
            "config": config,
            "train_pool": {
                "training_replays": 2, "s9_anchor_examples": 8,
                "expanded_train_examples": 32, "extra_examples": 24,
                "extras_per_replay_min": 12, "extras_per_replay_max": 12,
            },
            "validation_cohort": {"balanced_replays": 2},
            "models": {mode: {**hist, "mode": mode} for mode in ("last9", "prefix")},
            "s13_minus_s9_same_mode": {
                mode: change for mode in ("last9", "prefix")
            },
            "bootstrap": {"reps": 1000},
        }
        source = root / "s9_prefix_vs_last9_seed42_v1"
        target = root / "s13_dynamic_cutoffs_seed42_v1"
        source.mkdir()
        target.mkdir()
        (source / "summary.json").write_text(json.dumps(s9), encoding="utf-8")
        (target / "summary.json").write_text(json.dumps(s13), encoding="utf-8")
        for mode in ("last9", "prefix"):
            self.old.to_csv(source / f"{mode}_validation_predictions.csv", index=False)
            self.new.to_csv(target / f"{mode}_validation_predictions.csv", index=False)
        return target

    def test_audit_accepts_exact_matched_synthetic_predictions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self._fixture(root)
            with patch.dict(audit.TRAIN, {42: 2}), patch.dict(audit.VALIDATION, {42: 2}):
                rows, meta = audit.audit_seed(42, root)
            self.assertEqual(len(rows), 8)
            self.assertEqual(meta["train_examples_per_epoch"], 8)
            self.assertTrue(all(row["delta_accuracy"] == 0 for row in rows))
            self.assertTrue(all(row["delta_log_loss"] < 0 for row in rows))

    def test_audit_rejects_changed_label_even_if_saved_metrics_look_valid(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = self._fixture(root)
            path = target / "prefix_validation_predictions.csv"
            corrupted = pd.read_csv(path)
            corrupted.loc[:3, "y_true"] = 1
            corrupted.to_csv(path, index=False)
            with patch.dict(audit.TRAIN, {42: 2}), patch.dict(audit.VALIDATION, {42: 2}):
                with self.assertRaisesRegex(ValueError, "checkpoint y_true drift"):
                    audit.audit_seed(42, root)

    def test_audit_rejects_modified_heldout_source_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = self._fixture(root)
            path = target / "summary.json"
            d = json.loads(path.read_text())
            d["source_zip_sha256"] = "untrusted"
            path.write_text(json.dumps(d))
            with patch.dict(audit.TRAIN, {42: 2}), patch.dict(audit.VALIDATION, {42: 2}):
                with self.assertRaisesRegex(ValueError, "identity/config mismatch"):
                    audit.audit_seed(42, root)


if __name__ == "__main__":
    unittest.main()
