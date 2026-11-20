#!/usr/bin/env python3
"""V4 independent post-hoc integrity verifier (not a training/selection step).

Confirms SHA-pinned raw-derived dataset, replay-disjoint split membership,
all test prediction rows/labels/probabilities, reported metrics and
per-game outcomes. Outputs a NEW evidence folder; does not tune models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss,
    log_loss, roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sc2proj.utils import load_dataframe_from_zip


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, description):
    if not condition:
        raise ValueError(description)


def metric_values(y, prob):
    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype=float)
    require(len(y) > 0 and set(y) == {0, 1}, "Both test outcome classes required")
    require(np.isfinite(prob).all() and ((prob >= 0) & (prob <= 1)).all(),
            "Test contains nonfinite or nonprobability scores")
    pred = (prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "log_loss": float(log_loss(y, prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y, prob)),
    }


def verify_predictions(predictions: pd.DataFrame, expected: pd.DataFrame,
                       report_metrics: dict) -> dict:
    keys = ["replay_id", "time_sec"]
    required = set(keys + ["y_true", "y_prob", "y_pred"])
    require(required <= set(predictions.columns), "Incomplete test prediction columns")
    require(not predictions.duplicated(keys).any(), "Duplicate test prediction keys")
    require(not expected.duplicated(keys).any(), "Duplicate expected raw-snapshot keys")
    require(len(predictions) == len(expected), "Test prediction count differs from held-out source")
    comparison = predictions.merge(
        expected[keys + ["p1_wins"]], how="outer",
        on=keys, indicator=True, validate="one_to_one",
    )
    require((comparison["_merge"] == "both").all(),
            "Predictions omit or add replay/time snapshots")
    require(predictions["y_true"].isin([0, 1]).all(), "Invalid test labels")
    require((comparison["y_true"].astype(int) == comparison["p1_wins"].astype(int)).all(),
            "Test label disagrees with original raw-derived label")
    y = comparison["y_true"].to_numpy(dtype=int)
    probabilities = comparison["y_prob"].to_numpy(dtype=float)
    require(np.isfinite(probabilities).all(), "Nonfinite test probabilities")
    require(((probabilities >= 0) & (probabilities <= 1)).all(),
            "Test probabilities are out of [0,1]")
    require((comparison["y_pred"].to_numpy(dtype=int) ==
             (probabilities >= 0.5).astype(int)).all(),
            "Saved y_pred disagrees with 0.5 decision threshold")
    measured = metric_values(y, probabilities)
    for field in ("accuracy", "balanced_accuracy", "roc_auc", "log_loss", "brier"):
        if field in report_metrics:
            require(np.isclose(measured[field], report_metrics[field],
                               rtol=1e-6, atol=1e-8),
                    f"Recomputed {field} disagrees with saved metrics")
    match = comparison.groupby("replay_id", sort=True).agg(
        y_true=("y_true", "first"), unique_labels=("y_true", "nunique"),
        y_prob=("y_prob", "mean"), snapshots=("time_sec", "size")
    ).reset_index()
    require((match.unique_labels == 1).all(), "A replay has inconsistent saved outcomes")
    macro = metric_values(match.y_true.to_numpy(), match.y_prob.to_numpy())
    return {
        "n_test_snapshots": len(comparison),
        "n_test_replays": len(match),
        "snapshot_metrics": measured,
        "replay_mean_probability_metrics": macro,
        "n_min_max_snapshots_per_replay": [
            int(match.snapshots.min()), int(match.snapshots.max())
        ],
    }


def run(dataset_dir: Path, suite_root: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Refusing overwrite of verification report: {output_dir}")
    source = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    protocol = json.loads((suite_root / "protocol.json").read_text(encoding="utf-8"))
    require(source.get("schema") == "ml_sc2.v4.raw_replay_to_dataset.v1",
            "Not a raw-derived v4 dataset")
    require(protocol.get("schema") == "ml_sc2.v4.rebuilt_training_suite.v1",
            "Not a v4 training suite")
    require(source.get("historical_frozen_data_used_for_build") is False,
            "Frozen source contamination not excluded")
    require(protocol.get("original_frozen_training_data") is False,
            "Training output claims original frozen source")
    require(
        protocol["data_manifest_sha256"] == digest(dataset_dir / "manifest.json"),
        "Training suite uses a different v4 raw manifest"
    )
    require(
        source["dataset_zip_sha256"] == digest(dataset_dir / "sc2_v4_clean.zip")
        == protocol["data_zip_sha256"], "V4 raw dataset bytes changed"
    )
    profile_path = suite_root / "profiles" / f"sc2_v4_{protocol['profile']}.zip"
    require(digest(profile_path) == protocol["profile_zip_sha256"],
            "Training data profile SHA-256 mismatch")
    raw = load_dataframe_from_zip(dataset_dir / "sc2_v4_clean.zip")
    profile = load_dataframe_from_zip(profile_path)
    require(
        profile[["replay_id", "time_sec", "p1_wins"]].equals(
            raw[["replay_id", "time_sec", "p1_wins"]]),
        "Profile silently changed raw replay/time/label membership"
    )
    summaries = []
    for seed in protocol["seeds"]:
        split_path = suite_root / "splits" / f"v4_clean_split_seed_{seed}.json"
        split = json.loads(split_path.read_text(encoding="utf-8"))
        group_sets = [
            set(split[key]) for key in ("train_groups", "val_groups", "test_groups")
        ]
        require(
            not any(group_sets[i] & group_sets[j]
                    for i in range(3) for j in range(i+1, 3)),
            "Replay leakage across train/val/test groups"
        )
        require(
            set.union(*group_sets) == set(raw["replay_id"].astype(str)),
            "Train/val/test groups do not partition all raw replay identities"
        )
        expected = raw[raw["replay_id"].isin(group_sets[2])]
        for model in protocol["models"]:
            if model == "deep":
                run_dir = suite_root / "deep_finalist/v4_clean_raw" / f"seed_{seed}"
                filenames = [("selected", "test_predictions.csv")]
                meta_path = run_dir / "metrics_summary.json"
                require(meta_path.is_file(), f"Missing deep training metrics: {meta_path}")
                reports = json.loads(meta_path.read_text(encoding="utf-8"))
                metrics = reports["test_metrics"]
            else:
                run_dir = suite_root / "four_way/v4_clean_raw" / model / f"seed_{seed}"
                meta_path = run_dir / "metrics_summary.json"
                require(meta_path.is_file(), f"Missing model training metrics: {meta_path}")
                reports = json.loads(meta_path.read_text(encoding="utf-8"))
                require(reports["data_origin"] == "v4_raw_rebuild",
                        "Run report does not identify independent raw dataset")
                require(reports["pristine_blind_test_attested"] is False,
                        "Spurious blind-test attestation")
                filenames = [
                    (name, f"test_predictions_{name}.csv")
                    for name in ("uncalibrated", "platt", "isotonic")
                ]
            for variant, filename in filenames:
                path = run_dir / filename
                require(path.is_file(), f"Missing test predictions: {path}")
                pred = pd.read_csv(path)
                saved = metrics if model == "deep" else reports["scores"][variant]
                summary = verify_predictions(pred, expected, saved)
                summaries.append({
                    "seed": seed, "model": model, "variant": variant,
                    "predictions_sha256": digest(path),
                    **summary,
                })
    result = {
        "schema": "ml_sc2.v4.verified_prediction_integrity.v1",
        "dataset_zip_sha256": digest(dataset_dir / "sc2_v4_clean.zip"),
        "suite_protocol_sha256": digest(suite_root / "protocol.json"),
        "profile_sha256": digest(profile_path),
        "predictions_verified": len(summaries),
        "evaluations": summaries,
        "independent_game_aggregation": "mean of snapshot probabilities per replay; threshold at 0.5",
        "historical_b3_gate_closed": False,
        "pristine_blind_test_attested": False,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame([{
        "seed": x["seed"], "model": x["model"], "variant": x["variant"],
        "snapshots": x["n_test_snapshots"], "replays": x["n_test_replays"],
        "snapshot_accuracy": x["snapshot_metrics"]["accuracy"],
        "replay_accuracy": x["replay_mean_probability_metrics"]["accuracy"],
        "snapshot_log_loss": x["snapshot_metrics"]["log_loss"],
        "replay_log_loss": x["replay_mean_probability_metrics"]["log_loss"],
    } for x in summaries]).to_csv(output_dir / "summary.csv", index=False)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", required=True, type=Path)
    p.add_argument("--suite-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    a = p.parse_args()
    report = run(a.dataset_dir.resolve(), a.suite_root.resolve(),
                 a.output_dir.resolve())
    print(json.dumps({
        "schema": report["schema"],
        "predictions_verified": report["predictions_verified"],
        "output": str(a.output_dir)
    }, indent=2))


if __name__ == "__main__":
    main()
