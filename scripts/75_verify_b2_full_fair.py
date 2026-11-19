"""Verify B2 FAIR full-scale experiments without modifying archived evidence.

By default all nine runs must be present. --seed and --model restrict the scope
for staged verification. Full numerical test metrics are recomputed from the
saved predictions, so a mere existence check cannot mark a run as complete.
Historical test partitions were previously inspected: not a blind-test claim.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, log_loss, roc_auc_score,
)

SEEDS = (42, 43, 44)
MODELS = ("rf", "xgb", "deep")
CALIBRATION_METHODS = ("uncalibrated", "platt", "isotonic")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path: Path):
    require(path.is_file(), f"Missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_predictions(path: Path):
    require(path.is_file() and path.stat().st_size > 0, f"Missing predictions: {path}")
    df = pd.read_csv(path, dtype={"replay_id": str})
    require({"replay_id", "time_sec", "y_true", "y_prob", "y_pred"} <= set(df),
            f"Invalid prediction schema: {path}")
    require(not df.empty and not df[["replay_id", "time_sec", "y_true", "y_prob", "y_pred"]].isna().any().any(),
            f"Empty/NaN predictions: {path}")
    require(not df.duplicated(["replay_id", "time_sec"]).any(), f"Duplicate replay/time: {path}")
    require(set(df["y_true"].unique()) == {0, 1}, f"Invalid or single-class target: {path}")
    p = df["y_prob"].to_numpy(dtype=float)
    require(np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all(),
            f"Invalid probabilities: {path}")
    require(np.array_equal((p >= 0.5).astype(int), df["y_pred"].to_numpy(dtype=int)),
            f"Predicted labels disagree with probabilities: {path}")
    return df


def scores(df, calibration: bool, *, probability_dtype=np.float64):
    y = df["y_true"].to_numpy(dtype=int)
    p = df["y_prob"].to_numpy(dtype=probability_dtype)
    pred = (p >= 0.5).astype(int)
    result = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p)),
    }
    if calibration:
        bins = np.digitize(p, np.linspace(0, 1, 16)[1:-1], right=True)
        gaps = []
        ece = 0.0
        for i in range(15):
            mask = bins == i
            if mask.any():
                gap = abs(float(p[mask].mean() - y[mask].mean()))
                gaps.append(gap)
                ece += float(mask.mean()) * gap
        result.update({
            "brier": float(brier_score_loss(y, p)),
            "ece_15": float(ece),
            "mce_15": float(max(gaps, default=0.0)),
        })
    else:
        result["confusion_matrix"] = confusion_matrix(y, pred).tolist()
    return result


def check_scores(actual, recorded, path):
    for key, value in actual.items():
        require(key in recorded, f"Missing {key} in {path}")
        if isinstance(value, list):
            require(value == recorded[key], f"Confusion matrix mismatch in {path}")
        else:
            require(math.isclose(value, float(recorded[key]), abs_tol=1e-8, rel_tol=1e-8),
                    f"Metric {key} mismatch in {path}: {value} vs {recorded[key]}")


def check_orig_split(root, seed):
    path = root / "data/processed/splits" / f"real_v3_1_fixed_split_seed_{seed}.json"
    original = read_json(path)
    require(int(original["seed"]) == seed, f"Original split seed mismatch: {path}")
    groups = {key: set(map(str, original[f"{key}_groups"])) for key in ("train", "val", "test")}
    require(all(groups.values()), f"Empty original replay groups for seed {seed}")
    require(all(len(groups[k]) == len(original[f"{k}_groups"]) for k in groups),
            f"Duplicate original replay ID in seed {seed}")
    require(not (groups["train"] & groups["val"] or
                 groups["train"] & groups["test"] or
                 groups["val"] & groups["test"]), f"Leaking original split seed {seed}")
    return groups


def check_test_partition(df, original, count, run):
    require(len(df) == count, f"{run}: wrong test row count")
    require(set(df["replay_id"].astype(str).unique()) == original["test"],
            f"{run}: test replay IDs differ from frozen original split")


def compare_rows(left, right, where):
    keys = ["replay_id", "time_sec", "y_true"]
    require(len(left) == len(right), f"{where}: unequal prediction length")
    require(left[keys].equals(right[keys]), f"{where}: replay/time/label mismatch")


def verify_one(root: Path, base: Path, model: str, seed: int, original: dict):
    if model == "deep":
        run = base / "deep_finalist/real_v3_1_fixed" / f"seed_{seed}"
        summary = read_json(run / "metrics_summary.json")
        require(summary.get("seed") == seed and summary.get("model") == "deep_final",
                f"{run}: wrong seed/model")
        require(summary.get("device") == "cuda", f"{run}: GPU requested but run used CPU")
        require(summary.get("test_evaluation_policy") == "selected_candidate_only",
                f"{run}: invalid test evaluation policy")
        for file in ("artifacts_manifest.json", "config_used.json", "selected_model_state.pt",
                     "selected_scaler.joblib", "best_training_history.csv"):
            require((run / file).is_file() and (run / file).stat().st_size > 0,
                    f"{run}: missing {file}")
        config = read_json(run / "config_used.json")
        require(config.get("test_evaluation_policy") == "selected_candidate_only",
                f"{run}: config test-use policy mismatch")
        search = pd.read_csv(run / "validation_search_results.csv")
        require(len(search) >= 2 and not any(c.startswith("test_") for c in search.columns),
                f"{run}: candidate search contains test scores or too few candidates")
        require(summary.get("best_candidate") in set(search["candidate_name"]),
                f"{run}: selected candidate absent from validation search")
        val = read_predictions(run / "validation_predictions.csv")
        test = read_predictions(run / "test_predictions.csv")
        require(len(val) == summary["n_rows_val"], f"{run}: validation row count mismatch")
        require(set(val.replay_id.unique()) == original["val"],
                f"{run}: validation replay set mismatch")
        check_test_partition(test, original, summary["n_rows_test"], run)
        check_scores(scores(val, False), summary["validation_metrics"], run)
        check_scores(scores(test, False), summary["test_metrics"], run)
        return run, test

    run = base / "strict_calibration/real_v3_1_fixed" / model / f"seed_{seed}"
    summary = read_json(run / "metrics_summary.json")
    plan = read_json(run / "split_manifest.json")
    require(summary.get("seed") == seed and summary.get("model") == model,
            f"{run}: wrong seed/model")
    require(summary.get("protocol") == "B2_four_way_replay_disjoint" and
            summary.get("historical_test_previously_inspected") is True,
            f"{run}: invalid protocol disclosure")
    require(plan.get("seed") == seed, f"{run}: four-way seed mismatch")
    groups = {p: set(map(str, plan[f"{p}_groups"]))
              for p in ("train", "tune", "calibration", "test")}
    require(all(groups.values()), f"{run}: empty four-way replay group")
    require(all(len(groups[p]) == len(plan[f"{p}_groups"]) for p in groups),
            f"{run}: duplicate replay IDs")
    require(not any(groups[a] & groups[b] for i, a in enumerate(groups)
                    for b in tuple(groups)[i + 1:]), f"{run}: replay overlap")
    require(groups["train"] == original["train"] and
            groups["tune"] | groups["calibration"] == original["val"] and
            groups["test"] == original["test"], f"{run}: original split drift")
    require(summary["training"]["fit_groups"] == "train" and
            summary["training"]["calibration_groups"] == "calibration",
            f"{run}: wrong fitting partitions")
    if model == "xgb":
        require(summary["training"]["selection_groups"] == "tune",
                f"{run}: XGB early-stopping split wrong")
    columns = read_json(run / "feature_names.json")
    require(len(columns) == summary["n_features"], f"{run}: feature count mismatch")
    require((model != "rf" or all("counter" not in c.lower() for c in columns)),
            f"{run}: RF no_counter profile violated")
    baseline = None
    for method in CALIBRATION_METHODS:
        frame = read_predictions(run / f"test_predictions_{method}.csv")
        check_test_partition(frame, original, summary["partition_rows"]["test"], run)
        # XGBoost emits raw (uncalibrated) probabilities as float32. CSV
        # loading otherwise promotes them to float64; sklearn log_loss/ECE
        # can then differ from the originally saved float32-based metrics
        # by a few 1e-8. Calibrated Platt/isotonic outputs use float64.
        probability_dtype = np.float32 if model == "xgb" and method == "uncalibrated" else np.float64
        check_scores(scores(frame, True, probability_dtype=probability_dtype),
                     summary["scores"][method], run)
        if baseline is not None:
            compare_rows(baseline, frame, run)
        else:
            baseline = frame
    return run, baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-base", type=Path, default=Path("results/b2_fullscale_audit_20261119"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--model", choices=MODELS)
    args = parser.parse_args()
    root, base = args.repo_root.resolve(), args.output_base.resolve()
    seeds = (args.seed,) if args.seed else SEEDS
    models = (args.model,) if args.model else MODELS
    for seed in seeds:
        original = check_orig_split(root, seed)
        baseline = None
        for model in models:
            path, predictions = verify_one(root, base, model, seed, original)
            if baseline is not None:
                compare_rows(baseline, predictions, f"seed {seed}: cross-model test identity")
            else:
                baseline = predictions
            print(f"PASS B2 {model} seed_{seed}: {path}")
    print(f"PASS B2 verified {len(seeds) * len(models)} run(s); retrospective test was previously inspected")


if __name__ == "__main__":
    main()
