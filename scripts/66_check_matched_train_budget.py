"""Retrospective paired RF/XGB comparison using existing train-only calibration runs.

Both estimator families were fit on identical train replay groups for each seed;
XGBoost also used validation for early stopping. This is NOT a new untouched
holdout evaluation and does not re-train anything.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, log_loss

SEEDS = (42, 43, 44)
MODELS = {"rf": "rf_calibration", "xgb": "xgb_calibration"}
METRICS = ("accuracy", "balanced_accuracy", "roc_auc", "log_loss")
COLUMNS = ("seed", "model", "fit_partition", "train_rows", "val_rows", "test_rows", *METRICS)


def read_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    expected = {"replay_id", "time_sec", "y_true", "y_prob"}
    if expected - set(frame.columns) or frame.empty:
        raise ValueError(f"Incomplete prediction file: {path}")
    if frame[list(expected)].isna().any().any():
        raise ValueError(f"Missing values in {path}")
    if frame.duplicated(["replay_id", "time_sec"]).any():
        raise ValueError(f"Duplicate snapshots in {path}")
    if not set(frame.y_true.unique()).issubset({0, 1}):
        raise ValueError(f"Non-binary labels in {path}")
    probs = frame.y_prob.to_numpy(dtype=float)
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise ValueError(f"Invalid probabilities in {path}")
    return frame


def observed_scores(frame: pd.DataFrame) -> dict[str, float]:
    y = frame.y_true.to_numpy(dtype=int)
    p = frame.y_prob.to_numpy(dtype=float)
    pred = (p >= .5).astype(int)
    if len(set(y)) != 2:
        raise ValueError("Both classes must be present in the test set")
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p)),
    }


def read_run(repo: Path, dataset: str, model: str, seed: int):
    path = repo / "results" / MODELS[model] / dataset / f"seed_{seed}"
    meta = json.loads((path / "metrics_summary.json").read_text(encoding="utf-8"))
    if int(meta["seed"]) != seed or meta["profile"] != ("no_counter" if model == "rf" else "full"):
        raise ValueError(f"Unexpected archived run identity: {path}")
    validation = read_predictions(path / "validation_predictions.csv")
    test = read_predictions(path / "test_predictions.csv")
    if set(validation.replay_id) & set(test.replay_id):
        raise ValueError(f"Validation-test replay overlap: {path}")
    if len(validation) != meta["n_rows_val"] or len(test) != meta["n_rows_test"]:
        raise ValueError(f"Incorrect prediction row counts: {path}")
    scores = observed_scores(test)
    for name in METRICS:
        if not math.isclose(scores[name], float(meta["test_metrics"][name]), abs_tol=2e-6, rel_tol=0):
            raise ValueError(f"Stored metric mismatch: {path} {name}")
    return meta, validation, test, scores


def aligned_keys(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    fields = ["replay_id", "time_sec", "y_true"]
    l = left[fields].sort_values(fields).reset_index(drop=True)
    r = right[fields].sort_values(fields).reset_index(drop=True)
    return l.equals(r)


def derive(repo: Path, dataset: str = "real_v3_1_fixed") -> str:
    rows = []
    for seed in SEEDS:
        runs = {model: read_run(repo, dataset, model, seed) for model in MODELS}
        rf, xgb = runs["rf"], runs["xgb"]
        for key in ("n_rows_train", "n_rows_val", "n_rows_test"):
            if int(rf[0][key]) != int(xgb[0][key]):
                raise ValueError(f"Unequal fit-data budget {seed}: {key}")
        for section in (1, 2):
            if not aligned_keys(rf[section], xgb[section]):
                raise ValueError(f"RF/XGB prediction rows differ at seed {seed}")
        for model, (meta, _, test, scores) in runs.items():
            row = {"seed": str(seed), "model": model,
                   "fit_partition": "train", "train_rows": str(meta["n_rows_train"]),
                   "val_rows": str(meta["n_rows_val"]), "test_rows": str(len(test))}
            row.update({name: f"{scores[name]:.10f}" for name in METRICS})
            rows.append(row)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset", default="real_v3_1_fixed")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true", help="Compare with committed retrospective evidence")
    modes.add_argument("--write", action="store_true", help="Write a separate B2 evidence table")
    args = parser.parse_args()
    expected = derive(args.repo_root, args.dataset)
    path = args.repo_root / "results/b2_fit_budget/retrospective_train_only.csv"
    if args.write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(expected, encoding="utf-8")
        print(f"Saved {path} from 6 verified historical runs; no model was retrained")
        return 0
    if not path.exists() or path.read_text(encoding="utf-8") != expected:
        print("FAIL: B2 retrospective CSV differs from saved per-seed prediction evidence")
        return 1
    print("PASS: matched train rows, replay and label identities, metrics, and B2 publication")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
