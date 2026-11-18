"""S3: descriptive, endpoint-matched evaluation of frozen v3.1 predictions.

This never trains, calibrates, or changes historical model outputs. The held-out
labels and archived predictions have already been inspected in earlier phases;
these metrics cannot be portrayed as a new, untouched test evaluation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score

KEYS = ("replay_id", "time_sec")
LABEL = "p1_wins"
MODELS = ("rf", "xgb", "mlp")
PARTITIONS = ("val", "test")


def archived_prediction_path(results_root: str | Path, model: str, seed: int, partition: str) -> Path:
    if model not in MODELS or partition not in PARTITIONS or not isinstance(seed, int):
        raise ValueError("Unknown model, seed or partition")
    if model == "rf":
        dirname = "rf_calibration"
    elif model == "xgb":
        dirname = "xgb_calibration"
    else:
        dirname = "deep_final"
    file = "validation_predictions.csv" if partition == "val" else "test_predictions.csv"
    return Path(results_root) / dirname / "real_v3_1_fixed" / f"seed_{seed}" / file


def normalize_integer_times(values: pd.Series) -> np.ndarray:
    times = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(times).all() or (times < 0).any() or (np.abs(times - np.rint(times)) > 1e-6).any():
        raise ValueError("Nonintegral, negative or nonfinite timestamp in predictions")
    if (np.rint(times) % 15 != 0).any():
        raise ValueError("Predictions have a timestamp outside the 15s grid")
    return np.rint(times).astype(np.int64)


def endpoint_frame(index, partitioned, partition: str, common_indexes=()) -> pd.DataFrame:
    """Use the exact S1 sequence endpoints; optionally intersect longer windows.

    The caller must provide S2 validated load_index / load_replay_partitions,
    never derive eligible rows independently from archived predictions.
    """
    if partition not in PARTITIONS:
        raise ValueError("S3 endpoints supported for val and test only")
    windows = index.windows.iloc[partitioned.window_positions[partition]]
    out = windows.loc[:, [*KEYS, LABEL]].copy()
    out["time_sec"] = normalize_integer_times(out["time_sec"])
    if out.duplicated(list(KEYS)).any():
        raise ValueError("Duplicate S1 endpoint keys")
    for other in common_indexes:
        additional = other.windows[[*KEYS, LABEL]].copy()
        additional["time_sec"] = normalize_integer_times(additional["time_sec"])
        if additional.duplicated(list(KEYS)).any():
            raise ValueError("Duplicate common-index endpoint keys")
        out = out.merge(additional, how="inner", on=list(KEYS), validate="one_to_one", suffixes=("", "_other"))
        if not np.array_equal(out[LABEL].to_numpy(), out[f"{LABEL}_other"].to_numpy()):
            raise ValueError("Inconsistent labels between S1 indexes")
        out = out.drop(columns=[f"{LABEL}_other"])
    if out.empty:
        raise ValueError("No common sequence endpoints in the selected partition")
    return out.sort_values(list(KEYS), kind="stable").reset_index(drop=True)


def load_archived_predictions(path: str | Path, expected_groups: frozenset[str]) -> pd.DataFrame:
    source = Path(path)
    frame = pd.read_csv(source, dtype={"replay_id": "string"})
    required = {*KEYS, "y_true", "y_prob", "y_pred"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing prediction fields in {source}")
    frame = frame.loc[:, [*KEYS, "y_true", "y_prob", "y_pred"]].copy()
    if frame.isna().any().any():
        raise ValueError(f"Missing prediction values in {source}")
    if not frame["replay_id"].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError("Invalid replay ID in archived prediction file")
    frame["time_sec"] = normalize_integer_times(frame["time_sec"])
    if frame.duplicated(list(KEYS)).any():
        raise ValueError(f"Duplicate prediction endpoint in {source}")
    if not frame["replay_id"].isin(expected_groups).all():
        raise ValueError(f"Archived predictions include replay(s) outside selected split: {source}")
    for label in ("y_true", "y_pred"):
        numeric = pd.to_numeric(frame[label], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all() or not np.isin(numeric, [0, 1]).all():
            raise ValueError(f"Nonbinary {label} in {source}")
        frame[label] = numeric.astype(np.int8)
    probs = pd.to_numeric(frame["y_prob"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(probs).all() or ((probs < 0) | (probs > 1)).any():
        raise ValueError(f"Invalid probabilities in {source}")
    frame["y_prob"] = probs
    return frame


def match_one_model(endpoints: pd.DataFrame, predictions: pd.DataFrame, model: str) -> pd.DataFrame:
    """Strict 1:1 endpoint inclusion and label parity; extra early rows allowed."""
    if model not in MODELS:
        raise ValueError(f"Unknown model {model}")
    merged = endpoints.merge(predictions, on=list(KEYS), how="left", validate="one_to_one", indicator=True)
    if not merged["_merge"].eq("both").all():
        n_missing = int(merged["_merge"].eq("left_only").sum())
        raise ValueError(f"{model}: {n_missing} S1 endpoints missing archived predictions")
    if not np.array_equal(merged[LABEL].to_numpy(dtype=np.int8), merged["y_true"].to_numpy(dtype=np.int8)):
        raise ValueError(f"{model}: archived labels disagree with S1 target")
    return merged.loc[:, [*KEYS, LABEL, "y_prob", "y_pred"]].rename(columns={
        "y_prob": f"{model}_prob", "y_pred": f"{model}_pred",
    })


def compare_archived_baselines(index, partitioned, partition: str, results_root: str | Path,
                               models: tuple[str, ...] = MODELS, common_indexes=()) -> tuple[dict, pd.DataFrame]:
    """Return a JSON-safe report and aligned prediction columns; no writes."""
    if not models or len(set(models)) != len(models) or not set(models).issubset(MODELS):
        raise ValueError("Models must be a nonempty unique subset of rf, xgb, mlp")
    endpoints = endpoint_frame(index, partitioned, partition, common_indexes)
    matched = endpoints.copy()
    metrics = {}
    for model in models:
        path = archived_prediction_path(results_root, model, partitioned.seed, partition)
        archived = load_archived_predictions(path, partitioned.groups[partition])
        one = match_one_model(endpoints, archived, model)
        matched = matched.merge(one.drop(columns=[LABEL]), on=list(KEYS), validate="one_to_one")
        y = matched[LABEL].to_numpy(dtype=np.int8)
        probs = matched[f"{model}_prob"].to_numpy(dtype=np.float64)
        preds = matched[f"{model}_pred"].to_numpy(dtype=np.int8)
        metrics[model] = {
            "accuracy": float(accuracy_score(y, preds)),
            "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
            "roc_auc": float(roc_auc_score(y, probs)) if len(np.unique(y)) == 2 else None,
            "log_loss": float(log_loss(y, probs, labels=[0, 1])),
            "n_endpoints": int(len(y)),
            "source_prediction_csv": str(path),
            "fitting_protocol": "train_only_archived" if model in ("rf", "xgb") else "historical_deep_selection",
        }
    report = {
        "schema": "sc2.sequential.endpoint_baselines.v1",
        "status": "S3_MATCHED_ENDPOINTS_PASS",
        "seed": int(partitioned.seed),
        "partition": partition,
        "index_window_length": int(index.manifest["window_length"]),
        "common_window_lengths": [int(index.manifest["window_length"])] + [int(x.manifest["window_length"]) for x in common_indexes],
        "replays": int(endpoints.replay_id.nunique()),
        "endpoints": int(len(endpoints)),
        "models": metrics,
        "interpretation": "Retrospective historical scores; no new untouched holdout, no model selection on test.",
        "cautions": ["RF/XGB fit train-only, but XGB early stopping uses validation.",
                     "Historical deep candidate tests were previously inspected; not a newly blinded comparison.",
                     "Adjacent windows from the same replay are correlated; uncertainty must be replay-aware."],
    }
    return report, matched
