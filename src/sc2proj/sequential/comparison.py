"""S5: strict, retrospective comparison of S4 GRU with archived S3 baselines.

Validation-only. Replay-level paired bootstrap provides conditional, descriptive
intervals; it does not erase validation reuse or prior test inspection.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score

from .baselines import compare_archived_baselines, normalize_integer_times

MODELS = ("rf", "xgb", "mlp", "gru")
BASELINES = ("rf", "xgb", "mlp")
METRICS = ("accuracy", "balanced_accuracy", "roc_auc", "log_loss", "brier_score")
EXPECTED_SEEDS = (42, 43, 44)
VALIDATION_ENDPOINTS = {42: 112312, 43: 112943, 44: 111899}
REQUIRED_COLUMNS = ("replay_id", "time_sec", "y_true", "y_prob", "y_pred")


def model_metrics(labels: np.ndarray, probs: np.ndarray, preds: np.ndarray) -> dict:
    """Point estimates; every model scored on the same paired validation rows."""
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "roc_auc": float(roc_auc_score(labels, probs)) if len(np.unique(labels)) == 2 else None,
        "log_loss": float(log_loss(labels, np.clip(probs, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "brier_score": float(brier_score_loss(labels, probs)),
    }


def load_verified_gru(run_dir: str | Path, seed: int, index, matched: pd.DataFrame) -> pd.DataFrame:
    """GRU CSV must exactly equal all S1/S3 endpoints, never an arbitrary subset."""
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    required = {
        "status": "S4_VALIDATION_PASS", "seed": seed, "smoke": False,
        "window_length": 9, "feature_count": len(index.features),
        "source_zip_sha256": index.manifest["source_zip_sha256"],
        "test_evaluation_policy": "not_accessed_during_training",
        "checkpoint_selection": "validation_bce_only",
    }
    for key, value in required.items():
        if summary.get(key) != value:
            raise ValueError(f"S4 summary {key} mismatch for seed {seed}")
    if int(index.manifest["window_length"]) != 9:
        raise ValueError("S5 only supports S4 9-snapshot experiment")
    if summary.get("dataset") != "v3_1_fixed":
        raise ValueError("S4 dataset variant mismatch")
    pred = pd.read_csv(run_dir / "validation_predictions.csv", dtype={"replay_id": "string"})
    if not set(REQUIRED_COLUMNS).issubset(pred.columns):
        raise ValueError("Missing GRU prediction columns")
    pred = pred.loc[:, list(REQUIRED_COLUMNS)].copy()
    if pred.empty or pred.isna().any().any():
        raise ValueError("Empty GRU predictions or missing values")
    pred["time_sec"] = normalize_integer_times(pred["time_sec"])
    for key in ("y_true", "y_pred"):
        x = pd.to_numeric(pred[key], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(x).all() or not np.isin(x, [0, 1]).all():
            raise ValueError(f"Invalid GRU {key}")
        pred[key] = x.astype(np.int8)
    prob = pd.to_numeric(pred["y_prob"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(prob).all() or ((prob < 0) | (prob > 1)).any():
        raise ValueError("Invalid GRU probabilities")
    pred["y_prob"] = prob
    if not np.array_equal(pred["y_pred"].to_numpy(dtype=np.int8), (prob >= 0.5).astype(np.int8)):
        raise ValueError("GRU predictions disagree with 0.5 threshold")
    if pred.duplicated(["replay_id", "time_sec"]).any():
        raise ValueError("Duplicate GRU endpoint")
    pred = pred.sort_values(["replay_id", "time_sec"], kind="stable").reset_index(drop=True)
    expected = matched.loc[:, ["replay_id", "time_sec", "p1_wins"]].sort_values(
        ["replay_id", "time_sec"], kind="stable"
    ).reset_index(drop=True)
    if (len(pred) != len(expected) or
            not np.array_equal(pred["replay_id"].astype(str).to_numpy(), expected["replay_id"].astype(str).to_numpy()) or
            not np.array_equal(pred["time_sec"].to_numpy(dtype=np.int64), expected["time_sec"].to_numpy(dtype=np.int64))):
        raise ValueError("GRU endpoint coverage mismatch: missing/extra/wrong replay or timestamp")
    if not np.array_equal(pred["y_true"].to_numpy(dtype=np.int8), expected["p1_wins"].to_numpy(dtype=np.int8)):
        raise ValueError("GRU labels disagree with S1/S3")
    if summary.get("validation_windows") != len(pred) or summary.get("validation_metrics", {}).get("n_endpoints") != len(pred):
        raise ValueError("GRU summary window counts differ from predictions")
    if summary["validation_metrics"].get("n_replays") != int(pred["replay_id"].nunique()):
        raise ValueError("GRU summary replay count differs from predictions")
    estimates = model_metrics(pred.y_true.to_numpy(dtype=np.int8), prob, pred.y_pred.to_numpy(dtype=np.int8))
    for metric in ("accuracy", "balanced_accuracy", "roc_auc", "log_loss"):
        stored = summary["validation_metrics"].get(metric)
        if stored is None or not np.isclose(stored, estimates[metric], atol=2e-6, rtol=0):
            raise ValueError(f"GRU summary {metric} inconsistent with saved prediction CSV")
    return pred


def paired_replay_bootstrap(matched: pd.DataFrame, baseline: str, *, reps: int = 1000,
                            seed: int = 2026) -> dict:
    """Paired delta intervals via whole-replay resampling, weighted by snapshot counts.

    No row-wise bootstrap and no cross-seed pooling. Positive accuracy delta and
    negative log-loss delta favor GRU. These are conditional exploratory intervals.
    """
    if baseline not in BASELINES or reps < 0:
        raise ValueError("Invalid baseline or negative bootstrap repetitions")
    y = matched["p1_wins"].to_numpy(dtype=np.int8)
    if len(y) == 0:
        raise ValueError("No matched predictions")
    groups, replay_names = pd.factorize(matched["replay_id"], sort=False)
    n = len(replay_names)
    if n == 0:
        raise ValueError("No replay groups")
    grp_counts = np.bincount(groups, minlength=n)
    gru_pred = matched["gru_pred"].to_numpy(dtype=np.int8)
    ref_pred = matched[f"{baseline}_pred"].to_numpy(dtype=np.int8)
    gru_p = np.clip(matched["gru_prob"].to_numpy(dtype=np.float64), 1e-7, 1 - 1e-7)
    ref_p = np.clip(matched[f"{baseline}_prob"].to_numpy(dtype=np.float64), 1e-7, 1 - 1e-7)
    acc_diff = (gru_pred == y).astype(np.float64) - (ref_pred == y).astype(np.float64)
    def row_log_loss(prob):
        return -(y * np.log(prob) + (1 - y) * np.log1p(-prob))
    ll_diff = row_log_loss(gru_p) - row_log_loss(ref_p)
    grp_acc = np.bincount(groups, weights=acc_diff, minlength=n)
    grp_ll = np.bincount(groups, weights=ll_diff, minlength=n)
    observed_acc = float(acc_diff.mean())
    observed_ll = float(ll_diff.mean())
    if reps == 0:
        return {
            "replay_groups": n, "replicates": 0,
            "accuracy_delta": observed_acc, "log_loss_delta": observed_ll,
            "accuracy_delta_ci95": None, "log_loss_delta_ci95": None,
        }
    rng = np.random.default_rng(seed)
    acc_samples = np.empty(reps, dtype=np.float64)
    ll_samples = np.empty(reps, dtype=np.float64)
    for start in range(0, reps, 128):
        end = min(start + 128, reps)
        draws = rng.integers(0, n, size=(end - start, n))
        denominator = grp_counts[draws].sum(axis=1)
        acc_samples[start:end] = grp_acc[draws].sum(axis=1) / denominator
        ll_samples[start:end] = grp_ll[draws].sum(axis=1) / denominator
    return {
        "replay_groups": n, "replicates": reps,
        "accuracy_delta": observed_acc, "log_loss_delta": observed_ll,
        "accuracy_delta_ci95": [float(x) for x in np.quantile(acc_samples, [0.025, 0.975])],
        "log_loss_delta_ci95": [float(x) for x in np.quantile(ll_samples, [0.025, 0.975])],
    }


def compare_one_seed(index, partitioned, results_root: str | Path,
                     gru_run_dir: str | Path, *, reps: int = 1000,
                     bootstrap_seed: int = 2026) -> tuple[dict, pd.DataFrame]:
    if int(partitioned.seed) not in EXPECTED_SEEDS:
        raise ValueError("S5 frozen multi-seed comparison limited to 42/43/44")
    report, matched = compare_archived_baselines(index, partitioned, "val", results_root)
    seed = int(partitioned.seed)
    if report["status"] != "S3_MATCHED_ENDPOINTS_PASS" or len(matched) != VALIDATION_ENDPOINTS[seed]:
        raise ValueError(f"S3 endpoint count/integrity mismatch for seed {seed}")
    gru = load_verified_gru(gru_run_dir, seed, index, matched)
    # Both are sorted by the same validated keys; preserve the one-to-one mapping.
    matched = matched.sort_values(["replay_id", "time_sec"], kind="stable").reset_index(drop=True)
    matched["gru_prob"] = gru.y_prob.to_numpy(dtype=np.float64)
    matched["gru_pred"] = gru.y_pred.to_numpy(dtype=np.int8)
    y = matched["p1_wins"].to_numpy(dtype=np.int8)
    models = {}
    for model in MODELS:
        models[model] = model_metrics(y, matched[f"{model}_prob"].to_numpy(dtype=np.float64),
                                       matched[f"{model}_pred"].to_numpy(dtype=np.int8))
    differences = {}
    for i, baseline in enumerate(BASELINES):
        diff = {metric: float(models["gru"][metric] - models[baseline][metric])
                for metric in METRICS if models["gru"][metric] is not None and models[baseline][metric] is not None}
        # Reuse same paired ordering: bootstrap at replay level within each seed.
        ci = paired_replay_bootstrap(matched, baseline, reps=reps,
                                     seed=bootstrap_seed + seed * 13 + i)
        if not np.isclose(ci["accuracy_delta"], diff["accuracy"], atol=1e-12, rtol=0) or not np.isclose(
            ci["log_loss_delta"], diff["log_loss"], atol=1e-9, rtol=0
        ):
            raise ValueError("Paired delta consistency check failed")
        differences[baseline] = {"deltas_gru_minus_baseline": diff, "paired_bootstrap": ci}
    gru_meta = json.loads((Path(gru_run_dir) / "summary.json").read_text(encoding="utf-8"))
    if not isinstance(gru_meta.get("config"), dict) or not gru_meta["config"]:
        raise ValueError("Missing GRU hyperparameter config in S4 summary")
    if not isinstance(gru_meta.get("selected_epoch"), int) or gru_meta["selected_epoch"] < 1:
        raise ValueError("Invalid selected GRU checkpoint epoch")
    return {
        "gru_config": gru_meta["config"], "gru_selected_epoch": gru_meta["selected_epoch"],
        "seed": seed, "partition": "val", "endpoints": len(matched),
        "replays": int(matched.replay_id.nunique()), "source_zip_sha256": index.manifest["source_zip_sha256"],
        "models": models, "gru_minus_baseline": differences,
    }, matched


def summarize_multiseed(per_seed: list[dict], *, reps: int, bootstrap_seed: int) -> dict:
    seeds = [x["seed"] for x in per_seed]
    if seeds != list(EXPECTED_SEEDS):
        raise ValueError("Full S5 final report requires ordered seeds 42, 43, 44")
    canonical_config = json.dumps(per_seed[0].get("gru_config"), sort_keys=True)
    if any(json.dumps(x.get("gru_config"), sort_keys=True) != canonical_config for x in per_seed):
        raise ValueError("GRU hyperparameters differ across seeds; cannot report one fixed experiment")
    digest = {x["source_zip_sha256"] for x in per_seed}
    if len(digest) != 1:
        raise ValueError("Source dataset hashes differ across seeds")
    averages = {}
    for model in MODELS:
        averages[model] = {metric: float(np.mean([x["models"][model][metric] for x in per_seed]))
                           for metric in METRICS}
    comparison = {}
    for baseline in BASELINES:
        comparison[baseline] = {
            "mean_deltas_gru_minus_baseline": {
                metric: float(np.mean([x["gru_minus_baseline"][baseline]["deltas_gru_minus_baseline"][metric]
                                       for x in per_seed]))
                for metric in METRICS
            },
            "seeds_favoring_gru_accuracy": sum(
                x["gru_minus_baseline"][baseline]["deltas_gru_minus_baseline"]["accuracy"] > 0
                for x in per_seed),
            "seeds_favoring_gru_log_loss": sum(
                x["gru_minus_baseline"][baseline]["deltas_gru_minus_baseline"]["log_loss"] < 0
                for x in per_seed),
        }
    return {
        "schema": "sc2.sequential.gru_matched_multiseed.v1",
        "status": "S5_MULTISEED_VALIDATION_PASS",
        "dataset": "v3_1_fixed", "window_length": 9,
        "seeds": seeds, "partition": "val", "dataset_sha256": next(iter(digest)),
        "gru_config": per_seed[0]["gru_config"],
        "bootstrap": {"unit": "replay_id", "replicates_per_seed_and_model": reps,
                      "base_seed": bootstrap_seed, "ci": "percentile_95_conditional_exploratory"},
        "per_seed": per_seed, "unweighted_seed_means": averages,
        "mean_paired_differences": comparison,
        "interpretation": (
            "Retrospective validation-only comparison; early stopping/model selection used validation. "
            "Replay-bootstrap intervals are conditional and exploratory, not an unbiased prospective performance test. "
            "Seeds reuse a common replay corpus; they are not independent data replicates. "
            "ROC-AUC values are point estimates only (not bootstrapped). No test data were read."
        ),
    }
