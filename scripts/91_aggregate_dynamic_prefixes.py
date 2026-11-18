#!/usr/bin/env python3
"""S13 read-only multi-seed audit: frozen S9 versus S13 at identical checkpoints.

Checks 3 seeds x 2 GRU modes x 4 progress checkpoints, saved prediction keys,
labels, probability metrics, differences, bootstrap point estimates and source
provenance. Produces descriptive, unweighted seed means. Never reads test data,
trains models, or pools dependent seeds for confidence intervals.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sc2proj.sequential.comparison import METRICS, model_metrics
from sc2proj.sequential.progress import CHECKPOINTS
from sc2proj.sequential.variants import V31_SHA256

RESULTS = Path(__file__).resolve().parents[1] / "results/sequential"
SEEDS = (42, 43, 44)
MODES = ("last9", "prefix")
TRAIN = {42: 8283, 43: 8274, 44: 8262}
VALIDATION = {42: 2012, 43: 2076, 44: 2040}
KEYS = ("replay_id", "time_sec", "progress_percent", "y_true")


def load_predictions(path: Path, expected_replays: int) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"replay_id": "string"})
    required = set(KEYS) | {"y_prob", "y_pred"}
    if not required.issubset(df.columns) or df.empty or df[list(required)].isna().any().any():
        raise ValueError(f"Incomplete validation predictions: {path}")
    if (len(df) != 4 * expected_replays
            or df.replay_id.nunique() != expected_replays
            or df.duplicated(["replay_id", "time_sec"]).any()
            or df.duplicated(["replay_id", "progress_percent"]).any()
            or set(df.progress_percent) != set(CHECKPOINTS)
            or not df.groupby("replay_id").progress_percent.nunique().eq(4).all()
            or not df.groupby("replay_id").y_true.nunique().eq(1).all()):
        raise ValueError(f"Validation replay/checkpoint identity mismatch: {path}")
    prob = df.y_prob.to_numpy(dtype=np.float64)
    if (not np.isfinite(prob).all() or ((prob < 0) | (prob > 1)).any()
            or not np.isin(df.y_true, [0, 1]).all()
            or not np.isin(df.y_pred, [0, 1]).all()
            or not np.array_equal(
                (prob >= 0.5).astype(np.int8), df.y_pred.to_numpy(dtype=np.int8),
            )):
        raise ValueError(f"Invalid validation model predictions: {path}")
    return df.sort_values(
        ["replay_id", "progress_percent"], kind="stable",
    ).reset_index(drop=True)


def _close(actual, expected, where: str, atol: float = 2e-7):
    if not np.isclose(actual, expected, atol=atol, rtol=0):
        raise ValueError(f"Numerical mismatch in {where}: {actual} vs {expected}")


def audit_seed(seed: int, root: Path) -> tuple[list[dict], dict]:
    s9_dir = root / f"s9_prefix_vs_last9_seed{seed}_v1"
    s13_dir = root / f"s13_dynamic_cutoffs_seed{seed}_v1"
    s9 = json.loads((s9_dir / "summary.json").read_text(encoding="utf-8"))
    s13 = json.loads((s13_dir / "summary.json").read_text(encoding="utf-8"))
    if (s9.get("schema") != "sc2.sequential.prefix_vs_last9.v1"
            or s9.get("status") != "S9_MATCHED_VALIDATION_PASS"
            or s9.get("source_zip_sha256") != V31_SHA256
            or s9.get("seed") != seed
            or s9.get("train_examples") != 4 * TRAIN[seed]
            or s9.get("validation_examples") != 4 * VALIDATION[seed]
            or s13.get("schema") != "sc2.sequential.dynamic_cutoff_training.v1"
            or s13.get("status") != "S13_DYNAMIC_VALIDATION_PASS"
            or s13.get("seed") != seed
            or s13.get("source_zip_sha256") != V31_SHA256
            or s13.get("test_evaluation_policy") != "not_accessed_during_training"
            or s13.get("config") != s9.get("settings")):
        raise ValueError(f"Seed {seed}: S9/S13 frozen run identity/config mismatch")
    pool = s13.get("train_pool", {})
    if (pool.get("training_replays") != TRAIN[seed]
            or pool.get("s9_anchor_examples") != 4 * TRAIN[seed]
            or pool.get("expanded_train_examples") != 16 * TRAIN[seed]
            or pool.get("extra_examples") != 12 * TRAIN[seed]
            or pool.get("extras_per_replay_min") != 12
            or pool.get("extras_per_replay_max") != 12
            or s13.get("validation_cohort", {}).get("balanced_replays") != VALIDATION[seed]
            or s9.get("validation_coverage", {}).get("balanced_replays") != VALIDATION[seed]):
        raise ValueError(f"Seed {seed}: training or validation cohort mismatch")
    rows = []
    by_mode = {}
    for mode in MODES:
        history = s13.get("models", {}).get(mode, {})
        s9_arm = s9.get("arms", {}).get(mode, {})
        if (history.get("mode") != mode or history.get("seed") != seed
                or history.get("status") != "S13_FIT_PASS"
                or history.get("test_evaluation_policy") != "not_accessed_during_training"
                or history.get("checkpoint_selection") != "validation_bce_only"
                or history.get("train_examples_per_epoch") != 4 * TRAIN[seed]
                or history.get("train_pool_examples") != 16 * TRAIN[seed]
                or history.get("validation_examples") != 4 * VALIDATION[seed]
                or history.get("config") != s13["config"]
                or not history.get("history")
                or s9_arm.get("config") != s13["config"]):
            raise ValueError(f"Seed {seed} {mode}: invalid frozen training history")
        best = int(history.get("selected_epoch", 0))
        if not 1 <= best <= len(history["history"]):
            raise ValueError(f"Seed {seed} {mode}: invalid selected epoch")
        for h in history["history"]:
            if (h["epoch_train_examples"] != 4 * TRAIN[seed]
                    or not np.isfinite(h["validation_log_loss"])
                    or not np.isfinite(h["train_log_loss"])):
                raise ValueError(f"Seed {seed} {mode}: invalid dynamic per-epoch budget")
        _close(history["history"][best - 1]["validation_log_loss"],
               history["best_validation_log_loss"], "validation selected epoch", atol=1e-10)
        reference = load_predictions(
            s9_dir / f"{mode}_validation_predictions.csv", VALIDATION[seed],
        )
        candidate = load_predictions(
            s13_dir / f"{mode}_validation_predictions.csv", VALIDATION[seed],
        )
        for col in KEYS:
            if not np.array_equal(reference[col].to_numpy(), candidate[col].to_numpy()):
                raise ValueError(f"Seed {seed} {mode}: checkpoint {col} drift")
        by_mode[mode] = candidate
        for progress in CHECKPOINTS:
            a = reference.loc[reference.progress_percent == progress]
            b = candidate.loc[candidate.progress_percent == progress]
            y = a.y_true.to_numpy(dtype=np.int8)
            m9 = model_metrics(
                y, a.y_prob.to_numpy(dtype=np.float64), a.y_pred.to_numpy(dtype=np.int8),
            )
            m13 = model_metrics(
                y, b.y_prob.to_numpy(dtype=np.float64), b.y_pred.to_numpy(dtype=np.int8),
            )
            recorded = s13["s13_minus_s9_same_mode"][mode][str(progress)]
            historical = s9["paired_checkpoint_metrics"][str(progress)]["models"][mode]
            if recorded["n_replays"] != VALIDATION[seed]:
                raise ValueError(f"Seed {seed} {mode} {progress}: wrong sample count")
            boot = recorded["paired_replay_bootstrap"]
            if boot["replay_groups"] != VALIDATION[seed] or boot["replicates"] != s13["bootstrap"]["reps"]:
                raise ValueError(f"Seed {seed} {mode} {progress}: invalid bootstrap metadata")
            for metric in METRICS:
                _close(m9[metric], historical[metric], "S9 archived metric")
                _close(m9[metric], recorded["reference"][metric], "S13 baseline metric")
                _close(m13[metric], recorded["s13"][metric], "S13 archived metric")
                _close(m13[metric] - m9[metric],
                       recorded["delta_s13_minus_reference"][metric], "paired metric delta")
            for metric in ("accuracy", "log_loss"):
                _close(m13[metric] - m9[metric], boot[f"{metric}_delta"],
                       "paired bootstrap point estimate")
                ci = boot[f"{metric}_delta_ci95"]
                if len(ci) != 2 or not all(np.isfinite(ci)) or ci[0] > ci[1]:
                    raise ValueError(f"Seed {seed} {mode} {progress}: corrupt bootstrap interval")
            rows.append({
                "seed": seed, "mode": mode, "progress_percent": progress,
                "n_replays": len(y),
                **{f"s9_{metric}": m9[metric] for metric in METRICS},
                **{f"s13_{metric}": m13[metric] for metric in METRICS},
                **{f"delta_{metric}": m13[metric] - m9[metric] for metric in METRICS},
                "accuracy_ci95_low": boot["accuracy_delta_ci95"][0],
                "accuracy_ci95_high": boot["accuracy_delta_ci95"][1],
                "log_loss_ci95_low": boot["log_loss_delta_ci95"][0],
                "log_loss_ci95_high": boot["log_loss_delta_ci95"][1],
            })
    for col in KEYS:
        if not np.array_equal(by_mode["last9"][col].to_numpy(),
                              by_mode["prefix"][col].to_numpy()):
            raise ValueError(f"Seed {seed}: S13 Last9/Prefix validation cohort differs")
    info = {
        "seed": seed, "train_replays": TRAIN[seed],
        "validation_replays": VALIDATION[seed],
        "candidate_train_examples": 16 * TRAIN[seed],
        "train_examples_per_epoch": 4 * TRAIN[seed],
        "selected_epochs": {
            mode: s13["models"][mode]["selected_epoch"] for mode in MODES
        },
    }
    return rows, info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=RESULTS)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    output = args.output_dir.resolve()
    if output == RESULTS.resolve() or not output.is_relative_to(RESULTS.resolve()) or output.exists():
        ap.error("A new output directory inside results/sequential is required")
    data = []
    info = []
    for seed in SEEDS:
        observations, run = audit_seed(seed, args.results_dir)
        data.extend(observations)
        info.append(run)
        print(f"[S13] seed={seed}: exact S9/S13 validation, metrics, CI and train-budget PASS", flush=True)
    by_seed = pd.DataFrame(data).sort_values(
        ["seed", "progress_percent", "mode"], kind="stable",
    ).reset_index(drop=True)
    measure_cols = [
        f"{kind}_{m}" for kind in ("s9", "s13", "delta") for m in METRICS
    ]
    means = by_seed.groupby(["progress_percent", "mode"], as_index=False)[measure_cols].mean()
    direction = by_seed.groupby(
        ["progress_percent", "mode"], as_index=False,
    ).agg(
        seeds_improved_accuracy=("delta_accuracy", lambda v: int((v > 0).sum())),
        seeds_improved_log_loss=("delta_log_loss", lambda v: int((v < 0).sum())),
    )
    means = means.merge(direction, on=["progress_percent", "mode"], validate="one_to_one")
    report = {
        "schema": "sc2.sequential.dynamic_multiseed_validation.v1",
        "status": "S13_MULTISEED_VALIDATION_PASS",
        "source_zip_sha256": V31_SHA256,
        "seeds": list(SEEDS), "checkpoints_percent": list(CHECKPOINTS),
        "per_seed": info,
        "unweighted_seed_mean_metrics": means.to_dict(orient="records"),
        "limitations": [
            "Only historical validation; no untouched test predictions or labels were evaluated.",
            "Checkpoint percentages are retrospective and use completed replay lengths offline.",
            "Three random seeds overlap in replay corpus and are not independent replications.",
            "Per-seed paired replay-bootstrap intervals are conditional and not multiple-comparison adjusted.",
            "Mean seed deltas and counts of favorable seeds are descriptive; no pooled confidence interval.",
            "No post hoc checkpoint-specific model selection for deployable inference.",
            "Candidate cutoff pool computed from train replay geometry, but each model sees only current/past snapshots.",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    by_seed.to_csv(output / "metrics_by_seed_progress.csv", index=False)
    means.to_csv(output / "mean_progress_metrics.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "per_seed": info,
        "checkpoint_90": means.loc[means.progress_percent == 90].to_dict(orient="records"),
    }, indent=2))
    print(f"Saved S13 multiseed read-only audit to {output}")


if __name__ == "__main__":
    main()
