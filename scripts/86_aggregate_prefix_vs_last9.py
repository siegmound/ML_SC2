#!/usr/bin/env python3
"""S9: audit and aggregate frozen per-seed matched Prefix-vs-Last9 validation.

- Read-only: never loads torch checkpoints, dataset ZIP, or the test partition.
- Strictly verifies paired replay/time/label/checkpoint coverage, metrics, status,
  source SHA and matching fit configs; optionally crosschecks frozen S8 cohort.
- Do NOT pool bootstrap samples or treat seeds as independent experiments.
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

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/sequential"
SEEDS = (42, 43, 44)
MODES = ("last9", "prefix")
EXPECTED_TRAIN_REPLAYS = {42: 8283, 43: 8274, 44: 8262}
EXPECTED_VAL_REPLAYS = {42: 2012, 43: 2076, 44: 2040}
KEYS = ["replay_id", "time_sec", "progress_percent", "y_true"]


def read_predictions(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype={"replay_id": "string"})
    required = set(KEYS) | {"y_prob", "y_pred"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns in {path}")
    if frame.empty or frame.loc[:, list(required)].isna().any().any():
        raise ValueError(f"Missing/null predictions in {path}")
    if frame.duplicated(["replay_id", "time_sec"]).any():
        raise ValueError(f"Duplicate replay/timestamp in {path}")
    if frame.duplicated(["replay_id", "progress_percent"]).any():
        raise ValueError(f"Duplicate replay/checkpoint in {path}")
    if not set(frame.progress_percent) == set(CHECKPOINTS):
        raise ValueError(f"Unexpected checkpoint percentages in {path}")
    if not np.isin(frame.y_true, [0, 1]).all() or not np.isin(frame.y_pred, [0, 1]).all():
        raise ValueError(f"Nonbinary classification in {path}")
    prob = frame.y_prob.to_numpy(dtype=np.float64)
    if not np.isfinite(prob).all() or ((prob < 0) | (prob > 1)).any():
        raise ValueError(f"Invalid prediction probabilities in {path}")
    if not np.array_equal(frame.y_pred.to_numpy(dtype=np.int8),
                          (prob >= .5).astype(np.int8)):
        raise ValueError(f"Predictions mismatch 0.5 threshold in {path}")
    sizes = frame.groupby("replay_id")["progress_percent"].nunique()
    if not sizes.eq(len(CHECKPOINTS)).all():
        raise ValueError(f"Unbalanced replay cohort in {path}")
    if not frame.groupby("replay_id")["y_true"].nunique().eq(1).all():
        raise ValueError(f"Replay label changes within {path}")
    return frame.sort_values(["replay_id", "progress_percent"], kind="stable").reset_index(drop=True)


def audit_one(seed: int, source: Path, s8_cohort: Path | None):
    run_dir = source / f"s9_prefix_vs_last9_seed{seed}_v1"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if (summary.get("schema") != "sc2.sequential.prefix_vs_last9.v1"
            or summary.get("status") != "S9_MATCHED_VALIDATION_PASS"
            or summary.get("seed") != seed
            or summary.get("source_zip_sha256") != V31_SHA256):
        raise ValueError(f"Seed {seed}: S9 provenance/status mismatch")
    tc, vc = summary["train_coverage"], summary["validation_coverage"]
    if (tc.get("partition") != "train" or vc.get("partition") != "val"
            or tc["balanced_replays"] != EXPECTED_TRAIN_REPLAYS[seed]
            or vc["balanced_replays"] != EXPECTED_VAL_REPLAYS[seed]
            or list(tc["checkpoints"]) != list(CHECKPOINTS)
            or list(vc["checkpoints"]) != list(CHECKPOINTS)
            or summary.get("train_examples") != 4 * tc["balanced_replays"]
            or summary.get("validation_examples") != 4 * vc["balanced_replays"]):
        raise ValueError(f"Seed {seed}: missing/changed historical matched-cohort contract")
    arms = summary.get("arms", {})
    if set(arms) != set(MODES):
        raise ValueError(f"Seed {seed}: expected exactly both training arms")
    config = None
    for mode in MODES:
        arm = arms[mode]
        if (arm.get("mode") != mode or arm.get("seed") != seed
                or arm.get("source_zip_sha256") != V31_SHA256
                or arm.get("feature_count") != 64 or arm.get("smoke") is not False
                or arm.get("test_evaluation_policy") != "not_accessed_during_training"
                or arm.get("checkpoint_selection") != "validation_bce_only"
                or arm.get("train_examples") != 4 * tc["balanced_replays"]
                or arm.get("validation_examples") != 4 * vc["balanced_replays"]):
            raise ValueError(f"Seed {seed}: {mode} training provenance mismatch")
        if len(arm["features"]) != 64 or len(set(arm["features"])) != 64:
            raise ValueError(f"Seed {seed}: incorrect S9 input feature set")
        this_config = json.dumps(arm["config"], sort_keys=True)
        if config is None:
            config = this_config
        elif this_config != config:
            raise ValueError(f"Seed {seed}: S9 arm configs differ")
    by_arm = {
        mode: read_predictions(run_dir / f"{mode}_validation_predictions.csv")
        for mode in MODES
    }
    a, b = by_arm["last9"], by_arm["prefix"]
    if len(a) != 4 * EXPECTED_VAL_REPLAYS[seed] or len(a) != len(b):
        raise ValueError(f"Seed {seed}: unexpected validation prediction count")
    for key in KEYS:
        if not np.array_equal(a[key].to_numpy(), b[key].to_numpy()):
            raise ValueError(f"Seed {seed}: S9 mismatched paired key {key}")
    if s8_cohort is not None:
        ref = pd.read_csv(s8_cohort, dtype={"replay_id": "string"})
        ref = ref.loc[ref.seed == seed].sort_values(
            ["replay_id", "progress_percent"], kind="stable"
        ).reset_index(drop=True)
        if len(ref) != len(a):
            raise ValueError(f"Seed {seed}: S9 and frozen S8 cohort counts differ")
        for key, s8_key in (("replay_id", "replay_id"), ("time_sec", "time_sec"),
                            ("progress_percent", "progress_percent"),
                            ("y_true", "p1_wins")):
            if not np.array_equal(a[key].to_numpy(), ref[s8_key].to_numpy()):
                raise ValueError(f"Seed {seed}: S9 and S8 cohort differ in {key}")
    rows = []
    saved = summary["paired_checkpoint_metrics"]
    if set(saved) != set(map(str, CHECKPOINTS)):
        raise ValueError(f"Seed {seed}: missing progress scores")
    for p in CHECKPOINTS:
        stage_a = a.loc[a.progress_percent == p].reset_index(drop=True)
        stage_b = b.loc[b.progress_percent == p].reset_index(drop=True)
        if len(stage_a) != EXPECTED_VAL_REPLAYS[seed] or len(stage_b) != len(stage_a):
            raise ValueError(f"Seed {seed}: incorrect checkpoint cohort {p}")
        y = stage_a.y_true.to_numpy(dtype=np.int8)
        calculated = {}
        for mode, stage in (("last9", stage_a), ("prefix", stage_b)):
            calculated[mode] = model_metrics(
                y, stage.y_prob.to_numpy(dtype=np.float64),
                stage.y_pred.to_numpy(dtype=np.int8),
            )
            for metric in METRICS:
                if not np.isclose(calculated[mode][metric],
                                  saved[str(p)]["models"][mode][metric],
                                  rtol=0, atol=2e-8):
                    raise ValueError(f"Seed {seed} {p}% {mode}: saved {metric} differs")
        diffs = {
            m: float(calculated["prefix"][m] - calculated["last9"][m])
            for m in METRICS
        }
        if saved[str(p)]["n_replays"] != len(stage_a):
            raise ValueError(f"Seed {seed} {p}% has wrong replay count")
        for m in METRICS:
            if not np.isclose(diffs[m],
                              saved[str(p)]["delta_prefix_minus_last9"][m],
                              rtol=0, atol=2e-8):
                raise ValueError(f"Seed {seed} {p}% has wrong delta {m}")
        interval = saved[str(p)]["paired_replay_bootstrap"]
        if (interval.get("replay_groups") != len(stage_a)
                or interval.get("replicates") != summary["bootstrap"]["reps"]):
            raise ValueError(f"Seed {seed} {p}% invalid replay-bootstrap provenance")
        for metric in ("accuracy", "log_loss"):
            if not np.isclose(interval[f"{metric}_delta"], diffs[metric],
                              rtol=0, atol=2e-8):
                raise ValueError(f"Seed {seed} {p}% invalid bootstrap {metric} point estimate")
        for mode in MODES:
            rows.append({
                "seed": seed, "progress_percent": p,
                "mode": mode, "replays": len(stage_a),
                **calculated[mode],
            })
    return rows, summary, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--s8-cohort-csv", type=Path,
                        default=RESULTS / "s8_game_progress_w9_all_arms_v1"
                        / "matched_checkpoint_predictions.csv")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Fresh directory under results/sequential; no overwrite")
    args = parser.parse_args()
    source = args.results_dir.resolve()
    frozen_cohort = args.s8_cohort_csv.resolve()
    if not frozen_cohort.is_file():
        parser.error(f"Frozen S8 checkpoint cohort not found: {frozen_cohort}")
    output = args.output_dir.resolve() if args.output_dir is not None else None
    if output is not None and (
        output == RESULTS.resolve() or
        not output.is_relative_to(RESULTS.resolve()) or output.exists()
    ):
        parser.error("Output must be a new subdirectory under results/sequential")
    rows = []
    per_seed = []
    cfg = None
    for seed in SEEDS:
        scored, summary, config = audit_one(seed, source, frozen_cohort)
        if cfg is None:
            cfg = config
        elif cfg != config:
            raise ValueError("S9 fitting configurations differ across seeds")
        rows.extend(scored)
        per_seed.append({
            "seed": seed,
            "train_replays": summary["train_coverage"]["balanced_replays"],
            "validation_replays": summary["validation_coverage"]["balanced_replays"],
            "validation_checkpoint_scores": summary["paired_checkpoint_metrics"],
            "selected_epochs": {
                mode: summary["arms"][mode]["selected_epoch"] for mode in MODES
            },
        })
        print(f"[S9] seed={seed}: validated both modes at 4 exact S8 checkpoint cohorts",
              flush=True)
    tab = pd.DataFrame(rows).sort_values(
        ["seed", "progress_percent", "mode"], kind="stable"
    ).reset_index(drop=True)
    means = tab.groupby(["progress_percent", "mode"], as_index=False)[list(METRICS)].mean()
    contrasted = []
    for progress in CHECKPOINTS:
        changes = []
        for seed in SEEDS:
            a = tab.loc[(tab.seed == seed) & (tab.progress_percent == progress)
                        & (tab["mode"] == "last9")].iloc[0]
            b = tab.loc[(tab.seed == seed) & (tab.progress_percent == progress)
                        & (tab["mode"] == "prefix")].iloc[0]
            changes.append({m: float(b[m] - a[m]) for m in METRICS})
        contrasted.append({
            "progress_percent": progress,
            "seed_count": len(SEEDS),
            **{f"delta_{m}": float(np.mean([v[m] for v in changes])) for m in METRICS},
            "n_positive_accuracy_deltas": int(sum(v["accuracy"] > 0 for v in changes)),
            "n_lower_prefix_log_losses": int(sum(v["log_loss"] < 0 for v in changes)),
            "accuracy_delta_seed_min": float(min(v["accuracy"] for v in changes)),
            "accuracy_delta_seed_max": float(max(v["accuracy"] for v in changes)),
        })
    report = {
        "schema": "sc2.sequential.prefix_multiseed_validation.v1",
        "status": "S9_MULTISEED_VALIDATION_PASS",
        "seeds": list(SEEDS), "checkpoints_percent": list(CHECKPOINTS),
        "source_zip_sha256": V31_SHA256,
        "s8_cohort_crosscheck": "exact_key_and_label_match_PASS",
        "fitting_config": json.loads(cfg),
        "unweighted_mean_checkpoint_metrics": means.to_dict(orient="records"),
        "unweighted_mean_deltas": contrasted,
        "per_seed": per_seed,
        "limitations": [
            "Retrospective validation; every model was selected using validation BCE.",
            "All three seed splits reuse the same corpus. Means describe seed variation, not independent external replication.",
            "Whole-replay bootstrap confidence intervals appear only per seed; do not pool bootstrap samples or infer significance of seed mean.",
            "Final replay duration selects offline checkpoint snapshots and is never included in either model input.",
            "Exactly matched S8 complete-case cohorts exclude many shorter/gappy games; no untouched test accessed.",
        ],
    }
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
        tab.to_csv(output / "metrics_by_seed_checkpoint.csv", index=False)
        means.to_csv(output / "mean_checkpoint_metrics.csv", index=False)
        pd.DataFrame(contrasted).to_csv(output / "mean_paired_deltas.csv", index=False)
        (output / "summary.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"Saved S9 audited multiseed summary to {output}", flush=True)
    print(json.dumps({
        "status": report["status"], "s8_cohort_crosscheck": report["s8_cohort_crosscheck"],
        "mean_paired_deltas": contrasted,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
