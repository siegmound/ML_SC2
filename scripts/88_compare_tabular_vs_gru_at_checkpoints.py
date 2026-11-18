#!/usr/bin/env python3
"""S11: retrospective, identical-checkpoint tabular vs recurrent evaluation.

Compares frozen RF/XGB/MLP validation predictions with the S8 nine-step GRU
and S9 matched short9/prefix GRUs on EXACT same replay/time checkpoints.
No training, fitting, calibration, test predictions, or model selection.
Historical training budgets/feature profiles differ; results are descriptive.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sc2proj.sequential.baselines import compare_archived_baselines, endpoint_frame
from sc2proj.sequential.comparison import (
    METRICS, load_verified_gru, model_metrics, paired_replay_bootstrap,
)
from sc2proj.sequential.data import load_index, load_replay_partitions
from sc2proj.sequential.progress import CHECKPOINTS
from sc2proj.sequential.stage_compare import (
    align_validation_model, validate_stage_cohort,
)
from sc2proj.sequential.variants import V31_SHA256

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEQ = RESULTS / "sequential"
SEEDS = (42, 43, 44)
EXPECTED_COHORTS = {42: 2012, 43: 2076, 44: 2040}
ARMS = ("train_prior", "rf_archived", "xgb_archived", "mlp_archived",
        "s8_gru_w9_allwindows", "s9_gru_last9_fourpoints",
        "s9_gru_prefix_fourpoints")
COMPARE_ARMS = tuple(x for x in ARMS if x not in ("train_prior", "s8_gru_w9_allwindows"))
GRU_REFERENCE = "s8_gru_w9_allwindows"
TABULAR = {"rf_archived": "rf", "xgb_archived": "xgb", "mlp_archived": "mlp"}


def base_to_predictions(matched, prefix):
    return pd.DataFrame({
        "replay_id": matched.replay_id,
        "time_sec": matched.time_sec,
        "y_true": matched.p1_wins,
        "y_prob": matched[f"{prefix}_prob"],
        "y_pred": matched[f"{prefix}_pred"],
    })


def score_one(y, probs, preds):
    p = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(y, dtype=np.int8)
    predicted = np.asarray(preds, dtype=np.int8)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Nonfinite or out-of-range scores")
    return model_metrics(labels, p, predicted)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index-dir", type=Path, default=SEQ / "v3_1_fixed_w9_s1")
    ap.add_argument("--s8-dir", type=Path, default=SEQ / "s8_game_progress_w9_all_arms_v1")
    ap.add_argument("--s9-aggregate", type=Path,
                    default=SEQ / "s9_prefix_vs_last9_multiseed_v1/summary.json")
    ap.add_argument("--bootstrap-reps", type=int, default=1000)
    ap.add_argument("--bootstrap-seed", type=int, default=2026)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.bootstrap_reps < 0:
        ap.error("Negative bootstrap repetitions")
    output = args.output_dir.resolve()
    if output == SEQ.resolve() or not output.is_relative_to(SEQ.resolve()) or output.exists():
        ap.error("Output must be a FRESH directory under results/sequential")

    index = load_index(args.index_dir)
    if (index.manifest.get("source_zip_sha256") != V31_SHA256
            or int(index.manifest["window_length"]) != 9
            or index.manifest.get("subset_max_replays") is not None
            or len(index.ordered_rows) != 892047
            or len(index.windows) != 706952):
        raise ValueError("Frozen v3.1 full S1 index identity mismatch")
    s8_summary = json.loads((args.s8_dir / "summary.json").read_text(encoding="utf-8"))
    s9_summary = json.loads(args.s9_aggregate.read_text(encoding="utf-8"))
    if (s8_summary.get("schema") != "sc2.sequential.progress_curve.v1"
            or s8_summary.get("status") != "S8_VALIDATION_PROGRESS_PASS"
            or s8_summary.get("source_zip_sha256") != V31_SHA256
            or s8_summary.get("seeds") != list(SEEDS)
            or s8_summary.get("checkpoints_percent") != list(CHECKPOINTS)
            or "v31_original64" not in s8_summary.get("model_arms", [])):
        raise ValueError("Missing or wrong original S8 validation provenance")
    if (s9_summary.get("status") != "S9_MULTISEED_VALIDATION_PASS"
            or s9_summary.get("s8_cohort_crosscheck") != "exact_key_and_label_match_PASS"
            or s9_summary.get("source_zip_sha256") != V31_SHA256
            or s9_summary.get("seeds") != list(SEEDS)):
        raise ValueError("S11 requires previously audited S9 validation cohort")
    s8_all = pd.read_csv(
        args.s8_dir / "matched_checkpoint_predictions.csv",
        dtype={"replay_id": "string"},
    )
    needed_s8 = {"seed", "replay_id", "time_sec", "progress_percent", "p1_wins",
                 "v31_original64_prob", "v31_original64_pred"}
    if not needed_s8.issubset(s8_all.columns) or s8_all.duplicated(
        ["seed", "replay_id", "progress_percent"]
    ).any() or set(s8_all.seed) != set(SEEDS):
        raise ValueError("S8 predictions missing, repeated or unexpected seed IDs")
    rows = []
    contrasts = []
    per_seed = []
    for seed in SEEDS:
        split_file = ROOT / f"data/processed/splits/real_v3_1_fixed_split_seed_{seed}.json"
        partition = load_replay_partitions(split_file, index)
        if partition.seed != seed:
            raise ValueError("Unexpected split-manifest seed")
        cohort = s8_all.loc[s8_all.seed == seed].copy()
        n_replays = validate_stage_cohort(cohort)
        if n_replays != EXPECTED_COHORTS[seed]:
            raise ValueError(f"S8 seed {seed}: historical cohort count changed")
        val_ids = partition.groups["val"]
        if not set(cohort.replay_id.astype(str)).issubset(val_ids):
            raise ValueError("S8 checkpoint outside replay-disjoint validation split")

        # S3 explicitly matches archived RF/XGB/MLP on *every* S1 validation endpoint.
        s3_report, tabular_all = compare_archived_baselines(
            index, partition, "val", RESULTS,
            models=("rf", "xgb", "mlp"),
        )
        if s3_report["status"] != "S3_MATCHED_ENDPOINTS_PASS":
            raise ValueError(f"Seed {seed}: archived validation baseline gate failed")
        all_val_endpoints = endpoint_frame(index, partition, "val")
        historical_gru = load_verified_gru(
            SEQ / f"s4_gru_w9_seed{seed}_v1",
            seed, index, all_val_endpoints,
        )
        sources = {
            arm: base_to_predictions(tabular_all, name)
            for arm, name in TABULAR.items()
        }
        sources[GRU_REFERENCE] = historical_gru
        for mode in ("last9", "prefix"):
            frame = pd.read_csv(
                SEQ / f"s9_prefix_vs_last9_seed{seed}_v1" /
                f"{mode}_validation_predictions.csv",
                dtype={"replay_id": "string"},
            )
            sources[f"s9_gru_{mode}_fourpoints"] = frame

        aligned = {}
        for arm, predictions in sources.items():
            is_tabular = arm in TABULAR
            matched = align_validation_model(
                cohort, predictions, arm,
                require_exact=arm.startswith("s9_"),
                enforce_half_threshold=not is_tabular,
            )
            if arm == GRU_REFERENCE:
                if not np.allclose(
                    matched.y_prob.to_numpy(dtype=np.float64),
                    matched.v31_original64_prob.to_numpy(dtype=np.float64),
                    atol=1e-12, rtol=0,
                ) or not np.array_equal(
                    matched.y_pred.to_numpy(dtype=np.int8),
                    matched.v31_original64_pred.to_numpy(dtype=np.int8),
                ):
                    raise ValueError("S8 stored GRU predictions disagree with frozen S4")
            aligned[arm] = matched
        # A fair, nonfitted trivial reference: prior P1 win fraction estimated
        # only from TRAIN replays, not validation/test replay labels.
        training_rows = index.ordered_rows.loc[partition.row_masks["train"]]
        train_labels = training_rows.groupby(
            "replay_id", sort=False, observed=True
        )["p1_wins"].first().to_numpy(dtype=np.int8)
        if not len(train_labels):
            raise ValueError("Empty train replay prior")
        prior = float(train_labels.mean())
        baseline_pred = int(prior >= 0.5)
        ordered = aligned[GRU_REFERENCE]

        for progress in CHECKPOINTS:
            current = ordered.loc[
                ordered.progress_percent == progress
            ].reset_index(drop=True)
            y = current.p1_wins.to_numpy(dtype=np.int8)
            if len(y) != n_replays or len(np.unique(y)) != 2:
                raise ValueError("Invalid stage size or single-class stage")
            at_progress = {}
            for arm in ARMS:
                if arm == "train_prior":
                    probs = np.full(len(y), prior, dtype=np.float64)
                    preds = np.full(len(y), baseline_pred, dtype=np.int8)
                else:
                    group = aligned[arm].loc[
                        aligned[arm].progress_percent == progress
                    ].reset_index(drop=True)
                    if not np.array_equal(
                        group.replay_id.to_numpy(),
                        current.replay_id.to_numpy(),
                    ) or not np.array_equal(
                        group.time_sec.to_numpy(),
                        current.time_sec.to_numpy(),
                    ) or not np.array_equal(
                        group.p1_wins.to_numpy(),
                        current.p1_wins.to_numpy(),
                    ):
                        raise ValueError(f"{arm}: checkpoint population mismatch")
                    probs = group.y_prob.to_numpy(dtype=np.float64)
                    preds = group.y_pred.to_numpy(dtype=np.int8)
                metrics = score_one(y, probs, preds)
                at_progress[arm] = (probs, preds, metrics)
                rows.append({
                    "seed": seed, "progress_percent": progress,
                    "model": arm, "n_replays": len(y),
                    **metrics,
                })
            for i, arm in enumerate(COMPARE_ARMS):
                baseline = at_progress[GRU_REFERENCE]
                cand = at_progress[arm]
                paired = pd.DataFrame({
                    "replay_id": current.replay_id,
                    "p1_wins": y,
                    "v31_prob": baseline[0], "v31_pred": baseline[1],
                    "gru_prob": cand[0], "gru_pred": cand[1],
                })
                interval = paired_replay_bootstrap(
                    paired, "v31",
                    reps=args.bootstrap_reps,
                    seed=args.bootstrap_seed + 29 * seed + 101 * progress + i,
                )
                differences = {
                    f"delta_{m}": float(cand[2][m] - baseline[2][m])
                    for m in METRICS
                }
                if not np.isclose(
                    differences["delta_accuracy"], interval["accuracy_delta"],
                    rtol=0, atol=1e-12
                ) or not np.isclose(
                    differences["delta_log_loss"], interval["log_loss_delta"],
                    rtol=0, atol=1e-9,
                ):
                    raise ValueError("Paired model delta mismatch")
                contrasts.append({
                    "seed": seed, "progress_percent": progress, "model": arm,
                    "reference": GRU_REFERENCE, "n_replays": len(y),
                    **differences,
                    "accuracy_ci_low": (
                        interval["accuracy_delta_ci95"][0] if args.bootstrap_reps else None
                    ),
                    "accuracy_ci_high": (
                        interval["accuracy_delta_ci95"][1] if args.bootstrap_reps else None
                    ),
                    "log_loss_ci_low": (
                        interval["log_loss_delta_ci95"][0] if args.bootstrap_reps else None
                    ),
                    "log_loss_ci_high": (
                        interval["log_loss_delta_ci95"][1] if args.bootstrap_reps else None
                    ),
                })
        per_seed.append({
            "seed": seed, "matched_validation_replays": n_replays,
            "matched_validation_endpoints": len(cohort),
            "train_replay_p1_win_prior": prior,
            "archived_s3_validation_gate": s3_report["status"],
        })
        print(f"[S11] seed={seed}: {n_replays:,} identical replay checkpoints "
              f"across tabular and GRU models — PASS", flush=True)

    metrics = pd.DataFrame(rows)
    deltas = pd.DataFrame(contrasts)
    mean_metrics = metrics.groupby(
        ["progress_percent", "model"], as_index=False
    )[list(METRICS)].mean()
    mean_deltas = deltas.groupby(
        ["progress_percent", "model"], as_index=False
    )[[f"delta_{m}" for m in METRICS]].mean()
    report = {
        "schema": "sc2.sequential.stagewise_tabular_vs_gru.v1",
        "status": "S11_MATCHED_VALIDATION_MODELS_PASS",
        "seeds": list(SEEDS), "stages": list(CHECKPOINTS),
        "source_zip_sha256": V31_SHA256,
        "s8_s9_endpoint_and_label_parity": "PASS",
        "per_seed": per_seed,
        "models": list(ARMS),
        "unweighted_seed_means": mean_metrics.to_dict(orient="records"),
        "unweighted_seed_deltas_vs_s8_gru": mean_deltas.to_dict(orient="records"),
        "bootstrap": {"reps": args.bootstrap_reps,
                      "unit": "replay_id", "seed": args.bootstrap_seed},
        "limitations": [
            "Retrospective validation; prior test reports were inspected historically.",
            "Identical validation replay checkpoints do NOT imply equal training examples, features or budgets.",
            "S8 GRU trained on all 9-snapshot windows; S9 GRUs trained at only four checkpoints/replay.",
            "Historical tabular models use archived decisions and potentially differing feature/selection profiles.",
            "Three seeds reuse the same underlying corpus; bootstrap intervals are within-seed descriptive.",
            "Selected replay cohort requires a valid 9-snapshot endpoint at all four progress stages.",
            "Replay percentage uses known final duration only to select offline endpoints, not model inputs.",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(output / "metrics_by_seed_stage.csv", index=False)
    deltas.to_csv(output / "paired_deltas_vs_s8_gru.csv", index=False)
    mean_metrics.to_csv(output / "unweighted_mean_metrics.csv", index=False)
    mean_deltas.to_csv(output / "unweighted_mean_deltas.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "per_seed": per_seed,
        "accuracy_by_stage": [
            {"progress_percent": int(p),
             "accuracy_by_model": {
                 row.model: round(float(row.accuracy), 6)
                 for row in mean_metrics.loc[mean_metrics.progress_percent == p].itertuples()
             }}
            for p in CHECKPOINTS
        ],
    }, indent=2, sort_keys=True))
    print(f"Saved read-only S11 audit to {output}")


if __name__ == "__main__":
    main()
