#!/usr/bin/env python3
"""S10: diagnose saved S9 winner probabilities at four exact S8 checkpoints.

No training, no recalibration, no ZIP access, and no test evaluation.
Uses saved S9/S8 validation predictions plus S9 audit report, fails closed
on provenance, endpoint, label, score, and paired cohort inconsistencies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sc2proj.sequential.calibration import (
    calibration_diagnostics, paired_brier_bootstrap,
)
from sc2proj.sequential.comparison import METRICS, model_metrics
from sc2proj.sequential.progress import CHECKPOINTS
from sc2proj.sequential.variants import V31_SHA256

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/sequential"
SEEDS = (42, 43, 44)
MODES = ("last9", "prefix")
KEYS = ("replay_id", "time_sec", "progress_percent", "y_true")


def read_predictions(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, dtype={"replay_id": "string"})
    required = set(KEYS) | {"y_prob", "y_pred"}
    if data.empty or not required.issubset(data.columns):
        raise ValueError(f"Missing S9 prediction columns: {path}")
    if data.loc[:, list(required)].isna().any().any():
        raise ValueError(f"Missing S9 prediction values: {path}")
    if data.duplicated(["replay_id", "time_sec"]).any() or data.duplicated(
        ["replay_id", "progress_percent"]
    ).any():
        raise ValueError(f"Duplicate S9 checkpoint keys: {path}")
    if set(data.progress_percent) != set(CHECKPOINTS) or not np.isin(
        data.y_true, [0, 1]
    ).all() or not np.isin(data.y_pred, [0, 1]).all():
        raise ValueError(f"Invalid S9 labels/checkpoints: {path}")
    p = data.y_prob.to_numpy(dtype=np.float64)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() or not np.array_equal(
        (p >= 0.5).astype(np.int8), data.y_pred.to_numpy(dtype=np.int8)
    ):
        raise ValueError(f"Invalid S9 probabilities or threshold: {path}")
    if not data.groupby("replay_id")["progress_percent"].nunique().eq(4).all() or not data.groupby(
        "replay_id"
    )["y_true"].nunique().eq(1).all():
        raise ValueError(f"Unbalanced or inconsistent S9 replay cohort: {path}")
    return data.sort_values(["replay_id", "progress_percent"], kind="stable").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s9-summary", type=Path,
                    default=RESULTS / "s9_prefix_vs_last9_multiseed_v1" / "summary.json")
    ap.add_argument("--s8-predictions", type=Path,
                    default=RESULTS / "s8_game_progress_w9_all_arms_v1" /
                            "matched_checkpoint_predictions.csv")
    ap.add_argument("--bootstrap-reps", type=int, default=1000)
    ap.add_argument("--bootstrap-seed", type=int, default=2026)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()
    if args.bootstrap_reps < 0 or not 2 <= args.bins <= 100:
        ap.error("Bootstrap reps must be nonnegative; bins must be in [2,100]")
    if args.output_dir is None:
        ap.error("Supply a fresh output directory under results/sequential/")
    output = args.output_dir.resolve()
    if output == RESULTS.resolve() or not output.is_relative_to(RESULTS.resolve()) or output.exists():
        ap.error("Refusing to overwrite prior results or write outside results/sequential/")

    audit = json.loads(args.s9_summary.read_text(encoding="utf-8"))
    if (audit.get("status") != "S9_MULTISEED_VALIDATION_PASS"
            or audit.get("schema") != "sc2.sequential.prefix_multiseed_validation.v1"
            or audit.get("s8_cohort_crosscheck") != "exact_key_and_label_match_PASS"
            or audit.get("source_zip_sha256") != V31_SHA256
            or audit.get("seeds") != list(SEEDS)):
        raise ValueError("S10 requires verified frozen S9 aggregate report")
    summaries = {int(v["seed"]): v for v in audit["per_seed"]}
    if set(summaries) != set(SEEDS):
        raise ValueError("Incomplete S9 audit seed set")
    s8 = pd.read_csv(args.s8_predictions, dtype={"replay_id": "string"})
    s8_required = {"seed", "replay_id", "time_sec", "progress_percent", "p1_wins"}
    if not s8_required.issubset(s8.columns) or s8.duplicated(
        ["seed", "replay_id", "progress_percent"]
    ).any():
        raise ValueError("Missing or repeated S8 progress keys")
    if not set(SEEDS).issubset(set(s8.seed)):
        raise ValueError("S8 checkpoint cohort missing seed")
    stage_rows, bin_rows, paired_rows = [], [], []
    for seed in SEEDS:
        run = RESULTS / f"s9_prefix_vs_last9_seed{seed}_v1"
        short = read_predictions(run / "last9_validation_predictions.csv")
        prefix = read_predictions(run / "prefix_validation_predictions.csv")
        expected = summaries[seed]["validation_replays"]
        if len(short) != 4 * expected or len(prefix) != len(short):
            raise ValueError(f"Seed {seed}: S9 validation count changed")
        for key in KEYS:
            if not np.array_equal(short[key].to_numpy(), prefix[key].to_numpy()):
                raise ValueError(f"Seed {seed}: S9 arm key mismatch for {key}")
        selected_s8 = s8.loc[s8.seed == seed].sort_values(
            ["replay_id", "progress_percent"], kind="stable"
        ).reset_index(drop=True)
        if len(selected_s8) != len(short):
            raise ValueError(f"Seed {seed}: S8 cohort count mismatch")
        for key, s8_key in (
            ("replay_id", "replay_id"), ("time_sec", "time_sec"),
            ("progress_percent", "progress_percent"), ("y_true", "p1_wins")
        ):
            if not np.array_equal(short[key].to_numpy(), selected_s8[s8_key].to_numpy()):
                raise ValueError(f"Seed {seed}: S9/S8 key mismatch {key}")
        stored = summaries[seed]["validation_checkpoint_scores"]
        if set(stored) != set(map(str, CHECKPOINTS)):
            raise ValueError(f"Seed {seed}: saved checkpoint score set differs")
        for progress in CHECKPOINTS:
            a = short.loc[short.progress_percent == progress].reset_index(drop=True)
            b = prefix.loc[prefix.progress_percent == progress].reset_index(drop=True)
            if len(a) != expected or len(b) != expected:
                raise ValueError(f"Seed {seed} {progress}%: missing replay predictions")
            yy = a.y_true.to_numpy(dtype=np.int8)
            for mode, frame in (("last9", a), ("prefix", b)):
                p = frame.y_prob.to_numpy(dtype=np.float64)
                predicted = frame.y_pred.to_numpy(dtype=np.int8)
                recomputed = model_metrics(yy, p, predicted)
                for metric in METRICS:
                    observed = stored[str(progress)]["models"][mode][metric]
                    if not np.isclose(recomputed[metric], observed, rtol=0, atol=2e-8):
                        raise ValueError(f"Seed {seed} {progress}% {mode}: {metric} drift")
                diag = calibration_diagnostics(yy, p, bins=args.bins)
                if not np.isclose(diag["brier_score"], recomputed["brier_score"], atol=1e-12, rtol=0):
                    raise ValueError("S10 Brier calculation does not match S9")
                stage_rows.append({
                    "seed": seed, "progress_percent": progress, "mode": mode,
                    **{k: v for k, v in diag.items() if k != "reliability_bins"},
                    "accuracy": recomputed["accuracy"],
                    "balanced_accuracy": recomputed["balanced_accuracy"],
                    "roc_auc": recomputed["roc_auc"],
                })
                for item in diag["reliability_bins"]:
                    bin_rows.append({
                        "seed": seed, "progress_percent": progress, "mode": mode,
                        **item,
                    })
            bootstrap = paired_brier_bootstrap(
                yy, a.y_prob.to_numpy(dtype=np.float64),
                b.y_prob.to_numpy(dtype=np.float64),
                reps=args.bootstrap_reps, seed=args.bootstrap_seed + seed * 13 + progress,
            )
            expected_difference = (
                stored[str(progress)]["delta_prefix_minus_last9"]["brier_score"]
            )
            if not np.isclose(bootstrap["delta_brier_prefix_minus_last9"],
                              expected_difference, rtol=0, atol=2e-8):
                raise ValueError(f"Seed {seed} {progress}%: paired Brier mismatch")
            paired_rows.append({
                "seed": seed, "progress_percent": progress,
                **bootstrap,
                "delta_brier_ci95_low": (
                    bootstrap["delta_brier_ci95"][0] if args.bootstrap_reps else None),
                "delta_brier_ci95_high": (
                    bootstrap["delta_brier_ci95"][1] if args.bootstrap_reps else None),
            })
        print(f"[S10] seed={seed} checked {expected:,} replay(s) at all four checkpoints",
              flush=True)

    stages = pd.DataFrame(stage_rows).sort_values(
        ["seed", "progress_percent", "mode"], kind="stable").reset_index(drop=True)
    bins = pd.DataFrame(bin_rows)
    paired = pd.DataFrame(paired_rows)
    mean_rows = []
    for p in CHECKPOINTS:
        a = stages[(stages.progress_percent == p) & (stages["mode"] == "last9")]
        b = stages[(stages.progress_percent == p) & (stages["mode"] == "prefix")]
        a = a.sort_values("seed")
        b = b.sort_values("seed")
        if not np.array_equal(a.seed.to_numpy(), b.seed.to_numpy()):
            raise ValueError("Unequal seed sets in S10 comparison")
        mean_rows.append({
            "progress_percent": p,
            "mean_ece_last9": float(a.expected_calibration_error.mean()),
            "mean_ece_prefix": float(b.expected_calibration_error.mean()),
            "delta_ece_prefix_minus_last9": float(
                (b.expected_calibration_error.to_numpy() -
                 a.expected_calibration_error.to_numpy()).mean()),
            "mean_abs_bias_last9": float(a.calibration_bias_p1.abs().mean()),
            "mean_abs_bias_prefix": float(b.calibration_bias_p1.abs().mean()),
            "delta_brier_prefix_minus_last9": float(
                (b.brier_score.to_numpy() - a.brier_score.to_numpy()).mean()),
            "delta_log_loss_prefix_minus_last9": float(
                (b.log_loss.to_numpy() - a.log_loss.to_numpy()).mean()),
            "delta_roc_auc_prefix_minus_last9": float(
                (b.roc_auc.to_numpy() - a.roc_auc.to_numpy()).mean()),
        })
    means = pd.DataFrame(mean_rows)
    report = {
        "schema": "sc2.sequential.probability_diagnostics.v1",
        "status": "S10_VALIDATION_PROBABILITY_AUDIT_PASS",
        "seeds": list(SEEDS),
        "modes": list(MODES),
        "checkpoints_percent": list(CHECKPOINTS),
        "reliability_method": f"{args.bins} fixed equal-width probability bins",
        "s9_provenance": "S9_MULTISEED_VALIDATION_PASS",
        "s8_checkpoint_parity": "exact_key_and_label_match_PASS",
        "source_zip_sha256": V31_SHA256,
        "bootstrap": {
            "reps": args.bootstrap_reps, "base_seed": args.bootstrap_seed,
            "resampling_unit": "one selected checkpoint observation per replay",
            "unit_applies_separately_to_each_seed_and_checkpoint": True,
        },
        "mean_unweighted_seed_diagnostics": mean_rows,
        "limitations": [
            "Retrospective validation only; not a new prospective or untouched test.",
            "ECE depends on bin count and sample size; lower ECE is not calibration proof.",
            "Log loss, Brier score and AUC measure different aspects of probability quality.",
            "Validation selected training checkpoints; no new calibration model was fitted.",
            "All seed splits are dependent partitions of one corpus; no pooled confidence interval.",
            "Complete-case S8 replay cohort omits shorter/irregularly sampled games.",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    stages.to_csv(output / "probability_diagnostics_by_seed.csv", index=False)
    bins.to_csv(output / "reliability_bins_by_seed.csv", index=False)
    paired.to_csv(output / "paired_brier_deltas_by_seed.csv", index=False)
    means.to_csv(output / "unweighted_seed_mean_diagnostics.csv", index=False)
    for seed in SEEDS:
        for progress in CHECKPOINTS:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect reliability")
            for mode in MODES:
                selection = bins.loc[
                    (bins.seed == seed) & (bins.progress_percent == progress)
                    & (bins["mode"] == mode) & (bins.n > 0)
                ].sort_values("bin")
                ax.plot(selection.predicted_p1, selection.observed_p1,
                        marker="o", label=mode)
            ax.set(xlabel="Mean predicted P1-win probability",
                   ylabel="Observed P1-win frequency",
                   title=f"Validation reliability — seed {seed}, {progress}%")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.2)
            ax.legend()
            fig.tight_layout()
            fig.savefig(output / f"reliability_seed{seed}_p{progress}.png", dpi=170)
            plt.close(fig)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved S10 diagnostic audit to {output}", flush=True)
    print(json.dumps({
        "status": report["status"],
        "s8_checkpoint_parity": report["s8_checkpoint_parity"],
        "means": mean_rows,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
