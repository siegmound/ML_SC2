#!/usr/bin/env python3
"""S13: equal-per-epoch-budget dynamic train cutoffs for matched GRU arms.

Each TRAIN replay contributes exactly four examples per epoch, one sampled
from each anchor-centered stratum of the S12 16-cutoff pool. Selection varies
by epoch, is seeded/reproducible, and never reads validation/test labels.
S9 validation checkpoint cohort/config/early-stop are preserved.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from sc2proj.sequential.comparison import model_metrics, paired_replay_bootstrap
from sc2proj.sequential.data import (
    load_index, load_replay_partitions, prepare_sequential_data,
)
from sc2proj.sequential.dynamic_prefix import fit_dynamic_gru
from sc2proj.sequential.extra_prefixes import expand_train_checkpoint_examples
from sc2proj.sequential.gru import GRUConfig
from sc2proj.sequential.prefix import (
    MODES, CheckpointSequenceDataset, compare_modes, predict_checkpoint_gru,
)
from sc2proj.sequential.progress import CHECKPOINTS, select_exact_checkpoints
from sc2proj.sequential.variants import V31_SHA256

ROOT = Path(__file__).resolve().parents[1]
SEQ = ROOT / "results/sequential"
S9_AGG = SEQ / "s9_prefix_vs_last9_multiseed_v1/summary.json"
FIELDS = ("replay_id", "time_sec", "progress_percent", "y_true")


def read_reference_predictions(run_dir: Path, valid: pd.DataFrame,
                               config: GRUConfig, seed: int, kind: str):
    """Verify old S9/S12 validation endpoint parity and immutable experiment ID."""
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if kind == "s9":
        expected_schema, expected_status = (
            "sc2.sequential.prefix_vs_last9.v1", "S9_MATCHED_VALIDATION_PASS",
        )
        settings = summary.get("settings")
        arms = summary.get("arms")
    elif kind == "s12":
        expected_schema, expected_status = (
            "sc2.sequential.extra_prefix_training.v1",
            "S12_EXPANDED_TRAIN_VALIDATION_PASS",
        )
        settings = summary.get("config")
        arms = summary.get("s12_arms")
    else:
        raise ValueError("Only historical S9/S12 validation references allowed")
    if (summary.get("schema") != expected_schema
            or summary.get("status") != expected_status
            or summary.get("seed") != seed
            or summary.get("source_zip_sha256") != V31_SHA256
            or settings != asdict(config)
            or not isinstance(arms, dict)):
        raise ValueError(f"S13 invalid {kind} frozen reference metadata")
    if kind == "s9":
        if (summary.get("train_examples") != 4 * summary["train_coverage"]["balanced_replays"]
                or summary.get("validation_examples") != len(valid)):
            raise ValueError("S13 S9 historical training/validation population mismatch")
    elif (summary.get("validation_cohort", {}).get("balanced_replays")
          != valid.replay_id.nunique()):
        raise ValueError("S13 S12 historical validation cohort mismatch")
    canonical = valid.sort_values(
        ["replay_id", "progress_percent"], kind="stable",
    ).reset_index(drop=True)
    result = {}
    for mode in MODES:
        info = arms.get(mode, {})
        if (info.get("source_zip_sha256") != V31_SHA256
                or info.get("config") != asdict(config)
                or info.get("checkpoint_selection") != "validation_bce_only"
                or info.get("test_evaluation_policy") != "not_accessed_during_training"
                or info.get("smoke") is not False):
            raise ValueError(f"S13 invalid {kind} historical {mode} arm")
        saved = pd.read_csv(
            run_dir / f"{mode}_validation_predictions.csv",
            dtype={"replay_id": "string"},
        ).sort_values(
            ["replay_id", "progress_percent"], kind="stable",
        ).reset_index(drop=True)
        if saved.duplicated(["replay_id", "time_sec"]).any() or len(saved) != len(canonical):
            raise ValueError(f"S13 {kind} {mode} replay/time duplicate or missing")
        for old, expected in (
            ("replay_id", "replay_id"), ("time_sec", "time_sec"),
            ("progress_percent", "progress_percent"), ("y_true", "p1_wins"),
        ):
            if not np.array_equal(saved[old].to_numpy(), canonical[expected].to_numpy()):
                raise ValueError(f"S13 {kind} {mode}: validation {old} drift")
        probs = saved.y_prob.to_numpy(dtype=np.float64)
        if (not np.isfinite(probs).all() or ((probs < 0) | (probs > 1)).any()
                or saved[["y_true", "y_pred"]].isna().any().any()
                or not np.isin(saved.y_true, [0, 1]).all()
                or not np.isin(saved.y_pred, [0, 1]).all()
                or not np.array_equal(
                    (probs >= 0.5).astype(np.int8),
                    saved.y_pred.to_numpy(dtype=np.int8),
                )):
            raise ValueError(f"S13 corrupted {kind} {mode} probabilities")
        result[mode] = saved
    return result


def compare_to_reference(predicted: dict[str, pd.DataFrame],
                         reference: dict[str, pd.DataFrame], *,
                         reps: int, seed: int, label: str):
    """Paired S13-minus-reference deltas and replay-bootstrap accuracy/BCE CIs."""
    output = {}
    for mi, mode in enumerate(MODES):
        now = predicted[mode].sort_values(
            ["replay_id", "progress_percent"], kind="stable",
        ).reset_index(drop=True)
        old = reference[mode]
        for field in FIELDS:
            if not np.array_equal(now[field].to_numpy(), old[field].to_numpy()):
                raise ValueError(f"S13 {label} {mode}: validation {field} mismatch")
        stage = {}
        for pi, progress in enumerate(CHECKPOINTS):
            a = old.loc[old.progress_percent == progress].reset_index(drop=True)
            b = now.loc[now.progress_percent == progress].reset_index(drop=True)
            y = a.y_true.to_numpy(dtype=np.int8)
            before = model_metrics(
                y, a.y_prob.to_numpy(dtype=np.float64),
                a.y_pred.to_numpy(dtype=np.int8),
            )
            after = model_metrics(
                y, b.y_prob.to_numpy(dtype=np.float64),
                b.y_pred.to_numpy(dtype=np.int8),
            )
            difference = {m: float(after[m] - before[m]) for m in before}
            matched = pd.DataFrame({
                "replay_id": a.replay_id, "p1_wins": y,
                "v31_prob": a.y_prob, "v31_pred": a.y_pred,
                "gru_prob": b.y_prob, "gru_pred": b.y_pred,
            })
            bootstrap = paired_replay_bootstrap(
                matched, "v31", reps=reps,
                seed=seed + 103 * pi + 41 * mi,
            )
            if (not np.isclose(
                bootstrap["accuracy_delta"], difference["accuracy"],
                atol=1e-12, rtol=0,
            ) or not np.isclose(
                bootstrap["log_loss_delta"], difference["log_loss"],
                atol=1e-9, rtol=0,
            )):
                raise ValueError("S13 paired bootstrap/metric delta mismatch")
            stage[str(progress)] = {
                "n_replays": int(len(y)),
                "reference": before,
                "s13": after,
                "delta_s13_minus_reference": difference,
                "paired_replay_bootstrap": bootstrap,
            }
        output[mode] = stage
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index-dir", type=Path, default=SEQ / "v3_1_fixed_w9_s1")
    ap.add_argument("--dataset-zip", type=Path, default=
                    ROOT / "data/processed/starcraft_full_dataset_v3_1_fixed.zip")
    ap.add_argument("--split-json", type=Path, default=
                    ROOT / "data/processed/splits/real_v3_1_fixed_split_seed_42.json")
    ap.add_argument("--s9-aggregate", type=Path, default=S9_AGG)
    ap.add_argument("--s12-reference-dir", type=Path, default=None)
    ap.add_argument("--extra-per-replay", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--bootstrap-reps", type=int, default=1000)
    ap.add_argument("--bootstrap-seed", type=int, default=2026)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--smoke", action="store_true", help="Two batches/two epochs, no outputs")
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()
    if args.extra_per_replay != 12 or args.bootstrap_reps < 0:
        ap.error("S13 fixed experiment requires exactly 12 extra cutoffs and nonnegative bootstrap reps")
    if args.smoke and args.output_dir:
        ap.error("S13 smoke may not write results")
    if not args.smoke and args.output_dir is None:
        ap.error("S13 full run needs fresh --output-dir")
    output = None
    if args.output_dir is not None:
        output = args.output_dir.resolve()
        if output == SEQ.resolve() or not output.is_relative_to(SEQ.resolve()) or output.exists():
            ap.error("S13 output must be a new directory under results/sequential")
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        ap.error("S13 requested CUDA but CUDA is unavailable")
    config = GRUConfig(batch_size=args.batch_size, max_epochs=args.max_epochs)
    index = load_index(args.index_dir)
    if (index.manifest.get("source_zip_sha256") != V31_SHA256
            or int(index.manifest["window_length"]) != 9
            or int(index.manifest["stride"]) != 1
            or int(index.manifest["step_sec"]) != 15
            or index.manifest.get("subset_max_replays") is not None
            or len(index.features) != 64
            or len(index.ordered_rows) != 892047
            or len(index.windows) != 706952):
        raise ValueError("S13 frozen full v3.1 S1 index identity mismatch")
    part = load_replay_partitions(args.split_json, index)
    train_anchor, train_cov = select_exact_checkpoints(
        index, part, partition="train",
    )
    val_anchor, val_cov = select_exact_checkpoints(
        index, part, partition="val",
    )
    if not set(train_anchor.replay_id).isdisjoint(set(val_anchor.replay_id)):
        raise ValueError("S13 training/validation replay leakage")
    pool, extra = expand_train_checkpoint_examples(
        index, part, train_anchor, extras_per_replay=args.extra_per_replay,
    )
    if (extra["s9_anchor_examples"] != len(train_anchor)
            or len(pool) != extra["expanded_train_examples"]):
        raise ValueError("S13 training pool population corrupt")
    s9_agg = json.loads(args.s9_aggregate.read_text(encoding="utf-8"))
    if (s9_agg.get("status") != "S9_MULTISEED_VALIDATION_PASS"
            or s9_agg.get("source_zip_sha256") != V31_SHA256
            or s9_agg.get("s8_cohort_crosscheck") != "exact_key_and_label_match_PASS"
            or s9_agg.get("seeds") != [42, 43, 44]):
        raise ValueError("S13 requires frozen S8/S9 identical checkpoint audit")
    s9_dir = SEQ / f"s9_prefix_vs_last9_seed{part.seed}_v1"
    s9_refs = read_reference_predictions(
        s9_dir, val_anchor, config, part.seed, "s9",
    )
    s12_refs = None
    if args.s12_reference_dir is not None:
        s12_refs = read_reference_predictions(
            args.s12_reference_dir, val_anchor, config, part.seed, "s12",
        )
    print(json.dumps({
        "status": "S13_DYNAMIC_COHORT_PASS",
        "device": str(device), "seed": int(part.seed),
        "train_pool_examples": len(pool),
        "train_replays": train_cov["balanced_replays"],
        "train_examples_per_epoch": len(train_anchor),
        "validation_replays": val_cov["balanced_replays"],
        "s12_reference": s12_refs is not None,
        "equal_s9_epoch_budget": True,
    }, sort_keys=True), flush=True)

    prepared = prepare_sequential_data(index, part, args.dataset_zip, verify_sha=True)
    values = prepared.features_sorted
    reports, outputs, models = {}, {}, {}
    for mode in MODES:
        tr = CheckpointSequenceDataset(values, index, pool, mode)
        va = CheckpointSequenceDataset(values, index, val_anchor, mode)
        model, training = fit_dynamic_gru(
            tr, va, config, device, part.seed,
            smoke=args.smoke, verbose=True,
        )
        if training["train_examples_per_epoch"] != len(train_anchor):
            raise ValueError("S13 train optimizer budget differs from S9")
        if training["train_pool_examples"] != len(pool):
            raise ValueError("S13 expanded pool changed during training")
        reports[mode] = training
        models[mode] = model
        if not args.smoke:
            outputs[mode] = predict_checkpoint_gru(model, va, config, device)
        print(f"[S13] {mode}: selected epoch {training['selected_epoch']}: PASS", flush=True)
    if args.smoke:
        print(json.dumps({
            "status": "S13_SMOKE_PASS", "seed": part.seed,
            "train_examples_per_epoch": len(train_anchor),
            "validation_replays": val_cov["balanced_replays"],
        }, indent=2))
        return
    s13_comparison = compare_modes(
        outputs, reps=args.bootstrap_reps, seed=args.bootstrap_seed + part.seed,
    )
    s9_comparison = compare_to_reference(
        outputs, s9_refs, reps=args.bootstrap_reps,
        seed=args.bootstrap_seed + part.seed * 1009, label="S9",
    )
    s12_comparison = (
        compare_to_reference(
            outputs, s12_refs, reps=args.bootstrap_reps,
            seed=args.bootstrap_seed + part.seed * 1009 + 100000,
            label="S12",
        ) if s12_refs is not None else None
    )
    report = {
        "schema": "sc2.sequential.dynamic_cutoff_training.v1",
        "status": "S13_DYNAMIC_VALIDATION_PASS",
        "source_zip_sha256": V31_SHA256,
        "seed": int(part.seed), "device": str(device),
        "split_json": str(args.split_json.resolve()),
        "index_dir": str(args.index_dir.resolve()),
        "config": asdict(config),
        "s9_reference": str(s9_dir),
        "s12_reference": str(args.s12_reference_dir) if s12_refs is not None else None,
        "train_anchors": train_cov,
        "validation_cohort": val_cov,
        "train_pool": extra,
        "models": reports,
        "s13_prefix_minus_last9": s13_comparison,
        "s13_minus_s9_same_mode": s9_comparison,
        "s13_minus_s12_same_mode": s12_comparison,
        "test_evaluation_policy": "not_accessed_during_training",
        "bootstrap": {"unit": "replay_id", "reps": args.bootstrap_reps,
                      "seed": args.bootstrap_seed},
        "limitations": [
            "Retrospective validation reused for early stopping and S9/S12 exploratory comparisons.",
            "Four samples per train replay per epoch, equal batch size but altered endpoint diversity/order.",
            "S13 dynamic time strata are constructed from offline TRAIN replay geometry; never model inputs.",
            "S9/S12 complete-case validation excludes many short/irregular games.",
            "Seed splits overlap in the same replay corpus; confidence intervals unadjusted for multiple comparisons.",
            "Held-out test data not accessed for reporting or model selection.",
        ],
    }
    assert output is not None
    output.mkdir(parents=True, exist_ok=False)
    for mode in MODES:
        torch.save({
            "state_dict": models[mode].state_dict(),
            "mode": mode, "seed": part.seed, "config": asdict(config),
            "selected_epoch": reports[mode]["selected_epoch"],
            "feature_names": list(index.features),
            "source_zip_sha256": V31_SHA256,
            "train_pool": extra,
            "sampler": reports[mode]["sampler"],
        }, output / f"{mode}_best_model.pt")
        outputs[mode].to_csv(
            output / f"{mode}_validation_predictions.csv", index=False,
        )
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"], "seed": int(part.seed),
        "train_pool_examples": len(pool),
        "train_examples_per_epoch": len(train_anchor),
        "best_epochs": {mode: reports[mode]["selected_epoch"] for mode in MODES},
        "s13_vs_s9": {
            mode: {
                str(p): {
                    "accuracy_s9": s9_comparison[mode][str(p)]["reference"]["accuracy"],
                    "accuracy_s13": s9_comparison[mode][str(p)]["s13"]["accuracy"],
                    "delta_accuracy": s9_comparison[mode][str(p)][
                        "delta_s13_minus_reference"]["accuracy"],
                    "delta_log_loss": s9_comparison[mode][str(p)][
                        "delta_s13_minus_reference"]["log_loss"],
                    "accuracy_ci95": s9_comparison[mode][str(p)][
                        "paired_replay_bootstrap"]["accuracy_delta_ci95"],
                    "log_loss_ci95": s9_comparison[mode][str(p)][
                        "paired_replay_bootstrap"]["log_loss_delta_ci95"],
                } for p in CHECKPOINTS
            } for mode in MODES
        },
    }, indent=2, sort_keys=True))
    print(f"Saved validation-only S13 outputs to {output}")


if __name__ == "__main__":
    main()
