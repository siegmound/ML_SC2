#!/usr/bin/env python3
"""S9: fit matched 9-last-snapshot and growing-prefix GRUs at 25/50/75/90%.

Both models use same train/validation replay cohorts, labels, checkpoint times,
optimizer settings, recurrent width and random seed. Test is not iterated.
Full replay duration is used only by offline example selector, not as a feature.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from sc2proj.sequential.data import (
    load_index, load_replay_partitions, prepare_sequential_data,
)
from sc2proj.sequential.gru import GRUConfig
from sc2proj.sequential.prefix import (
    MODES, CheckpointSequenceDataset, compare_modes,
    fit_checkpoint_gru, predict_checkpoint_gru,
)
from sc2proj.sequential.progress import CHECKPOINTS, select_exact_checkpoints
from sc2proj.sequential.variants import V31_SHA256

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/sequential"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index-dir", type=Path, default=RESULTS / "v3_1_fixed_w9_s1")
    ap.add_argument("--dataset-zip", type=Path,
                    default=ROOT / "data/processed/starcraft_full_dataset_v3_1_fixed.zip")
    ap.add_argument("--split-json", type=Path,
                    default=ROOT / "data/processed/splits/real_v3_1_fixed_split_seed_42.json")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--bootstrap-reps", type=int, default=1000)
    ap.add_argument("--bootstrap-seed", type=int, default=2026)
    ap.add_argument("--smoke", action="store_true", help="Two batches per arm for two epochs; no writes")
    ap.add_argument("--output-dir", type=Path, default=None, help="Fresh results/sequential/ subdirectory")
    args = ap.parse_args()
    if args.bootstrap_reps < 0:
        ap.error("Bootstrap repetitions must be nonnegative")
    if args.smoke and args.output_dir is not None:
        ap.error("--smoke cannot write artifacts")
    if not args.smoke and args.output_dir is None:
        ap.error("Full run requires a fresh --output-dir")
    output = None
    if args.output_dir is not None:
        output = args.output_dir.resolve()
        base = RESULTS.resolve()
        if output == base or not output.is_relative_to(base) or output.exists():
            ap.error("Refusing to overwrite existing results or write outside results/sequential/")
    if args.device == "cuda" and not torch.cuda.is_available():
        ap.error("CUDA requested but unavailable")
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    config = GRUConfig(batch_size=args.batch_size, max_epochs=args.max_epochs)
    index = load_index(args.index_dir)
    if (index.manifest.get("source_zip_sha256") != V31_SHA256
            or int(index.manifest["window_length"]) != 9
            or int(index.manifest["stride"]) != 1
            or int(index.manifest["step_sec"]) != 15
            or index.manifest.get("subset_max_replays") is not None
            or len(index.features) != 64 or len(index.ordered_rows) != 892047
            or len(index.windows) != 706952):
        ap.error("S9 requires frozen full v3.1 S1 index (exact SHA, counts, 9x15s)")
    partitions = load_replay_partitions(args.split_json, index)
    train_cp, train_coverage = select_exact_checkpoints(index, partitions, partition="train")
    val_cp, val_coverage = select_exact_checkpoints(index, partitions, partition="val")
    if not set(train_cp.replay_id).isdisjoint(set(val_cp.replay_id)):
        raise ValueError("Replay leakage: training and validation checkpoint cohorts intersect")
    for checkpoint in (train_cp, val_cp):
        if len(checkpoint) != len(CHECKPOINTS) * checkpoint.replay_id.nunique():
            raise ValueError("Incomplete training/validation checkpoint quartet")
    print(json.dumps({
        "status": "S9_CHECKPOINT_COHORT_PASS", "seed": partitions.seed,
        "train": train_coverage, "validation": val_coverage,
        "checkpoints": list(CHECKPOINTS),
        "partition": "train+val_only", "device": str(device),
    }, sort_keys=True), flush=True)

    # Single shared train-only scaler for both arms, no test iteration.
    prepared = prepare_sequential_data(index, partitions, args.dataset_zip, verify_sha=True)
    values = prepared.features_sorted
    reports, predictions, models = {}, {}, {}
    for mode in MODES:
        train_data = CheckpointSequenceDataset(values, index, train_cp, mode)
        val_data = CheckpointSequenceDataset(values, index, val_cp, mode)
        model, summary = fit_checkpoint_gru(
            train_data, val_data, config, device, partitions.seed,
            smoke=args.smoke, verbose=True,
        )
        summary.update({
            "source_zip_sha256": index.manifest["source_zip_sha256"],
            "dataset": "v3_1_fixed", "features": list(index.features),
            "feature_count": len(index.features),
            "checkpoint_percents": list(CHECKPOINTS),
            "train_coverage": train_coverage, "validation_coverage": val_coverage,
            "index_dir": str(args.index_dir.resolve()),
            "split_json": str(args.split_json.resolve()),
            "python_version": sys.version.split()[0], "torch_version": torch.__version__,
            "caveat": (
                "Offline fraction checkpoints, full replay N not model input; "
                "checkpoint sample selection and validation checkpoint selection "
                "are retrospective. Each model sees only snapshots at/before endpoint. "
                "No historical test evaluation."),
        })
        reports[mode] = summary
        models[mode] = model
        if not args.smoke:
            predictions[mode] = predict_checkpoint_gru(model, val_data, config, device)
        print(f"[S9] arm={mode} seed={partitions.seed} selected_epoch="
              f"{summary['selected_epoch']}: PASS", flush=True)

    if args.smoke:
        print(json.dumps({"status": "S9_SMOKE_PASS", "models": reports},
                         indent=2, sort_keys=True))
        return
    paired = compare_modes(predictions, reps=args.bootstrap_reps,
                           seed=args.bootstrap_seed + partitions.seed)
    report = {
        "schema": "sc2.sequential.prefix_vs_last9.v1",
        "status": "S9_MATCHED_VALIDATION_PASS",
        "seed": partitions.seed, "device": str(device),
        "source_zip_sha256": index.manifest["source_zip_sha256"],
        "model_input": (
            "last9 or complete observed prefix ending at exact matching checkpoint; "
            "no percent or full-game duration supplied to model"),
        "checkpoint_definition": "ceil(p * full_replay_snapshot_count/100), offline only",
        "train_coverage": train_coverage, "validation_coverage": val_coverage,
        "train_examples": len(train_cp), "validation_examples": len(val_cp),
        "settings": config.__dict__, "arms": reports,
        "paired_checkpoint_metrics": paired,
        "bootstrap": {"reps": args.bootstrap_reps, "seed": args.bootstrap_seed,
                      "resampling_unit": "whole_replay"},
        "limitations": [
            "Retrospective validation, not untouched test.",
            "Both arms are freshly trained on same four checkpoints; S8 historical GRU used all valid 9-step windows.",
            "Only complete-case replays with valid 9-step checkpoints at all four percentages participate.",
            "Snapshot time gaps in historical replay prefixes are retained as missing observations, not imputed.",
            "Seed splits reuse the same underlying corpus, so means across seeds are not independent replicates.",
        ],
    }
    assert output is not None
    output.mkdir(parents=True, exist_ok=False)
    for mode in MODES:
        torch.save({
            "state_dict": models[mode].state_dict(), "mode": mode,
            "feature_names": list(index.features), "seed": partitions.seed,
            "config": reports[mode]["config"], "selected_epoch": reports[mode]["selected_epoch"],
            "source_zip_sha256": index.manifest["source_zip_sha256"],
            "train_checkpoints": list(CHECKPOINTS),
        }, output / f"{mode}_best_model.pt")
        predictions[mode].to_csv(output / f"{mode}_validation_predictions.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "seed": partitions.seed,
        "train_replays": train_coverage["balanced_replays"],
        "val_replays": val_coverage["balanced_replays"],
        "per_checkpoint": {
            p: {"accuracy": {mode: paired[p]["models"][mode]["accuracy"] for mode in MODES},
                "delta_prefix_minus_last9_accuracy": paired[p][
                    "delta_prefix_minus_last9"]["accuracy"],
                "delta_prefix_minus_last9_log_loss": paired[p][
                    "delta_prefix_minus_last9"]["log_loss"],
                "accuracy_ci95": paired[p]["paired_replay_bootstrap"]["accuracy_delta_ci95"]}
            for p in map(str, CHECKPOINTS)
        },
    }, indent=2, sort_keys=True))
    print(f"Saved S9 validation-only report to {output}")


if __name__ == "__main__":
    main()
