#!/usr/bin/env python3
"""S12: controlled 4-anchor + extra-train-cutoff GRU ablation (validation only).

Only change vs S9 is the number of TRAIN cutoffs from the SAME S9 replay IDs.
S9 train anchors are retained; both S12 arms get identical expanded endpoints.
S9 validation complete-case checkpoint cohort/config/scaler remain unchanged.
No historical test predictions or labels are ever evaluated.
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
from sc2proj.sequential.extra_prefixes import expand_train_checkpoint_examples
from sc2proj.sequential.gru import GRUConfig
from sc2proj.sequential.prefix import (
    MODES, CheckpointSequenceDataset, compare_modes,
    fit_checkpoint_gru, predict_checkpoint_gru,
)
from sc2proj.sequential.progress import CHECKPOINTS, select_exact_checkpoints
from sc2proj.sequential.variants import V31_SHA256

ROOT = Path(__file__).resolve().parents[1]
SEQ = ROOT / "results/sequential"
S9_AGG = SEQ / "s9_prefix_vs_last9_multiseed_v1/summary.json"


def verify_s9_frozen_reference(seed: int, anchors: pd.DataFrame,
                               valid: pd.DataFrame, config: GRUConfig,
                               *, agg_path: Path):
    summary = json.loads(agg_path.read_text(encoding="utf-8"))
    if (summary.get("schema") != "sc2.sequential.prefix_multiseed_validation.v1"
            or summary.get("status") != "S9_MULTISEED_VALIDATION_PASS"
            or summary.get("s8_cohort_crosscheck") != "exact_key_and_label_match_PASS"
            or summary.get("source_zip_sha256") != V31_SHA256
            or summary.get("seeds") != [42, 43, 44]):
        raise ValueError("S12 requires completed S8/S9 validation checkpoint audit")
    reference_dir = SEQ / f"s9_prefix_vs_last9_seed{seed}_v1"
    s9 = json.loads((reference_dir / "summary.json").read_text(encoding="utf-8"))
    if (s9.get("schema") != "sc2.sequential.prefix_vs_last9.v1"
            or s9.get("status") != "S9_MATCHED_VALIDATION_PASS"
            or s9.get("seed") != seed
            or s9.get("source_zip_sha256") != V31_SHA256
            or s9.get("train_examples") != len(anchors)
            or s9.get("validation_examples") != len(valid)
            or s9.get("train_coverage", {}).get("balanced_replays") != anchors.replay_id.nunique()
            or s9.get("validation_coverage", {}).get("balanced_replays") != valid.replay_id.nunique()
            or s9.get("settings") != asdict(config)):
        raise ValueError("S12 frozen S9 provenance/config/cohort mismatch")
    prediction_baselines = {}
    expected = valid.sort_values(["replay_id", "progress_percent"], kind="stable").reset_index(drop=True)
    for mode in MODES:
        arm = s9.get("arms", {}).get(mode, {})
        if (arm.get("smoke") is not False or arm.get("config") != asdict(config)
                or arm.get("source_zip_sha256") != V31_SHA256
                or arm.get("checkpoint_selection") != "validation_bce_only"
                or arm.get("test_evaluation_policy") != "not_accessed_during_training"):
            raise ValueError(f"S12 S9 arm {mode} metadata altered")
        old = pd.read_csv(
            reference_dir / f"{mode}_validation_predictions.csv",
            dtype={"replay_id": "string"},
        ).sort_values(["replay_id", "progress_percent"], kind="stable").reset_index(drop=True)
        if len(old) != len(expected) or old.duplicated(
            ["replay_id", "progress_percent"]
        ).any():
            raise ValueError(f"S12 S9 baseline {mode} checkpoint coverage changed")
        for a, b in (("replay_id", "replay_id"), ("time_sec", "time_sec"),
                     ("progress_percent", "progress_percent"),
                     ("y_true", "p1_wins")):
            if not np.array_equal(old[a].to_numpy(), expected[b].to_numpy()):
                raise ValueError(f"S12 S9 baseline {mode} mismatched validation {a}")
        p = old.y_prob.to_numpy(dtype=np.float64)
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() or not np.array_equal(
            (p >= .5).astype(np.int8), old.y_pred.to_numpy(dtype=np.int8)
        ):
            raise ValueError(f"S12 S9 baseline {mode} probability corruption")
        prediction_baselines[mode] = old
    return prediction_baselines, reference_dir


def compare_updated_to_s9(updated: dict, references: dict, *, reps: int, seed: int):
    """Strictly paired S12-minus-S9 within matching model, replay and stage."""
    results = {}
    fields = ("replay_id", "time_sec", "progress_percent", "y_true")
    for mode in MODES:
        new = updated[mode].sort_values(
            ["replay_id", "progress_percent"], kind="stable"
        ).reset_index(drop=True)
        old = references[mode]
        for field in fields:
            if not np.array_equal(new[field].to_numpy(), old[field].to_numpy()):
                raise ValueError(f"S12 {mode}: validation endpoint drift in {field}")
        results[mode] = {}
        for i, progress in enumerate(CHECKPOINTS):
            a = old.loc[old.progress_percent == progress].reset_index(drop=True)
            b = new.loc[new.progress_percent == progress].reset_index(drop=True)
            y = a.y_true.to_numpy(dtype=np.int8)
            before = model_metrics(y, a.y_prob.to_numpy(dtype=np.float64),
                                   a.y_pred.to_numpy(dtype=np.int8))
            after = model_metrics(y, b.y_prob.to_numpy(dtype=np.float64),
                                  b.y_pred.to_numpy(dtype=np.int8))
            delta = {m: float(after[m] - before[m]) for m in before}
            paired = pd.DataFrame({
                "replay_id": a.replay_id, "p1_wins": y,
                "v31_prob": a.y_prob, "v31_pred": a.y_pred,
                "gru_prob": b.y_prob, "gru_pred": b.y_pred,
            })
            ci = paired_replay_bootstrap(
                paired, "v31", reps=reps,
                seed=seed * 1009 + i + (0 if mode == "last9" else 101),
            )
            if (not np.isclose(delta["accuracy"], ci["accuracy_delta"], atol=1e-12, rtol=0)
                    or not np.isclose(delta["log_loss"], ci["log_loss_delta"], atol=1e-9, rtol=0)):
                raise ValueError("S12 paired confidence interval verification failed")
            results[mode][str(progress)] = {
                "n_replays": len(a),
                "s9_model": before,
                "s12_model": after,
                "delta_s12_minus_s9": delta,
                "paired_replay_bootstrap": ci,
            }
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index-dir", type=Path, default=SEQ / "v3_1_fixed_w9_s1")
    ap.add_argument("--dataset-zip", type=Path,
                    default=ROOT / "data/processed/starcraft_full_dataset_v3_1_fixed.zip")
    ap.add_argument("--split-json", type=Path,
                    default=ROOT / "data/processed/splits/real_v3_1_fixed_split_seed_42.json")
    ap.add_argument("--s9-aggregate", type=Path, default=S9_AGG)
    ap.add_argument("--extra-per-replay", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--bootstrap-reps", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true", help="Two batches and two epochs; no output")
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()
    if not 0 <= args.extra_per_replay <= 40 or args.bootstrap_reps < 0:
        ap.error("Invalid extra cutoffs or bootstrap repetitions")
    if args.smoke and args.output_dir is not None:
        ap.error("Smoke does not write output")
    if not args.smoke and args.output_dir is None:
        ap.error("Full S12 requires a fresh --output-dir")
    output = None
    if args.output_dir is not None:
        output = args.output_dir.resolve()
        if output == SEQ.resolve() or not output.is_relative_to(SEQ.resolve()) or output.exists():
            ap.error("Output must be a fresh subdirectory under results/sequential")
    config = GRUConfig(batch_size=args.batch_size, max_epochs=args.max_epochs)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        ap.error("CUDA explicitly requested but torch.cuda.is_available() is false")
    index = load_index(args.index_dir)
    if (index.manifest.get("source_zip_sha256") != V31_SHA256
            or int(index.manifest["window_length"]) != 9
            or int(index.manifest["stride"]) != 1
            or int(index.manifest["step_sec"]) != 15
            or index.manifest.get("subset_max_replays") is not None
            or len(index.features) != 64
            or len(index.ordered_rows) != 892047
            or len(index.windows) != 706952):
        raise ValueError("S12 requires frozen full v3.1 S1 index")
    partition = load_replay_partitions(args.split_json, index)
    train_anchors, train_cov = select_exact_checkpoints(index, partition, partition="train")
    valid, val_cov = select_exact_checkpoints(index, partition, partition="val")
    train_extra, expansion = expand_train_checkpoint_examples(
        index, partition, train_anchors,
        extras_per_replay=args.extra_per_replay,
    )
    if not set(train_extra.replay_id).isdisjoint(set(valid.replay_id)):
        raise ValueError("S12 training and validation replay overlap")
    original, reference_dir = verify_s9_frozen_reference(
        partition.seed, train_anchors, valid, config, agg_path=args.s9_aggregate,
    )
    print(json.dumps({
        "status": "S12_EXPANDED_TRAIN_COHORT_PASS",
        "seed": partition.seed, "train_expansion": expansion,
        "validation_balanced_replays": val_cov["balanced_replays"],
        "device": str(device), "same_as_s9_config": True,
    }, sort_keys=True), flush=True)

    # Exactly the same S9 train-only StandardScaler protocol.
    prepared = prepare_sequential_data(index, partition, args.dataset_zip, verify_sha=True)
    values = prepared.features_sorted
    reports, pred, fitted = {}, {}, {}
    for mode in MODES:
        tr = CheckpointSequenceDataset(values, index, train_extra, mode)
        va = CheckpointSequenceDataset(values, index, valid, mode)
        model, training = fit_checkpoint_gru(
            tr, va, config, device, partition.seed,
            smoke=args.smoke, verbose=True,
        )
        training.update({
            "experiment": "S12",
            "source_zip_sha256": V31_SHA256,
            "feature_names": list(index.features),
            "training_expansion": expansion,
            "s9_reference": str(reference_dir),
            "test_evaluation_policy": "not_accessed_during_training",
        })
        reports[mode] = training
        fitted[mode] = model
        if not args.smoke:
            pred[mode] = predict_checkpoint_gru(model, va, config, device)
        print(f"[S12] arm={mode} selected_epoch={training['selected_epoch']}: PASS", flush=True)
    if args.smoke:
        print(json.dumps({"status": "S12_SMOKE_PASS", "seed": partition.seed,
                          "training_expansion": expansion}, indent=2))
        return
    within = compare_modes(pred, reps=args.bootstrap_reps, seed=2026 + partition.seed)
    against_s9 = compare_updated_to_s9(
        pred, original, reps=args.bootstrap_reps, seed=2026 + partition.seed,
    )
    report = {
        "schema": "sc2.sequential.extra_prefix_training.v1",
        "status": "S12_EXPANDED_TRAIN_VALIDATION_PASS",
        "seed": partition.seed,
        "source_zip_sha256": V31_SHA256,
        "split_json": str(args.split_json.resolve()),
        "index_dir": str(args.index_dir.resolve()),
        "device": str(device),
        "config": asdict(config),
        "train_cohort": train_cov,
        "validation_cohort": val_cov,
        "training_expansion": expansion,
        "s9_reference_dir": str(reference_dir),
        "s12_arms": reports,
        "s12_prefix_minus_last9": within,
        "s12_minus_s9_same_mode": against_s9,
        "test_evaluation_policy": "not_accessed_during_training",
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "limitations": [
            "Retrospective validation and training checkpoint selection; test not accessed.",
            "Same S9 train replay IDs, four original train anchors and four val checkpoints; more train endpoints change gradient-step count per epoch.",
            "Additional train endpoints are selected using the full train replay's observed range only for offline sampling, never supplied as model features.",
            "Extra valid endpoints can be fewer than requested when source snapshot sequences contain gaps.",
            "Validation replay population requires all four nine-step checkpoints; short or gappy replay bias remains.",
            "Three random seed splits overlap in source replay corpus; do not treat seed estimates as independent.",
        ],
    }
    assert output is not None
    output.mkdir(parents=True, exist_ok=False)
    for mode in MODES:
        torch.save({
            "state_dict": fitted[mode].state_dict(), "mode": mode,
            "feature_names": list(index.features), "seed": partition.seed,
            "config": asdict(config), "selected_epoch": reports[mode]["selected_epoch"],
            "source_zip_sha256": V31_SHA256,
            "training_expansion": expansion,
        }, output / f"{mode}_best_model.pt")
        pred[mode].to_csv(output / f"{mode}_validation_predictions.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"], "seed": partition.seed,
        "validation_replays": val_cov["balanced_replays"],
        "extra_training_examples": expansion["extra_examples"],
        "per_checkpoint_vs_s9": {
            mode: {
                str(p): {
                    "accuracy_s9": against_s9[mode][str(p)]["s9_model"]["accuracy"],
                    "accuracy_s12": against_s9[mode][str(p)]["s12_model"]["accuracy"],
                    "delta_accuracy": against_s9[mode][str(p)]["delta_s12_minus_s9"]["accuracy"],
                    "delta_log_loss": against_s9[mode][str(p)]["delta_s12_minus_s9"]["log_loss"],
                }
                for p in CHECKPOINTS
            }
            for mode in MODES
        },
    }, indent=2, sort_keys=True))
    print(f"Saved independent S12 validation-only artifacts to {output}")


if __name__ == "__main__":
    main()
