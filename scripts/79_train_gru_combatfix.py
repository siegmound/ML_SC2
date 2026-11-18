#!/usr/bin/env python3
"""S6: train the unchanged S4 GRU on combatfix, after exact S1 parity checks.

The train/val/test replay groups and 9-snapshot endpoints are identical to S4.
Only v3.2 feature values and feature count differ; no test scores are accessed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from sc2proj.sequential.data import load_index, load_replay_partitions, prepare_sequential_data
from sc2proj.sequential.gru import GRUConfig, fit_gru, validation_predictions
from sc2proj.sequential.variants import verify_combatfix_parity

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/sequential"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-index-dir", type=Path, default=RESULTS / "v3_1_fixed_w9_s1")
    ap.add_argument("--index-dir", type=Path, default=RESULTS / "v3_2_combatfix_w9_s1")
    ap.add_argument("--dataset-zip", type=Path,
                    default=ROOT / "data/processed/starcraft_full_dataset_v3_2_combatfix.zip")
    ap.add_argument("--split-json", type=Path,
                    default=ROOT / "data/processed/splits/real_v3_1_fixed_split_seed_42.json")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--smoke", action="store_true", help="Two batches x two epochs; no writes")
    ap.add_argument("--output-dir", type=Path, default=None, help="Fresh path under results/sequential/")
    args = ap.parse_args()
    if args.smoke and args.output_dir is not None:
        ap.error("Smoke must not write artifacts")
    if not args.smoke and args.output_dir is None:
        ap.error("Full S6 fit requires --output-dir")
    output = None
    if args.output_dir is not None:
        output = args.output_dir.resolve()
        base = RESULTS.resolve()
        if output == base or not output.is_relative_to(base) or output.exists():
            ap.error("Refusing output outside results/sequential/ or existing directory")
    if args.device == "cuda" and not torch.cuda.is_available():
        ap.error("CUDA requested but unavailable")
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )

    reference = load_index(args.reference_index_dir)
    candidate = load_index(args.index_dir)
    parity = verify_combatfix_parity(reference, candidate, require_full=True)
    print(json.dumps(parity, sort_keys=True), flush=True)
    split_ref = load_replay_partitions(args.split_json, reference)
    split = load_replay_partitions(args.split_json, candidate)
    if split.seed != split_ref.seed or any(
        not (split.window_positions[p] == split_ref.window_positions[p]).all()
        for p in ("train", "val", "test")
    ):
        raise ValueError("S6 split/window positions differ from the original S4 run")
    # Same architecture, learning rate, patience and regularization as S4.
    # Smoke uses the same 32-batch setting as the historical S4 smoke gate.
    config = GRUConfig(batch_size=32) if args.smoke else GRUConfig()
    prepared = prepare_sequential_data(candidate, split, args.dataset_zip, verify_sha=True)
    model, report = fit_gru(prepared, config, device, smoke=args.smoke, verbose=True)
    report.update({
        "status": "S6_SMOKE_PASS" if args.smoke else "S6_FIT_PASS",
        "dataset": "v3_2_combatfix",
        "comparison_dataset": "v3_1_fixed",
        "window_length": 9,
        "feature_count": len(candidate.features),
        "source_zip_sha256": candidate.manifest["source_zip_sha256"],
        "reference_zip_sha256": reference.manifest["source_zip_sha256"],
        "index_dir": str(args.index_dir.resolve()),
        "reference_index_dir": str(args.reference_index_dir.resolve()),
        "split_json": str(args.split_json.resolve()),
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "parity": parity,
        "caveat": "Validation-only retrospective variant comparison; no new untouched test.",
    })
    if args.smoke:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    pred, metrics = validation_predictions(model, prepared, candidate, split, device,
                                           config.batch_size)
    report["validation_metrics"] = metrics
    report["status"] = "S6_VALIDATION_PASS"
    assert output is not None
    output.mkdir(parents=True, exist_ok=False)
    torch.save({
        "state_dict": model.state_dict(),
        "feature_count": len(candidate.features),
        "feature_names": list(candidate.features),
        "seed": split.seed,
        "window_length": 9,
        "selected_epoch": report["selected_epoch"],
        "model_config": report["config"],
        "source_zip_sha256": report["source_zip_sha256"],
        "comparison_dataset": "v3_1_fixed",
    }, output / "best_model.pt")
    pred.to_csv(output / "validation_predictions.csv", index=False)
    (output / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                         encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Saved {len(pred):,} combatfix validation endpoints to {output}")
    print("No test predictions or test metrics were generated.")


if __name__ == "__main__":
    main()
