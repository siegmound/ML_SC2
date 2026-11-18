#!/usr/bin/env python3
"""S4: train a GRU using replay-aware S2 windows; validation only, never test.

Run --smoke first on FAIR. Full runs require a fresh output directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from sc2proj.sequential.data import (
    load_index, load_replay_partitions, prepare_sequential_data,
)
from sc2proj.sequential.gru import GRUConfig, fit_gru, validation_predictions

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/sequential"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index-dir", type=Path, default=RESULTS / "v3_1_fixed_w9_s1")
    ap.add_argument("--dataset-zip", type=Path,
                    default=ROOT / "data/processed/starcraft_full_dataset_v3_1_fixed.zip")
    ap.add_argument("--split-json", type=Path,
                    default=ROOT / "data/processed/splits/real_v3_1_fixed_split_seed_42.json")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--hidden-size", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="At most two train/val batches per epoch, two epochs; no saved outputs")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Required for full runs; must be a new path under results/sequential")
    args = ap.parse_args()

    if args.smoke and args.output_dir is not None:
        ap.error("--smoke cannot write output files")
    if not args.smoke and args.output_dir is None:
        ap.error("Full runs require --output-dir (a new directory under results/sequential/)")
    output = None
    if args.output_dir is not None:
        output = args.output_dir.resolve()
        root = RESULTS.resolve()
        if output == root or not output.is_relative_to(root):
            ap.error("--output-dir must be a new subdirectory under results/sequential")
        if output.exists():
            ap.error(f"Refusing to overwrite existing output directory: {output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        ap.error("CUDA requested but not available")
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    try:
        config = GRUConfig(
            hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout,
            batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
            max_epochs=args.max_epochs, patience=args.patience, min_delta=args.min_delta,
            grad_clip=args.grad_clip, num_workers=args.num_workers,
        )
    except ValueError as exc:
        ap.error(str(exc))

    index = load_index(args.index_dir)
    if int(index.manifest["window_length"]) != 9:
        ap.error("S4 first GRU experiment is fixed to the 9-snapshot index")
    split = load_replay_partitions(args.split_json, index)
    prepared = prepare_sequential_data(index, split, args.dataset_zip, verify_sha=True)
    model, report = fit_gru(prepared, config, device, smoke=args.smoke)
    report.update({
        "dataset": "v3_1_fixed",
        "window_length": 9,
        "feature_count": len(index.features),
        "source_zip_sha256": index.manifest["source_zip_sha256"],
        "index_dir": str(args.index_dir.resolve()),
        "split_json": str(args.split_json.resolve()),
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "caveat": "Validation selected; historical baseline test has been inspected previously.",
    })
    if args.smoke:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    predictions, metrics = validation_predictions(
        model, prepared, index, split, device, config.batch_size
    )
    report["validation_metrics"] = metrics
    report["status"] = "S4_VALIDATION_PASS"
    assert output is not None
    output.mkdir(parents=True, exist_ok=False)
    torch.save({
        "state_dict": model.state_dict(),
        "feature_count": len(index.features),
        "feature_names": list(index.features),
        "seed": split.seed,
        "window_length": 9,
        "selected_epoch": report["selected_epoch"],
        "model_config": report["config"],
        "source_zip_sha256": report["source_zip_sha256"],
    }, output / "best_model.pt")
    predictions.to_csv(output / "validation_predictions.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Saved {len(predictions):,} matched validation endpoints to {output}")
    print("No test predictions or test metrics were generated.")


if __name__ == "__main__":
    main()
