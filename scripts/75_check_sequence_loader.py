#!/usr/bin/env python3
"""S2 FAIR gate: validate split, train-only scaling and PyTorch batch shapes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sc2proj.sequential.data import load_index, load_replay_partitions, make_loaders, prepare_sequential_data

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index-dir", type=Path, default=ROOT / "results/sequential/v3_1_fixed_w9_s1")
    ap.add_argument("--dataset-zip", type=Path, default=ROOT / "data/processed/starcraft_full_dataset_v3_1_fixed.zip")
    ap.add_argument("--split-json", type=Path, default=ROOT / "data/processed/splits/real_v3_1_fixed_split_seed_42.json")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--skip-sha", action="store_true", help="Only for synthetic development; never use for FAIR validation")
    args = ap.parse_args()
    if args.batch_size < 1:
        ap.error("--batch-size must be positive")
    index = load_index(args.index_dir)
    partition = load_replay_partitions(args.split_json, index)
    prepared = prepare_sequential_data(index, partition, args.dataset_zip, verify_sha=not args.skip_sha)
    loaders = make_loaders(prepared, batch_size=args.batch_size, num_workers=args.num_workers)
    checks = {}
    for part, loader in loaders.items():
        if len(loader.dataset) == 0:
            checks[part] = None
            continue
        features, labels = next(iter(loader))
        shape = tuple(features.shape)
        expected = (min(args.batch_size, len(loader.dataset)), int(index.manifest["window_length"]), len(index.features))
        if shape != expected or labels.shape != (shape[0],) or not np.isfinite(features.numpy()).all():
            raise ValueError(f"Invalid {part} batch: observed={shape}, expected={expected}")
        checks[part] = {"features": list(shape), "labels": list(labels.shape)}
    print(json.dumps({
        "status": "S2_LOADER_PASS",
        "seed": partition.seed,
        "source_sha256": index.manifest["source_zip_sha256"],
        "window_length": int(index.manifest["window_length"]),
        "feature_count": len(index.features),
        "partition_counts": partition.counts(index),
        "scaler_fitted_rows": int(partition.row_masks["train"].sum()),
        "scaler_n_samples_seen": int(prepared.scaler.n_samples_seen_),
        "batches": checks,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
