#!/usr/bin/env python3
"""S1/S6: build a replay-safe index for either frozen dataset variant (no training)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sc2proj.sequential import build_sequence_index, csv_schema_from_zip, read_snapshot_metadata, validate_schema

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = {
    variant: ROOT / f"data/processed/starcraft_full_dataset_{variant}.zip"
    for variant in ("v3_1_fixed", "v3_2_combatfix")
}
DEFAULT_FEATURES = ROOT / "data/processed/manifests/b3_feature_lineage_v1.json"
OUTPUT_BASE = ROOT / "results/sequential"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-variant", choices=tuple(DEFAULT_DATASETS), default="v3_1_fixed")
    parser.add_argument("--dataset-zip", type=Path, default=None)
    parser.add_argument("--feature-manifest", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--window", type=int, choices=(9, 17, 33), default=9)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--step-sec", type=int, default=15)
    parser.add_argument("--max-replays", type=int, default=None, help="Deterministic smoke subset; omit for full data")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate and count windows without writing")
    args = parser.parse_args()
    if args.max_replays is not None and args.max_replays < 1:
        parser.error("--max-replays must be positive")
    dataset_zip = args.dataset_zip or DEFAULT_DATASETS[args.dataset_variant]
    schema = csv_schema_from_zip(dataset_zip)
    features = validate_schema(schema, args.feature_manifest, dataset_variant=args.dataset_variant)
    meta = read_snapshot_metadata(dataset_zip)
    if args.max_replays is not None:
        chosen = set(sorted(meta["replay_id"].unique())[: args.max_replays])
        meta = meta.loc[meta["replay_id"].isin(chosen)].copy()
    index = build_sequence_index(meta, features, args.window, args.stride, args.step_sec)
    summary = index.stats()
    summary["dataset_zip"] = str(dataset_zip.resolve())
    summary["dataset_variant"] = args.dataset_variant
    summary["subset_max_replays"] = args.max_replays
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.dry_run:
        return

    output = args.output_dir or OUTPUT_BASE / f"{args.dataset_variant}_w{args.window}_s{args.stride}"
    output = output.resolve()
    if output == OUTPUT_BASE.resolve() or not output.is_relative_to(OUTPUT_BASE.resolve()):
        parser.error(f"Output must be inside {OUTPUT_BASE}")
    if output.exists():
        parser.error(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    index.ordered_rows.to_parquet(output / "ordered_rows.parquet", index=False)
    index.windows.to_parquet(output / "window_index.parquet", index=False)
    digest = hashlib.sha256()
    with dataset_zip.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    summary.update({
        "schema": "sc2.sequential.window_index.v1",
        "source_zip_sha256": digest.hexdigest(),
        "features": list(features),
        "ordered_rows_file": "ordered_rows.parquet",
        "window_index_file": "window_index.parquet",
        "stop_row_exclusive": True,
        "train_validation_test_partitioned": False,
        "note": "S1 is a replay-safe global index, not a trained model or split-aware loader (S2).",
    })
    (output / "manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved read-only index to {output}")


if __name__ == "__main__":
    main()
