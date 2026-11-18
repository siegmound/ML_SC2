#!/usr/bin/env python3
"""S3 FAIR audit: compare frozen baseline predictions at S1 window endpoints.

No model fitting, no recalibration, no held-out model selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sc2proj.sequential.data import load_index, load_replay_partitions
from sc2proj.sequential.baselines import compare_archived_baselines

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index-dir", type=Path, default=ROOT / "results/sequential/v3_1_fixed_w9_s1")
    ap.add_argument("--common-index-dir", action="append", type=Path, default=[],
                    help="Optional longer S1 index; retain only endpoints shared with all indexes")
    ap.add_argument("--split-json", type=Path, default=ROOT / "data/processed/splits/real_v3_1_fixed_split_seed_42.json")
    ap.add_argument("--results-root", type=Path, default=ROOT / "results")
    ap.add_argument("--partition", choices=("val", "test"), default="val")
    ap.add_argument("--models", nargs="+", choices=("rf", "xgb", "mlp"), default=["rf", "xgb", "mlp"])
    ap.add_argument("--output-dir", type=Path, default=None, help="Optional fresh path within results/sequential/; no overwrites")
    args = ap.parse_args()
    index = load_index(args.index_dir)
    partitioned = load_replay_partitions(args.split_json, index)
    others = [load_index(p) for p in args.common_index_dir]
    if others:
        for path, other in zip(args.common_index_dir, others):
            if other.manifest["source_zip_sha256"] != index.manifest["source_zip_sha256"] or tuple(other.features) != tuple(index.features):
                ap.error(f"Incompatible dataset or feature schema in common index: {path}")
    report, aligned = compare_archived_baselines(index, partitioned, args.partition,
                                                args.results_root, models=tuple(args.models), common_indexes=others)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output_dir is not None:
        output = args.output_dir.resolve()
        base = (ROOT / "results/sequential").resolve()
        if output == base or not output.is_relative_to(base):
            ap.error("Output must be a new subdirectory under results/sequential/")
        if output.exists():
            ap.error(f"Refusing to overwrite existing output directory: {output}")
        output.mkdir(parents=True)
        (output / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        aligned.to_parquet(output / "matched_predictions.parquet", index=False)
        print(f"Saved {len(aligned):,} matched endpoints to {output}")


if __name__ == "__main__":
    main()
