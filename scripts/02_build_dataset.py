from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sc2proj.builder import BuildConfig, DatasetBuilder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replays-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--dataset-version", type=str, default="v1")
    parser.add_argument("--snapshot-step-sec", type=int, default=15)
    parser.add_argument("--early-cut-sec", type=int, default=120)
    parser.add_argument("--min-match-sec", type=int, default=180)
    parser.add_argument("--max-match-sec", type=int, default=1800)
    args = parser.parse_args()

    config = BuildConfig(
        dataset_name=args.dataset_name,
        dataset_version=args.dataset_version,
        snapshot_step_sec=args.snapshot_step_sec,
        early_cut_sec=args.early_cut_sec,
        min_match_sec=args.min_match_sec,
        max_match_sec=args.max_match_sec,
    )
    builder = DatasetBuilder(config)
    replay_paths = sorted(args.replays_dir.rglob("*.SC2Replay"))
    output_zip = PROJECT_ROOT / "data" / "processed" / f"{args.dataset_name}.zip"
    manifest_path = PROJECT_ROOT / "data" / "processed" / "manifests" / f"{args.dataset_name}_manifest.json"
    audit_csv_path = PROJECT_ROOT / "data" / "interim" / "parser_audit" / f"{args.dataset_name}_replay_audit.csv"
    manifest = builder.build_from_replays(replay_paths, output_zip, manifest_path, audit_csv_path)
    print({k: manifest[k] for k in ("dataset_name", "dataset_version", "number_rows", "number_replays")})


if __name__ == "__main__":
    main()
