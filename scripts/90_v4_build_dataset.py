#!/usr/bin/env python3
"""Build an independent v4 clean-room dataset using actual raw SC2Replay files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from sc2proj.v4_dataset import build_dataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", type=Path, required=True,
                   help="Folder of real SC2Replay files; not a frozen CSV")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="A NEW results/v4_* folder; existing outputs are not overwritten")
    p.add_argument("--max-replays", type=int, default=None,
                   help="Optional deterministic content-hash selected SMOKE subset")
    p.add_argument("--step-sec", type=int, default=15)
    p.add_argument("--early-cut-sec", type=int, default=120)
    p.add_argument("--min-match-sec", type=int, default=180)
    p.add_argument("--max-match-sec", type=int, default=1800)
    a = p.parse_args()
    result = build_dataset(
        a.raw_root, a.output_dir, max_files=a.max_replays,
        snapshot_step_sec=a.step_sec, early_cut_sec=a.early_cut_sec,
        min_match_sec=a.min_match_sec, max_match_sec=a.max_match_sec,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
