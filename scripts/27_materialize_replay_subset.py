from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sc2proj.utils import dump_dataframe_to_zip, load_dataframe_from_zip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-zip', type=Path, required=True)
    parser.add_argument('--output-zip', type=Path, required=True)
    parser.add_argument('--max-replays', type=int, default=1500)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    df = load_dataframe_from_zip(args.dataset_zip)
    replays = sorted(df['replay_id'].astype(str).unique().tolist())
    rng = random.Random(args.seed)
    selected = sorted(rng.sample(replays, min(args.max_replays, len(replays))))
    subset = df[df['replay_id'].astype(str).isin(selected)].copy()
    inner_name = args.output_zip.stem + '.csv'
    dump_dataframe_to_zip(subset, args.output_zip, inner_name)
    print({'output_zip': str(args.output_zip), 'n_rows': int(len(subset)), 'n_replays': int(subset['replay_id'].nunique())})


if __name__ == '__main__':
    main()
