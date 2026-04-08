from __future__ import annotations
import argparse, random, zipfile
from pathlib import Path
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-zip', type=Path, required=True)
    ap.add_argument('--output-csv', type=Path, required=True)
    ap.add_argument('--output-zip', type=Path, required=True)
    ap.add_argument('--max-replays', type=int, required=True)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    with zipfile.ZipFile(args.dataset_zip) as zf:
        name = [n for n in zf.namelist() if n.endswith('.csv')][0]
        with zf.open(name) as fh:
            df = pd.read_csv(fh)
    replays = sorted(df['replay_id'].astype(str).unique().tolist())
    selected = set(random.Random(args.seed).sample(replays, min(args.max_replays, len(replays))))
    subset = df[df['replay_id'].astype(str).isin(selected)].copy()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(args.output_csv, index=False)
    compression = {'method': 'zip', 'archive_name': args.output_csv.name}
    subset.to_csv(args.output_zip, index=False, compression=compression)
    print({'output_zip': str(args.output_zip), 'n_rows': int(len(subset)), 'n_replays': int(subset['replay_id'].astype(str).nunique())})


if __name__ == '__main__':
    main()
