from __future__ import annotations

import argparse
import json
import random
import zipfile
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-zip', type=Path, required=True)
    ap.add_argument('--output-csv', type=Path, required=True)
    ap.add_argument('--output-zip', type=Path, required=True)
    ap.add_argument('--output-manifest', type=Path, required=True)
    ap.add_argument('--max-replays', type=int, default=3000)
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
    manifest = {
        'dataset_name': args.output_zip.stem,
        'source_zip': str(args.dataset_zip),
        'seed': args.seed,
        'requested_replays': args.max_replays,
        'n_rows': int(len(subset)),
        'n_replays': int(subset['replay_id'].astype(str).nunique()),
        'columns': subset.columns.tolist(),
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(manifest)


if __name__ == '__main__':
    main()
