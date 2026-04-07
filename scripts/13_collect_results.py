from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sc2proj.aggregation import aggregate_metrics, collect_summary_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-root', type=Path, default=PROJECT_ROOT / 'results')
    parser.add_argument('--experiments-root', type=Path, default=PROJECT_ROOT / 'experiments')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'results' / 'summaries')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = collect_summary_rows(args.results_root, args.experiments_root)
    df.to_csv(args.output_dir / 'master_results.csv', index=False)
    agg = aggregate_metrics(df)
    agg.to_csv(args.output_dir / 'aggregate_results.csv', index=False)
    (args.output_dir / 'collection_status.txt').write_text(f'Collected {len(df)} rows and {len(agg)} aggregate rows.\n', encoding='utf-8')


if __name__ == '__main__':
    main()
