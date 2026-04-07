from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sc2proj.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, default='real_v3_1_fixed')
    parser.add_argument('--results-root', type=Path, default=PROJECT_ROOT / 'results')
    parser.add_argument('--output-path', type=Path, default=PROJECT_ROOT / 'results' / 'block7_status_report.json')
    args = parser.parse_args()

    summary = {
        'dataset_name': args.dataset_name,
        'registered_dataset_manifest_exists': any((PROJECT_ROOT / 'data' / 'processed' / 'manifests').glob(f'{args.dataset_name}*_manifest.json')),
        'dataset_quality_summary_exists': (PROJECT_ROOT / 'experiments' / 'dataset_quality' / f'{args.dataset_name}_dataset_quality_summary.json').exists(),
        'split_manifests': sorted([p.name for p in (PROJECT_ROOT / 'data' / 'processed' / 'splits').glob('split_seed_*.json')]),
        'logreg_runs': sorted([str(p.relative_to(PROJECT_ROOT)) for p in (args.results_root / 'logreg').glob(f'**/{args.dataset_name}/**/metrics_summary.json')]),
        'xgb_runs': sorted([str(p.relative_to(PROJECT_ROOT)) for p in (args.results_root / 'xgb').glob(f'**/{args.dataset_name}/**/metrics_summary.json')]),
        'legacy_calibration_import_exists': (args.results_root / 'real_calibration_import' / 'import_manifest.json').exists(),
    }
    write_json(summary, args.output_path)
    print(summary)


if __name__ == '__main__':
    main()
