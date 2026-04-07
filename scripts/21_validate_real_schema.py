from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sc2proj.schema_compat import REQUIRED_PREDICTION_COLUMNS, validate_dataframe_schema, validate_dataset_zip_schema
from sc2proj.utils import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-zip', type=Path, default=None)
    parser.add_argument('--legacy-import-root', type=Path, default=PROJECT_ROOT / 'results' / 'legacy_imports')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'experiments' / 'schema_validation')
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    report = {
        'dataset_validation': None,
        'prediction_validations': [],
    }

    if args.dataset_zip is not None:
        dataset_result = validate_dataset_zip_schema(args.dataset_zip)
        report['dataset_validation'] = dataset_result.summary | {'ok': dataset_result.ok}

    pred_root = args.legacy_import_root / 'predictions'
    for csv_path in sorted(pred_root.glob('*.csv')):
        df = pd.read_csv(csv_path)
        result = validate_dataframe_schema(df, REQUIRED_PREDICTION_COLUMNS)
        row = {'path': str(csv_path.relative_to(args.legacy_import_root)), 'ok': result.ok, **result.summary}
        report['prediction_validations'].append(row)

    write_json(report, args.output_dir / 'schema_validation_report.json')
    pd.DataFrame(report['prediction_validations']).to_csv(args.output_dir / 'prediction_schema_validation.csv', index=False)
    print({
        'dataset_checked': args.dataset_zip is not None,
        'prediction_files_checked': len(report['prediction_validations']),
        'prediction_failures': sum(0 if row['ok'] else 1 for row in report['prediction_validations']),
    })


if __name__ == '__main__':
    main()
