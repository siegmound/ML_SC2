from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sc2proj.legacy_artifacts import canonicalize_legacy_summary, import_legacy_freeze, prediction_schema_summary
from sc2proj.utils import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze-zip', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=PROJECT_ROOT / 'results' / 'legacy_imports')
    args = parser.parse_args()

    imported = import_legacy_freeze(args.freeze_zip)
    ensure_dir(args.output_root)

    summary_root = args.output_root / 'summaries'
    pred_root = args.output_root / 'predictions'
    table_root = args.output_root / 'tables'
    ensure_dir(summary_root)
    ensure_dir(pred_root)
    ensure_dir(table_root)

    summary_index = []
    for idx, summary in enumerate(imported.summaries, start=1):
        safe_name = Path(summary.member_path).name.replace('.json', '')
        payload = canonicalize_legacy_summary(summary.payload, summary.member_path)
        payload['legacy_source_zip'] = summary.source_zip
        payload['legacy_member_path'] = summary.member_path
        out_path = summary_root / f'{idx:02d}_{safe_name}.json'
        write_json(payload, out_path)
        summary_index.append({'canonical_path': str(out_path.relative_to(args.output_root)), 'source_zip': summary.source_zip, 'member_path': summary.member_path, 'model': payload.get('model')})

    prediction_index = []
    for idx, pred in enumerate(imported.prediction_artifacts, start=1):
        safe_name = Path(pred.member_path).name
        out_path = pred_root / f'{idx:02d}_{safe_name}'
        pred.dataframe.to_csv(out_path, index=False)
        schema = prediction_schema_summary(pred.dataframe)
        write_json(schema, out_path.with_suffix('.schema.json'))
        prediction_index.append({'canonical_path': str(out_path.relative_to(args.output_root)), 'source_zip': pred.source_zip, 'member_path': pred.member_path, 'rows': schema['rows'], 'n_replays': schema['n_replays']})

    for key, df in imported.csv_tables.items():
        safe_name = key.replace('::', '__').replace('/', '__')
        df.to_csv(table_root / safe_name, index=False)

    manifest = {
        'freeze_zip': str(args.freeze_zip),
        'n_summary_files': len(summary_index),
        'n_prediction_files': len(prediction_index),
        'n_tables': len(imported.csv_tables),
        'summaries': summary_index,
        'predictions': prediction_index,
    }
    write_json(manifest, args.output_root / 'legacy_import_manifest.json')
    print(manifest)


if __name__ == '__main__':
    main()
