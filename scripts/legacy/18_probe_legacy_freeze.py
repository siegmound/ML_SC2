from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sc2proj.legacy_artifacts import canonicalize_legacy_summary, import_legacy_freeze, prediction_schema_summary
from sc2proj.utils import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze-zip', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'experiments' / 'legacy_probe')
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    imported = import_legacy_freeze(args.freeze_zip)

    pd.DataFrame(imported.inventory_rows).to_csv(args.output_dir / 'freeze_inventory.csv', index=False)
    pd.DataFrame(imported.nested_zip_rows).to_csv(args.output_dir / 'nested_zip_inventory.csv', index=False)

    canonical_rows = [canonicalize_legacy_summary(s.payload, s.member_path) | {'source_zip': s.source_zip, 'member_path': s.member_path} for s in imported.summaries]
    pd.DataFrame(canonical_rows).to_csv(args.output_dir / 'legacy_summaries.csv', index=False)

    pred_rows = []
    for pred in imported.prediction_artifacts:
        pred_rows.append({'source_zip': pred.source_zip, 'member_path': pred.member_path, **prediction_schema_summary(pred.dataframe)})
    pd.DataFrame(pred_rows).to_csv(args.output_dir / 'legacy_prediction_schemas.csv', index=False)

    summary = {
        'freeze_zip': str(args.freeze_zip),
        'top_level_members': len(imported.inventory_rows),
        'nested_members': len(imported.nested_zip_rows),
        'summary_json_count': len(imported.summaries),
        'prediction_artifact_count': len(imported.prediction_artifacts),
        'csv_table_count': len(imported.csv_tables),
    }
    write_json(summary, args.output_dir / 'legacy_probe_summary.json')
    print(summary)


if __name__ == '__main__':
    main()
