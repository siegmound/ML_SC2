from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sc2proj.utils import ensure_dir, write_json


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--legacy-import-root', type=Path, default=PROJECT_ROOT / 'results' / 'legacy_imports')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'data' / 'processed' / 'manifests')
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    manifest = _read_json(args.legacy_import_root / 'legacy_import_manifest.json')

    dataset_manifest = {
        'dataset_name': 'legacy_v3_1_fixed_from_artifacts',
        'dataset_version': 'legacy_import_only',
        'source_type': 'artifact_only',
        'raw_dataset_available': False,
        'import_manifest': str((args.legacy_import_root / 'legacy_import_manifest.json').resolve()),
        'notes': 'Built from legacy result artifacts because the raw zipped dataset is not present in the current workspace.',
    }

    models = []
    for item in manifest['summaries']:
        payload = _read_json(args.legacy_import_root / item['canonical_path'])
        models.append({
            'model': payload.get('model'),
            'summary_path': item['canonical_path'],
            'dataset_path_reported': payload.get('dataset_path'),
            'rows': payload.get('rows'),
            'n_features': payload.get('n_features'),
            'n_replays_total': payload.get('n_replays_total'),
            'test_accuracy': payload.get('test_accuracy'),
            'test_auc': payload.get('test_auc'),
            'test_logloss': payload.get('test_logloss'),
        })
    experiment_manifest = {
        'experiment_name': 'legacy_freeze_import',
        'dataset_name': dataset_manifest['dataset_name'],
        'models': models,
        'prediction_artifacts': manifest['predictions'],
    }

    write_json(dataset_manifest, args.output_dir / 'legacy_v3_1_fixed_dataset_manifest.json')
    write_json(experiment_manifest, args.output_dir / 'legacy_v3_1_fixed_experiment_manifest.json')
    print({'dataset_manifest': 'legacy_v3_1_fixed_dataset_manifest.json', 'experiment_manifest': 'legacy_v3_1_fixed_experiment_manifest.json', 'models': len(models)})


if __name__ == '__main__':
    main()
