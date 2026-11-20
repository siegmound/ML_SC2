"""Check schema lineage and optional frozen ZIP headers without rebuilding data.

B3's manifest records column provenance, NOT the missing transformation formulas.
The v3.2 feature list is sourced from saved XGB model configs, not a replay parser.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path('data/processed/manifests/b3_feature_lineage_v1.json')
DROP = {'replay_id', 'time_sec', 'p1_wins'}
RUNS = {
    'v3_1_fixed': 'results/xgb_full/real_v3_1_fixed_fullgpu_clean',
    'v3_2_combatfix': 'results/xgb_full/real_v3_2_combatfix_fullgpu_clean',
}


def parser_fields(source: Path) -> list[str]:
    tree = ast.parse(source.read_text(encoding='utf-8'))
    matches = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == 'snapshot'
            for target in node.targets
        ) and isinstance(node.value, ast.Dict):
            keys = [key.value for key in node.value.keys if isinstance(key, ast.Constant)]
            matches.append(keys)
    if len(matches) != 1:
        raise ValueError(f'Expected one snapshot dictionary, got {len(matches)}')
    return [key for key in matches[0] if key not in DROP]


def run_features(root: Path, dataset: str) -> list[str]:
    features = None
    for seed in (42, 43, 44):
        path = root / RUNS[dataset] / f'seed_{seed}/config_used.json'
        config = json.loads(path.read_text(encoding='utf-8'))
        if config.get('profile') != 'full':
            raise ValueError(f'Expected full-profile XGB features: {path}')
        names = config['features']
        if len(names) != len(set(names)) or DROP.intersection(names):
            raise ValueError(f'Invalid model feature schema: {path}')
        if features is None:
            features = names
        elif features != names:
            raise ValueError(f'Feature schema differs across seeds in {dataset}')
    return features


def derive(root: Path) -> dict:
    snapshot = parser_fields(root / 'src/sc2proj/parser.py')
    v31 = run_features(root, 'v3_1_fixed')
    v32 = run_features(root, 'v3_2_combatfix')
    a, b, c = set(snapshot), set(v31), set(v32)
    if not b.issubset(c):
        raise ValueError('v3.2 is not a strict feature superset of v3.1')
    return {
        'schema_version': 'b3_feature_lineage_v1',
        'basis': 'snapshot parser AST + saved full-profile XGB configs (three seeds each)',
        'clean_room_predictors': snapshot,
        'v3_1_predictors': v31,
        'v3_2_predictors': v32,
        'counts': {
            'clean_room': len(snapshot), 'v3_1': len(v31), 'v3_2': len(v32),
            'clean_room_in_v3_1': len(a & b),
            'missing_v3_1_formulas': len(b - a),
            'clean_room_only': len(a - b),
            'added_in_v3_2': len(c - b),
        },
        'frozen_v3_1_without_clean_room_builder': sorted(b - a),
        'clean_room_fields_not_in_v3_1': sorted(a - b),
        'v3_2_added_features': sorted(c - b),
        'reproduction_status': 'NOT_REPRODUCIBLE_FROM_CURRENT_CLEAN_ROOM_PARSER',
        'warning': 'Name overlap does not prove numerical equivalence. Derivation formulas and per-replay raw exclusions are not recovered by a schema comparison.',
    }


def verify_zip_header(path: Path, names: list[str]) -> None:
    with zipfile.ZipFile(path) as archive:
        csv_names = [name for name in archive.namelist() if name.endswith('.csv')]
        if len(csv_names) != 1:
            raise ValueError(f'Expected one dataset CSV inside {path}')
        with archive.open(csv_names[0]) as stream:
            header = next(csv.reader([stream.readline().decode('utf-8-sig')]))
    got = [name for name in header if name not in DROP]
    if got != names:
        raise ValueError(f'{path}: archived CSV header differs from model feature order')
    print(f'PASS ZIP header: {path} ({len(got)} predictors)')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=ROOT)
    parser.add_argument('--zip-v3-1', type=Path, help='Optional actual frozen v3.1 ZIP')
    parser.add_argument('--zip-v3-2', type=Path, help='Optional actual frozen v3.2 ZIP')
    args = parser.parse_args()
    actual = derive(args.repo_root)
    committed = json.loads((args.repo_root / MANIFEST).read_text(encoding='utf-8'))
    if actual != committed:
        print('FAIL: feature lineage manifest no longer matches parser/configs')
        return 1
    for dataset, archive in [('v3_1', args.zip_v3_1), ('v3_2', args.zip_v3_2)]:
        if archive:
            verify_zip_header(archive, actual[f'{dataset}_predictors'])
    print('PASS: parser/config feature lineage; schemas are NOT full replay reconstruction')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
