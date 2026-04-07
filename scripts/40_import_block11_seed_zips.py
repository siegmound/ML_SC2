from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sc2proj.utils import ensure_dir, write_json


def infer_model(zip_path: Path) -> str:
    name = zip_path.name.lower()
    if "xgb" in name:
        return "xgb"
    if "rf" in name:
        return "rf"
    raise ValueError(f"Cannot infer model from {zip_path}")


def infer_seed(zip_path: Path) -> int:
    stem = zip_path.stem.lower()
    for token in stem.replace('-', '_').split('_'):
        if token.isdigit():
            return int(token)
    raise ValueError(f"Cannot infer seed from {zip_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip-paths', nargs='+', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'results' / 'block11_imported')
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    imported = []
    metric_rows = []
    for zip_path in args.zip_paths:
        model = infer_model(zip_path)
        seed = infer_seed(zip_path)
        run_dir = args.output_dir / model / 'real_v3_1_fixed_smoke3000' / f'seed_{seed}'
        ensure_dir(run_dir)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(run_dir)
        nested = run_dir / f'seed_{seed}'
        copied = []
        if nested.exists():
            for p in nested.iterdir():
                target = run_dir / p.name
                if target.exists():
                    target.unlink()
                p.replace(target)
                copied.append(target.name)
            nested.rmdir()
        metrics_path = run_dir / 'metrics_summary.json'
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
            metric_rows.append({'model': model, 'seed': seed, **metrics})
        imported.append({'zip_path': str(zip_path), 'model': model, 'seed': seed, 'run_dir': str(run_dir), 'files': copied})
    if metric_rows:
        pd.DataFrame(metric_rows).sort_values(['model','seed']).to_csv(args.output_dir / 'imported_metrics.csv', index=False)
    write_json({'imports': imported}, args.output_dir / 'import_manifest.json')


if __name__ == '__main__':
    main()
