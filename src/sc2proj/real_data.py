from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import ensure_dir, load_dataframe_from_zip, write_json

REQUIRED_MIN_COLUMNS = ["replay_id", "time_sec", "p1_wins"]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_zip_dataset(zip_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(zip_path, 'r') as zf:
        csv_members = [n for n in zf.namelist() if n.endswith('.csv')]
    if len(csv_members) != 1:
        raise ValueError(f'Expected exactly one CSV in {zip_path}, found {csv_members}')
    df = load_dataframe_from_zip(zip_path)
    missing = [c for c in REQUIRED_MIN_COLUMNS if c not in df.columns]
    dtypes = {c: str(t) for c, t in df.dtypes.items()}
    feature_columns = [c for c in df.columns if c not in REQUIRED_MIN_COLUMNS]
    return {
        'zip_path': str(zip_path),
        'zip_name': zip_path.name,
        'zip_sha256': sha256_file(zip_path),
        'csv_member': csv_members[0],
        'n_rows': int(len(df)),
        'n_replays': int(df['replay_id'].astype(str).nunique()) if 'replay_id' in df.columns else None,
        'n_columns': int(len(df.columns)),
        'columns': df.columns.tolist(),
        'feature_columns': feature_columns,
        'label_distribution': {
            'positive_rate': float(df['p1_wins'].mean()) if 'p1_wins' in df.columns else None,
            'positive_count': int(df['p1_wins'].sum()) if 'p1_wins' in df.columns else None,
        },
        'time_range_sec': {
            'min': float(df['time_sec'].min()) if 'time_sec' in df.columns else None,
            'max': float(df['time_sec'].max()) if 'time_sec' in df.columns else None,
        },
        'missing_required_columns': missing,
        'dtypes': dtypes,
    }


def build_real_dataset_manifest(zip_path: Path, dataset_name: str, dataset_version: str, notes: str = '') -> dict[str, Any]:
    info = inspect_zip_dataset(zip_path)
    manifest = {
        'dataset_name': dataset_name,
        'dataset_version': dataset_version,
        'dataset_kind': 'real_zipped_tabular_dataset',
        'source_zip': info['zip_name'],
        'source_zip_path': str(zip_path),
        'source_zip_sha256': info['zip_sha256'],
        'csv_member': info['csv_member'],
        'n_rows': info['n_rows'],
        'n_replays': info['n_replays'],
        'n_columns': info['n_columns'],
        'feature_columns': info['feature_columns'],
        'required_columns_present': len(info['missing_required_columns']) == 0,
        'missing_required_columns': info['missing_required_columns'],
        'positive_rate': info['label_distribution']['positive_rate'],
        'time_min_sec': info['time_range_sec']['min'],
        'time_max_sec': info['time_range_sec']['max'],
        'notes': notes,
    }
    return manifest


def write_manifest_pair(dataset_manifest: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    ensure_dir(output_dir)
    dataset_path = output_dir / f"{dataset_manifest['dataset_name']}_dataset_manifest.json"
    experiment_path = output_dir / f"{dataset_manifest['dataset_name']}_experiment_manifest.json"
    write_json(dataset_manifest, dataset_path)
    write_json({
        'experiment_name': f"dataset_registration_{dataset_manifest['dataset_name']}",
        'dataset_name': dataset_manifest['dataset_name'],
        'dataset_version': dataset_manifest['dataset_version'],
        'status': 'registered',
        'source_zip': dataset_manifest['source_zip'],
    }, experiment_path)
    return dataset_path, experiment_path
