from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class LegacySummary:
    source_zip: str
    member_path: str
    payload: dict[str, Any]


@dataclass
class LegacyPredictionArtifact:
    source_zip: str
    member_path: str
    dataframe: pd.DataFrame


@dataclass
class LegacyFreezeImport:
    inventory_rows: list[dict[str, Any]]
    summaries: list[LegacySummary]
    csv_tables: dict[str, pd.DataFrame]
    nested_zip_rows: list[dict[str, Any]]
    prediction_artifacts: list[LegacyPredictionArtifact]


SUMMARY_SUFFIXES = (
    '_summary.json',
    '_comparison_summary.json',
    '_calibration_summary.json',
)


def _safe_json_load(raw: bytes) -> dict[str, Any] | None:
    try:
        obj = json.loads(raw)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def inspect_zip_inventory(zip_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            rows.append({
                'zip_path': str(zip_path),
                'member_path': info.filename,
                'file_size': int(info.file_size),
                'compressed_size': int(info.compress_size),
                'is_dir': bool(info.is_dir()),
                'suffix': Path(info.filename).suffix.lower(),
            })
    return rows


def import_legacy_freeze(zip_path: Path) -> LegacyFreezeImport:
    inventory_rows = inspect_zip_inventory(zip_path)
    summaries: list[LegacySummary] = []
    csv_tables: dict[str, pd.DataFrame] = {}
    nested_zip_rows: list[dict[str, Any]] = []
    prediction_artifacts: list[LegacyPredictionArtifact] = []

    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            raw = zf.read(member)
            lower = member.lower()
            if lower.endswith('.json'):
                obj = _safe_json_load(raw)
                if obj is not None and ('summary' in lower or any(lower.endswith(sfx) for sfx in SUMMARY_SUFFIXES)):
                    summaries.append(LegacySummary(str(zip_path), member, obj))
            elif lower.endswith('.csv'):
                try:
                    csv_tables[member] = pd.read_csv(io.BytesIO(raw))
                except Exception:
                    pass
            elif lower.endswith('.zip'):
                try:
                    nested = zipfile.ZipFile(io.BytesIO(raw))
                except Exception:
                    continue
                for inner_info in nested.infolist():
                    nested_zip_rows.append({
                        'zip_path': str(zip_path),
                        'container_member': member,
                        'member_path': inner_info.filename,
                        'file_size': int(inner_info.file_size),
                        'compressed_size': int(inner_info.compress_size),
                        'suffix': Path(inner_info.filename).suffix.lower(),
                    })
                    inner_raw = nested.read(inner_info.filename)
                    inner_lower = inner_info.filename.lower()
                    if inner_lower.endswith('.json'):
                        obj = _safe_json_load(inner_raw)
                        if obj is not None and ('summary' in inner_lower or any(inner_lower.endswith(sfx) for sfx in SUMMARY_SUFFIXES)):
                            summaries.append(LegacySummary(f'{zip_path}::{member}', inner_info.filename, obj))
                    elif inner_lower.endswith('.csv'):
                        try:
                            df = pd.read_csv(io.BytesIO(inner_raw))
                        except Exception:
                            continue
                        csv_tables[f'{member}::{inner_info.filename}'] = df
                        cols = set(df.columns)
                        if {'replay_id', 'time_sec'} <= cols and any(c.startswith('y_prob') for c in cols):
                            prediction_artifacts.append(LegacyPredictionArtifact(f'{zip_path}::{member}', inner_info.filename, df))
    return LegacyFreezeImport(inventory_rows, summaries, csv_tables, nested_zip_rows, prediction_artifacts)


def canonicalize_legacy_summary(payload: dict[str, Any], source_name: str) -> dict[str, Any]:
    canonical = {
        'source_name': source_name,
        'model': payload.get('model'),
        'dataset_path': payload.get('dataset_path'),
        'rows': payload.get('rows'),
        'n_features': payload.get('n_features'),
        'n_replays_total': payload.get('n_replays_total'),
        'n_replays_train': payload.get('n_replays_train'),
        'n_replays_val': payload.get('n_replays_val'),
        'n_replays_test': payload.get('n_replays_test'),
        'cv_accuracy': payload.get('cv_accuracy'),
        'validation_logloss': payload.get('validation_logloss'),
        'test_accuracy': payload.get('test_accuracy'),
        'test_balanced_accuracy': payload.get('test_balanced_accuracy'),
        'test_auc': payload.get('test_auc'),
        'test_logloss': payload.get('test_logloss'),
        'best_params': payload.get('best_params'),
        'best_iteration': payload.get('best_iteration') or payload.get('best_epoch'),
        'device': payload.get('device'),
        'notes': payload.get('notes'),
    }
    return canonical


def prediction_schema_summary(df: pd.DataFrame) -> dict[str, Any]:
    prob_cols = [c for c in df.columns if c.startswith('y_prob')]
    pred_cols = [c for c in df.columns if c.startswith('y_pred')]
    return {
        'columns': list(df.columns),
        'rows': int(len(df)),
        'n_replays': int(df['replay_id'].astype(str).nunique()) if 'replay_id' in df.columns else None,
        'time_min': float(df['time_sec'].min()) if 'time_sec' in df.columns else None,
        'time_max': float(df['time_sec'].max()) if 'time_sec' in df.columns else None,
        'probability_columns': prob_cols,
        'prediction_columns': pred_cols,
        'target_present': 'y_true' in df.columns,
    }
