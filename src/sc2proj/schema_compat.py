from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import load_dataframe_from_zip


REQUIRED_DATASET_COLUMNS = ['replay_id', 'time_sec', 'p1_wins']
REQUIRED_PREDICTION_COLUMNS = ['replay_id', 'time_sec', 'y_true']
OPTIONAL_METADATA_COLUMNS = ['matchup', 'race_matchup', 'map_name', 'league']


@dataclass
class SchemaValidationResult:
    ok: bool
    summary: dict[str, Any]
    missing_columns: list[str]
    unexpected_object_columns: list[str]


def validate_dataframe_schema(df: pd.DataFrame, required_columns: list[str], object_columns_allowed: list[str] | None = None) -> SchemaValidationResult:
    object_columns_allowed = object_columns_allowed or ['replay_id', *OPTIONAL_METADATA_COLUMNS]
    missing = [c for c in required_columns if c not in df.columns]
    unexpected_object = [
        c for c in df.columns
        if df[c].dtype == 'object' and c not in object_columns_allowed
    ]
    summary = {
        'rows': int(len(df)),
        'columns': list(df.columns),
        'required_columns': required_columns,
        'missing_columns': missing,
        'unexpected_object_columns': unexpected_object,
    }
    return SchemaValidationResult(ok=(not missing and not unexpected_object), summary=summary, missing_columns=missing, unexpected_object_columns=unexpected_object)


def validate_dataset_zip_schema(dataset_zip: Path) -> SchemaValidationResult:
    df = load_dataframe_from_zip(dataset_zip)
    return validate_dataframe_schema(df, REQUIRED_DATASET_COLUMNS)


def compare_feature_sets(reference_columns: list[str], candidate_columns: list[str], ignore_columns: list[str] | None = None) -> dict[str, Any]:
    ignore = set(ignore_columns or [])
    ref = [c for c in reference_columns if c not in ignore]
    cand = [c for c in candidate_columns if c not in ignore]
    ref_set = set(ref)
    cand_set = set(cand)
    return {
        'reference_count': len(ref),
        'candidate_count': len(cand),
        'intersection_count': len(ref_set & cand_set),
        'reference_only': sorted(ref_set - cand_set),
        'candidate_only': sorted(cand_set - ref_set),
        'intersection': sorted(ref_set & cand_set),
    }
