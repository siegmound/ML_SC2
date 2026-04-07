from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

METRIC_COLUMNS = ['accuracy', 'balanced_accuracy', 'roc_auc', 'log_loss']


def collect_summary_rows(*roots: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob('metrics_summary.json'):
            try:
                data = json.loads(path.read_text(encoding='utf-8'))
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            row = {'source_file': str(path)}
            for key, value in data.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    row[key] = value
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    group_cols = [c for c in ['model_name', 'dataset_name', 'cv_scoring', 'selection_metric', 'device'] if c in df.columns]
    agg_map = {}
    for col in METRIC_COLUMNS:
        if col in df.columns:
            agg_map[col] = ['mean', 'std', 'min', 'max', 'count']
    if not agg_map:
        return pd.DataFrame()
    agg = df.groupby(group_cols, dropna=False).agg(agg_map)
    agg.columns = ['_'.join([c for c in tup if c]) for tup in agg.columns.to_flat_index()]
    return agg.reset_index()
