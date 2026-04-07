from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class DatasetQualityResult:
    feature_profile: pd.DataFrame
    missingness_report: pd.DataFrame
    constant_features: pd.DataFrame
    correlation_matrix: pd.DataFrame
    summary: Dict[str, object]


def build_dataset_quality_report(df: pd.DataFrame, target_col: str = "p1_wins", group_col: str = "replay_id", time_col: str = "time_sec") -> DatasetQualityResult:
    feature_cols = [c for c in df.columns if c not in {target_col, group_col, time_col}]
    rows = []
    for col in feature_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        finite_mask = np.isfinite(series.to_numpy(dtype=float, na_value=np.nan))
        clean = series[finite_mask]
        zero_ratio = float((clean == 0).mean()) if len(clean) else np.nan
        std = float(clean.std()) if len(clean) else np.nan
        q1 = float(clean.quantile(0.25)) if len(clean) else np.nan
        q3 = float(clean.quantile(0.75)) if len(clean) else np.nan
        iqr = q3 - q1 if np.isfinite(q1) and np.isfinite(q3) else np.nan
        outlier_ratio = float((((clean < (q1 - 1.5 * iqr)) | (clean > (q3 + 1.5 * iqr))).mean())) if len(clean) and np.isfinite(iqr) else np.nan
        rows.append({
            "feature": col,
            "dtype": str(df[col].dtype),
            "nan_count": int(series.isna().sum()),
            "inf_count": int((~np.isfinite(series.to_numpy(dtype=float, na_value=np.nan))).sum() - int(series.isna().sum())),
            "zero_ratio": zero_ratio,
            "unique_count": int(df[col].nunique(dropna=True)),
            "mean": float(clean.mean()) if len(clean) else np.nan,
            "std": std,
            "min": float(clean.min()) if len(clean) else np.nan,
            "max": float(clean.max()) if len(clean) else np.nan,
            "outlier_ratio": outlier_ratio,
            "constant_flag": bool(df[col].nunique(dropna=False) <= 1),
            "near_constant_flag": bool(np.isfinite(std) and std < 1e-8),
        })
    feature_profile = pd.DataFrame(rows).sort_values("feature").reset_index(drop=True)
    missingness_report = feature_profile[["feature", "nan_count", "inf_count"]].copy()
    constant_features = feature_profile[feature_profile["constant_flag"] | feature_profile["near_constant_flag"]].copy()
    correlation_matrix = df[feature_cols].corr(numeric_only=True)
    summary = {
        "n_rows": int(len(df)),
        "n_replays": int(df[group_col].nunique()),
        "n_features": int(len(feature_cols)),
        "n_constant_features": int(len(constant_features)),
        "target_balance": df[target_col].value_counts(normalize=True, dropna=False).sort_index().to_dict(),
    }
    return DatasetQualityResult(feature_profile, missingness_report, constant_features, correlation_matrix, summary)
