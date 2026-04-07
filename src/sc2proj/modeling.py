from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler


@dataclass
class SearchResult:
    best_estimator: object
    best_params: dict
    best_score: float
    scoring: str
    cv_results: list[dict]


def make_numeric_preprocessor(kind: str = "standard") -> Pipeline:
    scaler = StandardScaler() if kind == "standard" else QuantileTransformer(output_distribution="normal", random_state=42)
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler),
        ]
    )


def threshold_predictions(y_prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (np.asarray(y_prob) >= threshold).astype(int)


def score_probabilities(scoring: str, y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> float:
    if scoring == "accuracy":
        return float((y_true == y_pred).mean())
    if scoring == "roc_auc":
        return float(roc_auc_score(y_true, y_prob))
    if scoring == "neg_log_loss":
        return float(-log_loss(y_true, y_prob, labels=[0, 1]))
    raise ValueError(f"Unsupported scoring: {scoring}")


def run_group_cv_search(
    estimator,
    param_grid: Iterable[dict],
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    scoring: str,
    n_splits: int = 3,
):
    cv = GroupKFold(n_splits=n_splits)
    cv_rows: list[dict] = []
    best_score: Optional[float] = None
    best_params: Optional[dict] = None
    best_estimator = None

    for params in param_grid:
        fold_scores: list[float] = []
        for fold_idx, (fit_idx, eval_idx) in enumerate(cv.split(X, y, groups=groups), start=1):
            model = clone(estimator)
            model.set_params(**params)
            model.fit(X.iloc[fit_idx], y.iloc[fit_idx])
            if not hasattr(model, "predict_proba"):
                raise ValueError("Estimator must expose predict_proba for canonical evaluation.")
            y_prob = model.predict_proba(X.iloc[eval_idx])[:, 1]
            y_pred = threshold_predictions(y_prob)
            score = score_probabilities(scoring, y.iloc[eval_idx].to_numpy(), y_prob, y_pred)
            fold_scores.append(score)
            cv_rows.append({
                "params": params,
                "fold": fold_idx,
                "score": score,
                "scoring": scoring,
            })
        mean_score = float(np.mean(fold_scores))
        if best_score is None or mean_score > best_score:
            best_score = mean_score
            best_params = dict(params)
            best_estimator = clone(estimator).set_params(**params)

    assert best_estimator is not None and best_params is not None and best_score is not None
    return SearchResult(
        best_estimator=best_estimator,
        best_params=best_params,
        best_score=best_score,
        scoring=scoring,
        cv_results=cv_rows,
    )
