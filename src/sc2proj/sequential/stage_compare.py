"""S11: strict same-checkpoint alignment of frozen tabular and recurrent predictions.

All inputs are saved validation predictions. Additional archived endpoints are
allowed, but there must be a 1:1 match for every requested S8 checkpoint.
No model is trained or calibrated here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CHECKPOINT_KEYS = ("replay_id", "time_sec")
S8_KEYS = (*CHECKPOINT_KEYS, "progress_percent", "p1_wins")


def align_validation_model(stage: pd.DataFrame, predictions: pd.DataFrame,
                           model_name: str, *, require_exact: bool = False) -> pd.DataFrame:
    """Return S8 checkpoints plus aligned model probabilities; fail closed."""
    needed = set(CHECKPOINT_KEYS) | {"y_true", "y_prob", "y_pred"}
    if not set(S8_KEYS).issubset(stage.columns) or not needed.issubset(predictions.columns):
        raise ValueError(f"{model_name}: checkpoint or prediction columns missing")
    if stage.empty or predictions.empty:
        raise ValueError(f"{model_name}: empty validation source")
    stage = stage.loc[:, list(S8_KEYS)].copy()
    pred = predictions.loc[:, [*CHECKPOINT_KEYS, "y_true", "y_prob", "y_pred"]].copy()
    stage["replay_id"] = stage["replay_id"].astype(str)
    pred["replay_id"] = pred["replay_id"].astype(str)
    if (stage.isna().any().any() or pred.isna().any().any() or
            stage.duplicated(CHECKPOINT_KEYS).any() or
            stage.duplicated(["replay_id", "progress_percent"]).any() or
            pred.duplicated(CHECKPOINT_KEYS).any()):
        raise ValueError(f"{model_name}: missing fields or duplicate checkpoint keys")
    if not np.isin(stage.p1_wins, [0, 1]).all() or not np.isin(
        pred.y_true, [0, 1]
    ).all() or not np.isin(pred.y_pred, [0, 1]).all():
        raise ValueError(f"{model_name}: nonbinary labels/predictions")
    p = pred.y_prob.to_numpy(dtype=np.float64)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError(f"{model_name}: invalid probabilities")
    if require_exact and len(pred) != len(stage):
        raise ValueError(f"{model_name}: expected only selected S8 checkpoints")
    joined = stage.merge(pred, on=list(CHECKPOINT_KEYS), how="left",
                         validate="one_to_one", indicator=True)
    if not joined._merge.eq("both").all():
        raise ValueError(f"{model_name}: missing frozen S8 endpoint prediction")
    joined = joined.drop(columns="_merge")
    if not np.array_equal(joined.p1_wins.to_numpy(dtype=np.int8),
                          joined.y_true.to_numpy(dtype=np.int8)):
        raise ValueError(f"{model_name}: winner label mismatch")
    if not np.array_equal(
        (joined.y_prob.to_numpy(dtype=np.float64) >= .5).astype(np.int8),
        joined.y_pred.to_numpy(dtype=np.int8),
    ):
        raise ValueError(f"{model_name}: y_pred differs from 0.5 decision threshold")
    return joined.sort_values(
        ["replay_id", "progress_percent"], kind="stable"
    ).reset_index(drop=True)


def validate_stage_cohort(stage: pd.DataFrame, *, checkpoints=(25, 50, 75, 90)):
    if stage.empty or not set(S8_KEYS).issubset(stage.columns):
        raise ValueError("Invalid S8 checkpoint cohort")
    if set(stage.progress_percent) != set(checkpoints) or stage.duplicated(
        ["replay_id", "progress_percent"]
    ).any() or stage.duplicated(list(CHECKPOINT_KEYS)).any():
        raise ValueError("S8 cohort has wrong checkpoints or repeated keys")
    counts = stage.groupby("replay_id")["progress_percent"].nunique()
    if not counts.eq(len(checkpoints)).all() or not stage.groupby(
        "replay_id"
    )["p1_wins"].nunique().eq(1).all():
        raise ValueError("S8 cohort must contain exactly the same replay IDs at every stage")
    for p in checkpoints:
        if int((stage.progress_percent == p).sum()) != stage.replay_id.nunique():
            raise ValueError("Incomplete progress group")
    return int(stage.replay_id.nunique())
