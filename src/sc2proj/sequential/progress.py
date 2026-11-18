"""S8: retrospective game-progress checkpoint selection and validation scoring.

Progress percentages are used ONLY by the offline evaluator to choose
a replay's prediction time. The GRU itself receives only its preceding
9 snapshots; replay duration and fraction are never model inputs.
"""
from __future__ import annotations

from math import ceil

import numpy as np
import pandas as pd

from .comparison import model_metrics, paired_replay_bootstrap

CHECKPOINTS = (25, 50, 75, 90)
ARMS = ("v31_original64", "v32_common64", "v32_full73")
REQUIRED = ("replay_id", "time_sec", "p1_wins")


def select_exact_checkpoints(index, partitions, checkpoints: tuple[int, ...] = CHECKPOINTS):
    """Select endpoint at ceil(p% * N)-th snapshot, or exclude entire replay.

    N is the number of stored snapshots in the complete source replay, not
    the number of eligible 9-step windows. Never shift a requested checkpoint
    to the future or to a different progress percentage. For the primary
    comparison require a valid 9-step endpoint at EVERY requested fraction.
    """
    if not checkpoints or len(set(checkpoints)) != len(checkpoints) or any(
        not isinstance(p, int) or p <= 0 or p >= 100 for p in checkpoints
    ):
        raise ValueError("Progress checkpoints must be distinct integer percents in (0,100)")
    if int(index.manifest["window_length"]) != 9 or int(index.manifest["stride"]) != 1:
        raise ValueError("S8 requires the fixed nine-snapshot stride-one S1 index")
    rows = index.ordered_rows
    validation_ids = partitions.groups["val"]
    val_rows = rows.loc[rows["replay_id"].astype(str).isin(validation_ids), list(REQUIRED)]
    val_windows = index.windows.iloc[partitions.window_positions["val"]]
    window_keys = set(zip(val_windows["replay_id"].astype(str),
                          val_windows["time_sec"].to_numpy(dtype=np.int64)))
    if len(window_keys) != len(val_windows):
        raise ValueError("Duplicate S1 validation endpoints")
    selected = []
    eligible_count = {p: 0 for p in checkpoints}
    count_all = 0
    dropped = 0
    for replay, frame in val_rows.groupby("replay_id", sort=False, observed=True):
        count_all += 1
        replay = str(replay)
        n = len(frame)
        times = frame["time_sec"].to_numpy(dtype=np.int64)
        y = frame["p1_wins"].to_numpy(dtype=np.int8)
        if not np.all(y == y[0]):
            raise ValueError("Inconsistent winner labels within a replay")
        points = []
        for percent in checkpoints:
            observed_count = ceil(percent * n / 100)
            # The observed_count-th snapshot is the last visible one.
            t = int(times[observed_count - 1])
            if (replay, t) not in window_keys:
                continue
            eligible_count[percent] += 1
            points.append({
                "replay_id": replay, "time_sec": t, "p1_wins": int(y[0]),
                "progress_percent": percent, "observed_snapshot_count": observed_count,
                "replay_snapshot_count": n,
                "observed_fraction": float(observed_count / n),
            })
        if len(points) == len(checkpoints):
            selected.extend(points)
        else:
            dropped += 1
    result = pd.DataFrame(selected, columns=[
        *REQUIRED, "progress_percent", "observed_snapshot_count",
        "replay_snapshot_count", "observed_fraction",
    ])
    if result.empty:
        raise ValueError("No replay has all requested valid 9-step checkpoints")
    if len(result) != (count_all - dropped) * len(checkpoints):
        raise ValueError("Incomplete balanced replay-checkpoint population")
    if result.duplicated(["replay_id", "progress_percent"]).any() or result.duplicated(
        ["replay_id", "time_sec"]
    ).any():
        raise ValueError("Repeated checkpoint endpoints")
    if not result.groupby("replay_id")["progress_percent"].nunique().eq(len(checkpoints)).all():
        raise ValueError("Every selected replay needs all checkpoints")
    result = result.sort_values(["replay_id", "progress_percent"], kind="stable").reset_index(drop=True)
    coverage = {
        "validation_replays_total": count_all,
        "balanced_replays": count_all - dropped,
        "replays_excluded": dropped,
        "raw_eligible_replays_by_checkpoint": {str(p): eligible_count[p] for p in checkpoints},
        "checkpoints": list(checkpoints),
        "selection_rule": "exact_ceil_percent_of_full_replay_snapshot_count",
        "population_rule": "complete_case_same_replays_at_all_checkpoints",
        "window_length": 9,
    }
    return result, coverage


def join_checked_predictions(checkpoints: pd.DataFrame,
                             predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Strict left joins: no missing predictions, label mismatch or key drift."""
    if not predictions or not set(predictions).issubset(ARMS):
        raise ValueError("Unknown or empty model arm selection")
    merged = checkpoints.copy()
    for arm, pred in predictions.items():
        if pred.duplicated(["replay_id", "time_sec"]).any():
            raise ValueError(f"{arm}: repeated prediction endpoint")
        rhs = pred.loc[:, ["replay_id", "time_sec", "y_true", "y_prob", "y_pred"]].rename(
            columns={"y_true": f"{arm}_y_true", "y_prob": f"{arm}_prob",
                     "y_pred": f"{arm}_pred"})
        merged = merged.merge(rhs, how="left", on=["replay_id", "time_sec"],
                              validate="many_to_one", indicator=True)
        if not merged["_merge"].eq("both").all():
            raise ValueError(f"{arm}: missing selected checkpoint predictions")
        merged = merged.drop(columns="_merge")
        if not np.array_equal(merged[f"{arm}_y_true"].to_numpy(dtype=np.int8),
                              merged["p1_wins"].to_numpy(dtype=np.int8)):
            raise ValueError(f"{arm}: winner label mismatch at checkpoint")
        merged = merged.drop(columns=f"{arm}_y_true")
    return merged


def score_progress(selected: pd.DataFrame, arms: tuple[str, ...] = ARMS,
                   checkpoints: tuple[int, ...] = CHECKPOINTS,
                   *, reps: int = 1000, rng_seed: int = 2026):
    """One matched replay observation per checkpoint/arm; replay-paired CIs.

    A gain from 25% to 90% is an *endpoint-conditional* contrast, not a
    generalization guarantee. 25%-to-90% deltas keep the SAME replay IDs.
    """
    if reps < 0 or not arms or len(set(arms)) != len(arms):
        raise ValueError("Invalid bootstrap reps or model arms")
    if not set(arms).issubset(ARMS):
        raise ValueError("Unknown arm")
    metrics = {}
    for progress in checkpoints:
        stage = selected.loc[selected["progress_percent"] == progress].sort_values(
            ["replay_id"], kind="stable"
        ).reset_index(drop=True)
        if stage.empty:
            raise ValueError(f"Empty checkpoint {progress}")
        by_arm = {}
        y = stage["p1_wins"].to_numpy(dtype=np.int8)
        for arm in arms:
            by_arm[arm] = model_metrics(
                y, stage[f"{arm}_prob"].to_numpy(dtype=np.float64),
                stage[f"{arm}_pred"].to_numpy(dtype=np.int8))
        metrics[str(progress)] = {"replays": len(stage), "models": by_arm}
    gains = {}
    first, last = checkpoints[0], checkpoints[-1]
    early = selected.loc[selected["progress_percent"] == first].sort_values(
        "replay_id", kind="stable").reset_index(drop=True)
    late = selected.loc[selected["progress_percent"] == last].sort_values(
        "replay_id", kind="stable").reset_index(drop=True)
    if not np.array_equal(early.replay_id.to_numpy(), late.replay_id.to_numpy()):
        raise ValueError("Checkpoint progression uses different replay populations")
    if not np.array_equal(early.p1_wins.to_numpy(), late.p1_wins.to_numpy()):
        raise ValueError("Checkpoint progression has inconsistent winner labels")
    for i, arm in enumerate(arms):
        pair = late.loc[:, ["replay_id", "p1_wins"]].copy()
        pair["gru_prob"] = late[f"{arm}_prob"].to_numpy(dtype=np.float64)
        pair["gru_pred"] = late[f"{arm}_pred"].to_numpy(dtype=np.int8)
        pair["v31_prob"] = early[f"{arm}_prob"].to_numpy(dtype=np.float64)
        pair["v31_pred"] = early[f"{arm}_pred"].to_numpy(dtype=np.int8)
        ci = paired_replay_bootstrap(pair, "v31", reps=reps, seed=rng_seed + i)
        delta_acc = (metrics[str(last)]["models"][arm]["accuracy"] -
                     metrics[str(first)]["models"][arm]["accuracy"])
        delta_loss = (metrics[str(last)]["models"][arm]["log_loss"] -
                      metrics[str(first)]["models"][arm]["log_loss"])
        if not np.isclose(ci["accuracy_delta"], delta_acc, atol=1e-12, rtol=0) or not np.isclose(
            ci["log_loss_delta"], delta_loss, atol=1e-9, rtol=0
        ):
            raise ValueError("Progression point estimates do not match paired bootstrap")
        gains[arm] = {
            "from_percent": first, "to_percent": last,
            "accuracy_delta": float(delta_acc),
            "log_loss_delta": float(delta_loss),
            "paired_replay_bootstrap": ci,
        }
    return metrics, gains
