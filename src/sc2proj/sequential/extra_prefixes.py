"""S12: add train-only historical cutoffs to S9 checkpoint-anchored GRU training.

Keep the SAME train replay cohort and unchanged four validation checkpoints as
S9. Training example expansion is the only intervention; both S12 GRU arms
receive exactly the same endpoint times and labels, but use different contexts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANCHOR_CHECKPOINTS = (25, 50, 75, 90)
FIELDS = (
    "replay_id", "time_sec", "p1_wins", "progress_percent",
    "observed_snapshot_count", "replay_snapshot_count", "observed_fraction",
)


def expand_train_checkpoint_examples(index, partitions, anchors: pd.DataFrame,
                                     *, extras_per_replay: int = 12):
    """Add up to K equally spaced valid TRAIN endpoints between original anchors.

    Uses original anchor replay IDs exclusively; does not include other train
    replay groups, let alone validation/test. Original 4 anchors are preserved.
    `progress_percent=-1` denotes an additional *training-only* sample,
    never a model input. Number of extra endpoints may vary with data gaps.
    """
    if not isinstance(extras_per_replay, int) or not 0 <= extras_per_replay <= 40:
        raise ValueError("extras_per_replay must be an integer in [0,40]")
    if anchors.empty or not set(FIELDS).issubset(anchors.columns):
        raise ValueError("Missing original S9 train anchors")
    if anchors[list(FIELDS)].isna().any().any():
        raise ValueError("Null training anchor metadata")
    if set(anchors.progress_percent) != set(ANCHOR_CHECKPOINTS):
        raise ValueError("Training anchors must be exactly the four S9 checkpoints")
    if anchors.duplicated(["replay_id", "time_sec"]).any() or anchors.duplicated(
        ["replay_id", "progress_percent"]
    ).any():
        raise ValueError("Duplicate S9 anchors")
    per_replay = anchors.groupby("replay_id", sort=False, observed=True)
    if not per_replay.size().eq(4).all() or not per_replay.p1_wins.nunique().eq(1).all():
        raise ValueError("Incomplete or contradictory S9 train replay anchors")
    if not set(anchors.replay_id.astype(str)).issubset(partitions.groups["train"]):
        raise ValueError("Attempted training expansion outside train replay groups")
    if not set(partitions.groups["train"]).isdisjoint(partitions.groups["val"]):
        raise ValueError("Train/validation replay leakage")
    if not set(partitions.groups["train"]).isdisjoint(partitions.groups["test"]):
        raise ValueError("Train/test replay leakage")
    window_positions = partitions.window_positions["train"]
    pool = index.windows.iloc[window_positions].loc[
        :, ["replay_id", "time_sec", "p1_wins", "stop_row"]
    ].copy()
    if pool.empty or pool.duplicated(["replay_id", "time_sec"]).any():
        raise ValueError("Empty/duplicated train window pool")
    pool["replay_id"] = pool["replay_id"].astype(str)
    anchor_replays = set(anchors.replay_id.astype(str))
    pool = pool[pool.replay_id.isin(anchor_replays)].copy()
    if not set(pool.replay_id).issubset(partitions.groups["train"]):
        raise ValueError("Train window pool contains held-out replay")
    starts = index.ordered_rows.reset_index().groupby(
        "replay_id", sort=False, observed=True
    )["index"].min()
    sizes = index.ordered_rows.groupby("replay_id", sort=False, observed=True).size()
    pool_by_replay = {
        str(replay): frame.sort_values(
            ["stop_row", "time_sec"], kind="stable"
        ).reset_index(drop=True)
        for replay, frame in pool.groupby("replay_id", sort=False, observed=True)
    }
    extras = []
    selected_counts = []
    for replay, original in per_replay:
        key = str(replay)
        candidates = pool_by_replay.get(key)
        if candidates is None:
            raise ValueError(f"Missing training replay windows: {key}")
        if candidates.empty:
            raise ValueError(f"Missing training replay windows: {key}")
        anchored = candidates.loc[candidates.time_sec.isin(original.time_sec)]
        if len(anchored) != 4:
            raise ValueError(f"S9 anchor endpoints not all present in S1 train windows: {key}")
        if (not np.array_equal(
            anchored.p1_wins.to_numpy(dtype=np.int8),
            np.full(4, int(original.p1_wins.iloc[0]), dtype=np.int8),
        ) or len(set(anchored.stop_row)) != 4):
            raise ValueError("S9 anchor winner or endpoint order mismatch")
        first, last = int(anchored.stop_row.min()), int(anchored.stop_row.max())
        between = candidates.loc[
            (candidates.stop_row >= first)
            & (candidates.stop_row <= last)
            & ~candidates.time_sec.isin(original.time_sec),
        ].reset_index(drop=True)
        k = min(extras_per_replay, len(between))
        selected_counts.append(k)
        if not k:
            continue
        offsets = np.rint(np.linspace(0, len(between) - 1, num=k)).astype(np.int64)
        if len(np.unique(offsets)) != k:
            raise RuntimeError("Training endpoint selection is not unique")
        chosen = between.iloc[offsets]
        rstart = int(starts[key])
        total = int(sizes[key])
        for row in chosen.itertuples(index=False):
            seen = int(row.stop_row) - rstart
            if not 9 <= seen <= total:
                raise ValueError("Selected prefix crosses a replay boundary")
            extras.append({
                "replay_id": key,
                "time_sec": int(row.time_sec),
                "p1_wins": int(row.p1_wins),
                "progress_percent": -1,
                "observed_snapshot_count": seen,
                "replay_snapshot_count": total,
                "observed_fraction": float(seen / total),
            })
    anchor_copy = anchors.loc[:, list(FIELDS)].copy()
    if extras:
        expanded = pd.concat(
            [anchor_copy, pd.DataFrame(extras, columns=FIELDS)], ignore_index=True,
        )
    else:
        expanded = anchor_copy.copy()
    expanded = expanded.sort_values(
        ["replay_id", "time_sec"], kind="stable"
    ).reset_index(drop=True)
    if (expanded.duplicated(["replay_id", "time_sec"]).any()
            or set(expanded.replay_id) != anchor_replays
            or len(expanded) != len(anchors) + len(extras)):
        raise ValueError("Expanded training cohort corrupted")
    for replay, group in expanded.groupby("replay_id", sort=False, observed=True):
        orig = per_replay.get_group(replay).loc[:, list(FIELDS)]
        current = group.loc[group.progress_percent != -1, list(FIELDS)]
        if (len(orig) != len(current)
                or not orig.sort_values("time_sec").reset_index(drop=True).equals(
                    current.sort_values("time_sec").reset_index(drop=True)
                )):
            raise ValueError("S12 expansion altered a historical S9 anchor")
    info = {
        "schema": "sc2.sequential.train_extra_checkpoints.v1",
        "training_replays": len(anchor_replays),
        "s9_anchor_examples": len(anchors),
        "extra_examples": len(extras),
        "expanded_train_examples": len(expanded),
        "extras_per_replay_requested": extras_per_replay,
        "extras_per_replay_min": min(selected_counts),
        "extras_per_replay_max": max(selected_counts),
        "sampling": "deterministic_even_grid_between_existing_25_and_90_percent_train_anchors",
        "observed_prefix_count": "exclusive_stop_row_minus_replay_start",
        "s9_original_train_anchors_retained": True,
        "heldout_groups_not_selected": True,
    }
    return expanded, info
