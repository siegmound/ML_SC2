"""Replay-safe, read-only index of fixed-length temporal windows.

Offsets always refer to `ordered_rows`, NOT to the source CSV's original
order. An S2 data loader must apply `source_row` when gathering feature rows.
"""
from __future__ import annotations

import csv
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPLAY = "replay_id"
TIME = "time_sec"
TARGET = "p1_wins"
NON_FEATURES = frozenset((REPLAY, TIME, TARGET))
INDEX_COLUMNS = ("start_row", "stop_row", REPLAY, TIME, TARGET)


@dataclass(frozen=True)
class SequenceIndex:
    """The sorted row permutation and a table of [start, stop) windows."""

    ordered_rows: pd.DataFrame
    windows: pd.DataFrame
    feature_names: tuple[str, ...]
    window_length: int
    stride: int
    step_sec: int

    def stats(self) -> dict:
        return {
            "rows": int(len(self.ordered_rows)),
            "replays": int(self.ordered_rows[REPLAY].nunique()),
            "windows": int(len(self.windows)),
            "windowed_replays": int(self.windows[REPLAY].nunique()),
            "feature_count": len(self.feature_names),
            "window_length": self.window_length,
            "stride": self.stride,
            "step_sec": self.step_sec,
        }


def csv_schema_from_zip(dataset_zip: str | Path) -> tuple[str, ...]:
    """Read a ZIP's CSV header without decompressing its entire payload."""
    path = Path(dataset_zip)
    if not zipfile.is_zipfile(path):
        raise ValueError(f"Not a valid ZIP (possibly a Git LFS pointer): {path}")
    with zipfile.ZipFile(path) as archive:
        csv_names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"Expected exactly one CSV inside {path}; got {csv_names}")
        with archive.open(csv_names[0]) as stream:
            line = stream.readline().decode("utf-8-sig")
    columns = tuple(next(csv.reader([line])))
    if len(columns) != len(set(columns)):
        raise ValueError("Duplicate CSV column names")
    for required in NON_FEATURES:
        if required not in columns:
            raise ValueError(f"Missing required column {required}")
    return columns


def validate_schema(columns: tuple[str, ...], manifest: str | Path,
                    dataset_variant: str = "v3_1_fixed") -> tuple[str, ...]:
    """Fail on addition, omission, or reordering vs the frozen B3 variant schema."""
    variants = {"v3_1_fixed": "v3_1_predictors",
                "v3_2_combatfix": "v3_2_predictors"}
    if dataset_variant not in variants:
        raise ValueError(f"Unknown dataset variant: {dataset_variant}")
    spec = json.loads(Path(manifest).read_text(encoding="utf-8"))
    expected = tuple(spec[variants[dataset_variant]])
    actual = tuple(c for c in columns if c not in NON_FEATURES)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        raise ValueError(
            f"{dataset_variant} feature schema mismatch: missing={missing}, unexpected={unexpected}, "
            f"order_matches={actual == expected}"
        )
    return actual


def read_snapshot_metadata(dataset_zip: str | Path) -> pd.DataFrame:
    """Only read identifier, time and target; never load 64 feature columns."""
    csv_schema_from_zip(dataset_zip)
    with zipfile.ZipFile(dataset_zip) as archive:
        name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
        with archive.open(name) as stream:
            metadata = pd.read_csv(
                stream,
                usecols=[REPLAY, TIME, TARGET],
                dtype={REPLAY: "string", TIME: "int32", TARGET: "int8"},
            )
            # Preserve global CSV offsets when the CLI later selects a smoke subset.
            metadata["source_row"] = np.arange(len(metadata), dtype=np.int64)
            return metadata


def build_sequence_index(
    metadata: pd.DataFrame,
    feature_names: tuple[str, ...],
    window_length: int = 9,
    stride: int = 1,
    step_sec: int = 15,
) -> SequenceIndex:
    """Build windows within consecutive time runs, with no future observations.

    The source dataframe's physical row positions are recorded as `source_row`.
    `start_row` and `stop_row` are offsets into `ordered_rows` after sorting.
    Gaps larger than one step break a run and cannot be crossed by a window.
    """
    if window_length < 1 or stride < 1 or step_sec < 1:
        raise ValueError("window_length, stride and step_sec must be positive")
    for name in (REPLAY, TIME, TARGET):
        if name not in metadata:
            raise ValueError(f"Missing required column: {name}")
    if len(set(feature_names)) != len(feature_names) or NON_FEATURES.intersection(feature_names):
        raise ValueError("Feature list contains duplicates or reserved columns")
    if "source_row" in metadata.columns:
        data = metadata.loc[:, [REPLAY, TIME, TARGET, "source_row"]].copy()
        if (
            not pd.api.types.is_integer_dtype(data["source_row"])
            or data["source_row"].isna().any()
            or (data["source_row"] < 0).any()
            or data["source_row"].duplicated().any()
        ):
            raise ValueError("source_row must contain unique nonnegative integer offsets")
    else:
        data = metadata.loc[:, [REPLAY, TIME, TARGET]].copy()
        data["source_row"] = np.arange(len(data), dtype=np.int64)
    if data.isna().any().any():
        raise ValueError("Null replay ID, time, or target")
    if not data[REPLAY].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError("Replay IDs must be nonempty strings")
    if not pd.api.types.is_integer_dtype(data[TIME]):
        raise ValueError("time_sec must have integer dtype")
    if (data[TIME] < 0).any() or (data[TIME] % step_sec != 0).any():
        raise ValueError("Invalid time_sec: expected nonnegative step-aligned timestamps")
    if not data[TARGET].isin([0, 1]).all():
        raise ValueError("Target must be binary (0/1)")
    if data.duplicated([REPLAY, TIME]).any():
        raise ValueError("Duplicate (replay_id, time_sec) key")
    if data.groupby(REPLAY, sort=False, observed=True)[TARGET].nunique().gt(1).any():
        raise ValueError("Conflicting target labels within a replay")

    ordered = data.sort_values([REPLAY, TIME], kind="mergesort").reset_index(drop=True)
    starts_all: list[np.ndarray] = []
    stops_all: list[np.ndarray] = []
    # Replays are contiguous in the sorted table; individual gaps form runs.
    for _, group in ordered.groupby(REPLAY, sort=False, observed=True):
        positions = group.index.to_numpy(dtype=np.int64)
        times = group[TIME].to_numpy(dtype=np.int64)
        boundaries = np.r_[0, np.flatnonzero(np.diff(times) != step_sec) + 1, len(times)]
        for begin, end in zip(boundaries[:-1], boundaries[1:]):
            if end - begin < window_length:
                continue
            local = np.arange(begin, end - window_length + 1, stride, dtype=np.int64)
            starts_all.append(positions[local])
            stops_all.append(positions[local + window_length - 1] + 1)

    starts = np.concatenate(starts_all) if starts_all else np.empty(0, dtype=np.int64)
    stops = np.concatenate(stops_all) if stops_all else np.empty(0, dtype=np.int64)
    end_rows = ordered.iloc[stops - 1][[REPLAY, TIME, TARGET]].reset_index(drop=True)
    windows = pd.DataFrame({"start_row": starts, "stop_row": stops})
    windows[REPLAY] = end_rows[REPLAY].to_numpy()
    windows[TIME] = end_rows[TIME].to_numpy()
    windows[TARGET] = end_rows[TARGET].to_numpy()
    return SequenceIndex(ordered, windows, tuple(feature_names), window_length, stride, step_sec)
