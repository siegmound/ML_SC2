"""S2: replay-disjoint partitions and memory-efficient temporal PyTorch datasets.

The S1 index refers to *sorted* rows while its source_row refers to original
CSV offsets. This module explicitly applies that permutation before slicing
windows. Neither feature preprocessing nor window selection inspects test labels
for model selection; labels are only returned to the caller for evaluation.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

REPLAY, TIME, TARGET = "replay_id", "time_sec", "p1_wins"
PARTS = ("train", "val", "test")
RESERVED = frozenset((REPLAY, TIME, TARGET))


@dataclass(frozen=True)
class IndexedSequences:
    ordered_rows: pd.DataFrame
    windows: pd.DataFrame
    features: tuple[str, ...]
    manifest: dict


@dataclass(frozen=True)
class PartitionedSequences:
    groups: dict[str, frozenset[str]]
    row_masks: dict[str, np.ndarray]
    window_positions: dict[str, np.ndarray]
    seed: int

    def counts(self, index: IndexedSequences) -> dict:
        return {
            part: {
                "groups": len(self.groups[part]),
                "rows": int(self.row_masks[part].sum()),
                "windows": int(len(self.window_positions[part])),
                "windowed_groups": int(index.windows.iloc[self.window_positions[part]][REPLAY].nunique()),
            }
            for part in PARTS
        }


@dataclass(frozen=True)
class PreparedSequences:
    features_sorted: np.ndarray
    scaler: StandardScaler
    partitions: PartitionedSequences
    datasets: dict[str, "SequenceWindowDataset"]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for block in iter(lambda: fp.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_index(index_dir: str | Path) -> IndexedSequences:
    """Load S1 artifacts; fail closed on corrupt windows or changed metadata."""
    root = Path(index_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "sc2.sequential.window_index.v1":
        raise ValueError("Unsupported S1 index schema")
    features = tuple(manifest["features"])
    if not features or len(set(features)) != len(features) or RESERVED.intersection(features):
        raise ValueError("Invalid feature list in S1 manifest")
    ordered = pd.read_parquet(root / "ordered_rows.parquet")
    windows = pd.read_parquet(root / "window_index.parquet")
    required_rows = {REPLAY, TIME, TARGET, "source_row"}
    required_windows = {"start_row", "stop_row", REPLAY, TIME, TARGET}
    if not required_rows.issubset(ordered.columns) or not required_windows.issubset(windows.columns):
        raise ValueError("S1 index missing required columns")
    if len(ordered) != int(manifest["rows"]) or len(windows) != int(manifest["windows"]):
        raise ValueError("S1 Parquet lengths disagree with manifest")
    if int(manifest["feature_count"]) != len(features):
        raise ValueError("S1 feature count disagrees with manifest")
    if len(ordered) == 0:
        raise ValueError("S1 index contains no rows")
    source_row = ordered["source_row"].to_numpy()
    if not np.issubdtype(source_row.dtype, np.integer) or (source_row < 0).any() or len(np.unique(source_row)) != len(source_row):
        raise ValueError("S1 source_row must be unique nonnegative integer CSV offsets")
    times = ordered[TIME].to_numpy()
    targets = ordered[TARGET].to_numpy()
    replays = ordered[REPLAY].astype(str).to_numpy()
    if pd.isna(ordered[[REPLAY, TIME, TARGET]]).any().any() or not np.isin(targets, [0, 1]).all():
        raise ValueError("Missing metadata or invalid winner labels")
    if len(windows):
        starts = windows["start_row"].to_numpy()
        stops = windows["stop_row"].to_numpy()
        if not np.issubdtype(starts.dtype, np.integer) or not np.issubdtype(stops.dtype, np.integer):
            raise ValueError("Window offsets must be integers")
        length = int(manifest["window_length"])
        step = int(manifest["step_sec"])
        if length < 1 or step < 1 or (starts < 0).any() or (stops > len(ordered)).any() or not np.all(stops - starts == length):
            raise ValueError("S1 window bounds or lengths are invalid")
        ends = stops - 1
        if not np.array_equal(replays[ends], windows[REPLAY].astype(str).to_numpy()) or not np.array_equal(times[ends], windows[TIME].to_numpy()) or not np.array_equal(targets[ends], windows[TARGET].to_numpy()):
            raise ValueError("S1 window endpoint metadata mismatch")
        # Ensure every row in a window belongs to one uninterrupted replay run.
        breaks = np.r_[True, (replays[1:] != replays[:-1]) | (times[1:] - times[:-1] != step)]
        latest_run_start = np.maximum.accumulate(np.where(breaks, np.arange(len(ordered)), 0))
        if (latest_run_start[ends] > starts).any():
            raise ValueError("S1 window crosses a replay boundary or timestamp gap")
    return IndexedSequences(ordered, windows, features, manifest)


def load_replay_partitions(split_json: str | Path, index: IndexedSequences) -> PartitionedSequences:
    """Reuse historical replay-group lists; strict coverage for full S1 indexes."""
    spec = json.loads(Path(split_json).read_text(encoding="utf-8"))
    groups: dict[str, frozenset[str]] = {}
    for part in PARTS:
        values = spec.get(f"{part}_groups")
        if not isinstance(values, list) or not values or any(not isinstance(x, str) or not x for x in values):
            raise ValueError(f"Invalid or empty {part}_groups")
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate replay IDs in {part}_groups")
        groups[part] = frozenset(values)
        reported = spec.get(f"n_{part}_groups")
        if reported is not None and reported != len(values):
            raise ValueError(f"Incorrect n_{part}_groups metadata")
    if any(groups[a].intersection(groups[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Replay leakage: overlapping group assignments")
    all_ids = frozenset().union(*groups.values())
    index_ids = frozenset(index.ordered_rows[REPLAY].astype(str).unique())
    if not index_ids.issubset(all_ids):
        raise ValueError("S1 index contains replay IDs missing from split manifest")
    if index.manifest.get("subset_max_replays") is None and index_ids != all_ids:
        raise ValueError("Full S1 index does not exactly cover split manifest replay IDs")
    row_ids = index.ordered_rows[REPLAY].astype(str)
    window_ids = index.windows[REPLAY].astype(str)
    row_masks = {part: row_ids.isin(groups[part]).to_numpy(dtype=bool) for part in PARTS}
    window_positions = {
        part: np.flatnonzero(window_ids.isin(groups[part]).to_numpy(dtype=bool))
        for part in PARTS
    }
    if sum(int(x.sum()) for x in row_masks.values()) != len(index.ordered_rows) or sum(len(x) for x in window_positions.values()) != len(index.windows):
        raise ValueError("Unassigned or multiply assigned rows/windows")
    if not row_masks["train"].any() or len(window_positions["train"]) == 0:
        raise ValueError("No train observations/windows")
    return PartitionedSequences(groups, row_masks, window_positions, int(spec["seed"]))


def load_features_in_sorted_order(dataset_zip: str | Path, index: IndexedSequences, *, verify_sha: bool = True) -> np.ndarray:
    """Load a single float32 feature matrix. Permute physical CSV rows to S1 order."""
    archive_path = Path(dataset_zip)
    if verify_sha and sha256_file(archive_path) != index.manifest["source_zip_sha256"]:
        raise ValueError("Dataset ZIP SHA-256 differs from S1 index source")
    if not zipfile.is_zipfile(archive_path):
        raise ValueError("Dataset must be a real ZIP, not a Git LFS pointer")
    with zipfile.ZipFile(archive_path) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError("Expected exactly one CSV in dataset ZIP")
        with archive.open(names[0]) as stream:
            frame = pd.read_csv(stream, usecols=list(index.features), dtype={f: np.float32 for f in index.features})
    if set(frame.columns) != set(index.features):
        raise ValueError("ZIP feature columns do not match index manifest")
    source_rows = index.ordered_rows["source_row"].to_numpy(dtype=np.int64)
    if len(source_rows) and source_rows.max() >= len(frame):
        raise ValueError("source_row references nonexistent CSV row")
    source_matrix = frame.loc[:, list(index.features)].to_numpy(dtype=np.float32, copy=False)
    matrix = np.ascontiguousarray(source_matrix[source_rows], dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite source features (NaN/Inf)")
    return matrix


class SequenceWindowDataset(Dataset):
    """Slices a shared, normalized [snapshot,feature] array on demand."""
    def __init__(self, normalized_sorted: np.ndarray, windows: pd.DataFrame, window_positions: np.ndarray):
        if normalized_sorted.ndim != 2 or normalized_sorted.dtype != np.float32 or not normalized_sorted.flags.c_contiguous:
            raise ValueError("Expected contiguous float32 [rows,features] array")
        self.values = normalized_sorted
        selected = windows.iloc[window_positions]
        self.starts = selected["start_row"].to_numpy(dtype=np.int64, copy=True)
        self.stops = selected["stop_row"].to_numpy(dtype=np.int64, copy=True)
        self.targets = selected[TARGET].to_numpy(dtype=np.float32, copy=True)
        self.replay_ids = selected[REPLAY].astype(str).to_numpy()
        self.end_times = selected[TIME].to_numpy(dtype=np.int32, copy=True)
        if len(self.starts) and (self.starts.min() < 0 or self.stops.max() > len(self.values)):
            raise ValueError("Window bounds out of normalized feature matrix")

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = self.values[self.starts[i]:self.stops[i]]
        return torch.from_numpy(sequence), torch.tensor(self.targets[i], dtype=torch.float32)


def prepare_sequential_data(index: IndexedSequences, split: PartitionedSequences, dataset_zip: str | Path, *, verify_sha: bool = True) -> PreparedSequences:
    """Fit StandardScaler only on distinct TRAIN replay snapshots."""
    ordered = load_features_in_sorted_order(dataset_zip, index, verify_sha=verify_sha)
    scaler = StandardScaler()
    scaler.fit(ordered[split.row_masks["train"]])
    normalized = np.ascontiguousarray(scaler.transform(ordered), dtype=np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("Scaler produced non-finite feature values")
    datasets = {
        part: SequenceWindowDataset(normalized, index.windows, split.window_positions[part])
        for part in PARTS
    }
    return PreparedSequences(normalized, scaler, split, datasets)


def make_loaders(prepared: PreparedSequences, batch_size: int = 512, num_workers: int = 0) -> Mapping[str, DataLoader]:
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    result = {}
    for part in PARTS:
        generator = torch.Generator().manual_seed(prepared.partitions.seed)
        result[part] = DataLoader(
            prepared.datasets[part], batch_size=batch_size,
            shuffle=(part == "train"), num_workers=num_workers,
            generator=generator, pin_memory=torch.cuda.is_available(),
        )
    return result
