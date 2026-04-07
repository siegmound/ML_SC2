from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .utils import ensure_dir, write_json


@dataclass
class LoadedSplitData:
    feature_columns: list[str]
    X_train: pd.DataFrame
    y_train: pd.Series
    groups_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    groups_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    groups_test: pd.Series
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


META_COLS = {"p1_wins", "replay_id", "time_sec"}


def load_split_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def select_split_frames(df: pd.DataFrame, split_manifest: dict) -> LoadedSplitData:
    train_groups = set(map(str, split_manifest["train_groups"]))
    val_groups = set(map(str, split_manifest["val_groups"]))
    test_groups = set(map(str, split_manifest["test_groups"]))

    replay_ids = df["replay_id"].astype(str)
    train_df = df[replay_ids.isin(train_groups)].copy()
    val_df = df[replay_ids.isin(val_groups)].copy()
    test_df = df[replay_ids.isin(test_groups)].copy()

    overlap = (
        set(train_df["replay_id"].astype(str).unique()) & set(val_df["replay_id"].astype(str).unique())
    ) | (
        set(train_df["replay_id"].astype(str).unique()) & set(test_df["replay_id"].astype(str).unique())
    ) | (
        set(val_df["replay_id"].astype(str).unique()) & set(test_df["replay_id"].astype(str).unique())
    )
    if overlap:
        raise ValueError(f"Replay overlap detected when materializing split: {sorted(overlap)[:10]}")

    feature_columns = [c for c in df.columns if c not in META_COLS]

    return LoadedSplitData(
        feature_columns=feature_columns,
        X_train=train_df[feature_columns],
        y_train=train_df["p1_wins"].astype(int),
        groups_train=train_df["replay_id"].astype(str),
        X_val=val_df[feature_columns],
        y_val=val_df["p1_wins"].astype(int),
        groups_val=val_df["replay_id"].astype(str),
        X_test=test_df[feature_columns],
        y_test=test_df["p1_wins"].astype(int),
        groups_test=test_df["replay_id"].astype(str),
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )


def downsample_frame(df: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=seed).sort_index().copy()


def downsample_loaded_split(loaded: LoadedSplitData, seed: int, max_train_rows: int | None = None, max_val_rows: int | None = None, max_test_rows: int | None = None) -> LoadedSplitData:
    train_df = downsample_frame(loaded.train_df, max_train_rows, seed)
    val_df = downsample_frame(loaded.val_df, max_val_rows, seed + 1)
    test_df = downsample_frame(loaded.test_df, max_test_rows, seed + 2)
    cols = loaded.feature_columns
    return LoadedSplitData(
        feature_columns=cols,
        X_train=train_df[cols],
        y_train=train_df["p1_wins"].astype(int),
        groups_train=train_df["replay_id"].astype(str),
        X_val=val_df[cols],
        y_val=val_df["p1_wins"].astype(int),
        groups_val=val_df["replay_id"].astype(str),
        X_test=test_df[cols],
        y_test=test_df["p1_wins"].astype(int),
        groups_test=test_df["replay_id"].astype(str),
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )


class RunLogger:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def log(self, message: str) -> None:
        self.lines.append(message)
        print(message)

    def dump(self, path: Path) -> None:
        ensure_dir(path.parent)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
def write_predictions(
    path: Path,
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
) -> None:
    ensure_dir(path.parent)
    out = pd.DataFrame(
        {
            "replay_id": test_df["replay_id"].astype(str).values,
            "time_sec": test_df["time_sec"].values,
            "y_true": test_df["p1_wins"].astype(int).values,
            "y_pred": np.asarray(y_pred).astype(int),
        }
    )
    if y_prob is not None:
        out["y_prob"] = np.asarray(y_prob, dtype=float)
    out.to_csv(path, index=False)


def make_run_dir(base_output_dir: Path, model_name: str, dataset_name: str, seed: int) -> Path:
    run_dir = base_output_dir / model_name / dataset_name / f"seed_{seed}"
    ensure_dir(run_dir)
    return run_dir


def write_artifacts_manifest(run_dir: Path, artifact_paths: Iterable[Path]) -> None:
    manifest = {
        "artifacts": [str(p.relative_to(run_dir)) for p in artifact_paths],
    }
    write_json(manifest, run_dir / "artifacts_manifest.json")
