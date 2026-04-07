from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .bridge import ReplayBridge
from .utils import dump_dataframe_to_zip, ensure_dir, write_json


@dataclass
class BuildConfig:
    dataset_name: str
    dataset_version: str = "v1"
    snapshot_step_sec: int = 15
    early_cut_sec: int = 120
    min_match_sec: int = 180
    max_match_sec: int = 1800


class DatasetBuilder:
    def __init__(self, config: BuildConfig) -> None:
        self.config = config
        self.bridge = ReplayBridge(
            snapshot_step_sec=config.snapshot_step_sec,
            early_cut_sec=config.early_cut_sec,
            min_match_sec=config.min_match_sec,
            max_match_sec=config.max_match_sec,
        )

    def build_from_replays(self, replay_paths: Iterable[Path], output_zip: Path, manifest_path: Path, audit_csv_path: Path) -> dict:
        frames = []
        audit_rows = []
        included_replays = []
        excluded_replays = []
        for replay_path in replay_paths:
            result = self.bridge.process_replay(replay_path)
            audit_rows.append(result.metadata)
            if result.status == "ok" and result.dataframe is not None:
                frames.append(result.dataframe)
                included_replays.append(result.replay_id)
            else:
                excluded_replays.append({"replay_id": result.replay_id, "reason": result.reason})
        if not frames:
            raise RuntimeError("No replay produced usable snapshots.")
        df = pd.concat(frames, ignore_index=True)
        df.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
        df.dropna(inplace=True)
        dump_dataframe_to_zip(df, output_zip, f"{self.config.dataset_name}.csv")
        audit_df = pd.DataFrame(audit_rows)
        ensure_dir(audit_csv_path.parent)
        audit_df.to_csv(audit_csv_path, index=False)
        manifest = {
            "dataset_name": self.config.dataset_name,
            "dataset_version": self.config.dataset_version,
            "parser_version": "clean_room_v1",
            "bridge_version": "clean_room_v1",
            "builder_version": "clean_room_v1",
            "snapshot_step_sec": self.config.snapshot_step_sec,
            "early_cut_sec": self.config.early_cut_sec,
            "min_match_sec": self.config.min_match_sec,
            "max_match_sec": self.config.max_match_sec,
            "number_rows": int(len(df)),
            "number_replays": int(df["replay_id"].nunique()),
            "number_features": int(len(df.columns) - 3),
            "feature_list": [c for c in df.columns if c not in {"replay_id", "time_sec", "p1_wins"}],
            "target_name": "p1_wins",
            "group_name": "replay_id",
            "python_version": sys.version,
            "dataframe_sha256": hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest(),
            "included_replays": sorted(included_replays),
            "excluded_replays": excluded_replays,
        }
        write_json(manifest, manifest_path)
        return manifest
