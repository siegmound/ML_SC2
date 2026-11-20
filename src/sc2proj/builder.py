from __future__ import annotations

import hashlib
import sys
from collections import Counter
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
        replay_paths = list(replay_paths)
        frozen_archive_names = {
            "starcraft_full_dataset_v3_1_fixed.zip",
            "starcraft_full_dataset_v3_2_combatfix.zip",
        }
        if output_zip.name in frozen_archive_names:
            raise ValueError("Refusing to regenerate/overwrite a frozen final dataset; choose a new dataset name")
        for path in (output_zip, manifest_path, audit_csv_path):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite existing dataset/audit output: {path}")
        # The bridge currently assigns replay_id from a basename, not the full path.
        # Replays from different packs can share a basename: reject before parsing
        # rather than merging unrelated match snapshots into one replay group.
        counts = Counter(Path(path).name for path in replay_paths)
        duplicate_ids = [name for name, count in counts.items() if count > 1]
        if duplicate_ids:
            raise ValueError(
                f"Ambiguous replay_id basenames across input replaypacks: {sorted(duplicate_ids)[:10]}"
            )

        frames = []
        audit_rows = []
        included_replays = []
        excluded_replays = []
        for replay_path in replay_paths:
            result = self.bridge.process_replay(replay_path)
            result.metadata["source_path"] = str(replay_path)
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
        if df.empty:
            raise RuntimeError("All replay snapshots were removed by quality filtering")
        if not set(df["p1_wins"].unique()).issubset({0, 1}):
            raise ValueError("Dataset contains unknown or nonbinary outcome labels")
        if df.duplicated(["replay_id", "time_sec"]).any():
            raise ValueError("Dataset contains duplicate replay/time snapshots")
        if (df.groupby("replay_id")["p1_wins"].nunique() != 1).any():
            raise ValueError("Dataset contains conflicting winners for a replay")

        retained = set(df["replay_id"].unique())
        vanished = set(included_replays) - retained
        for replay_id in sorted(vanished):
            excluded_replays.append({"replay_id": replay_id, "reason": "all_rows_removed_by_quality_filter"})
        included_replays = sorted(retained)
        for row in audit_rows:
            if row["replay_id"] in vanished:
                row["parse_status"] = "filtered"
                row["failure_reason"] = "all_rows_removed_by_quality_filter"
                row["num_snapshots"] = 0

        dump_dataframe_to_zip(df, output_zip, f"{self.config.dataset_name}.csv")
        audit_df = pd.DataFrame(audit_rows)
        ensure_dir(audit_csv_path.parent)
        audit_df.to_csv(audit_csv_path, index=False)
        manifest = {
            "dataset_name": self.config.dataset_name,
            "dataset_version": self.config.dataset_version,
            "parser_version": "clean_room_v1",
            "bridge_version": "clean_room_v1",
            "builder_version": "clean_room_b3_guarded",
            "lineage_status": "independent_clean_room_not_reproduction_of_frozen_v3_1_or_v3_2",
            "input_replay_count": int(len(replay_paths)),
            "excluded_replay_count": int(len(excluded_replays)),
            "exclusion_reason_counts": dict(sorted(Counter(x["reason"] for x in excluded_replays).items())),
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
