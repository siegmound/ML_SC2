"""Independent raw-game-to-v4 dataset build with replay-level lineage.

Never reuses the v3.1/v3.2 frozen CSV as source. File-content SHA-256
defines replay identity, so byte duplicates across tournament packs cannot
cross train/test splits. A per-file status record is preserved, including
rejections. Output directory must not exist; no historical asset is edited.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import platform
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .v4 import V4ReplayBridge

COLUMNS = ("replay_id", "time_sec", "p1_wins")
FROZEN_NAMES = {
    "starcraft_full_dataset_v3_1_fixed.zip",
    "starcraft_full_dataset_v3_2_combatfix.zip",
}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def code_version_sha256() -> dict[str, str]:
    base = Path(__file__).resolve().parent
    return {
        name: file_sha256(base / name)
        for name in (
            "v4.py", "v4_dataset.py", "parser.py", "bridge.py",
            "constants.py", "counters.py",
        )
    }


def source_catalog(root: Path, max_files: int | None = None) -> tuple[list[dict], list[dict]]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Raw replay directory absent: {root}")
    files = sorted(
        p for p in root.rglob("*")
        if p.is_file() and not p.is_symlink()
        and p.suffix.lower() == ".sc2replay"
    )
    if not files:
        raise ValueError("No .SC2Replay files under specified raw root")
    catalog = []
    for path in files:
        with path.open("rb") as f:
            magic = f.read(4)
        digest = file_sha256(path)
        catalog.append({
            "source_relpath": path.relative_to(root).as_posix(),
            "original_filename": path.name,
            "sha256": digest,
            "bytes": path.stat().st_size,
            "valid_mpq_magic": magic in (b"MPQ\x1a", b"MPQ\x1b"),
            "appledouble": magic == b"\x00\x05\x16\x07",
            "_path": path,
        })
    # Deterministic sampling by *replay content*, not filesystem ordering.
    identities = sorted({r["sha256"] for r in catalog})
    selected = set(identities[:max_files] if max_files else identities)
    accepted = [x for x in catalog if x["sha256"] in selected]
    skipped = [x for x in catalog if x["sha256"] not in selected]
    if not accepted:
        raise ValueError("No source files selected")
    return accepted, skipped


def validate_replay_frame(df: pd.DataFrame, replay_id: str,
                          early_cut_sec: int, step_sec: int) -> str | None:
    if not set(COLUMNS).issubset(df.columns):
        return "missing_required_columns"
    if df.empty:
        return "no_snapshots"
    if df["replay_id"].astype(str).nunique() != 1 or (
        df["replay_id"].astype(str) != replay_id
    ).any():
        return "replay_id_inconsistent"
    if df["p1_wins"].isna().any() or not df["p1_wins"].isin([0, 1]).all():
        return "invalid_winner"
    if df["p1_wins"].nunique() != 1:
        return "contradictory_winner"
    times = pd.to_numeric(df["time_sec"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(times).all() or not (times > early_cut_sec).all():
        return "invalid_snapshot_times"
    if len(set(times)) != len(times) or not np.all(np.diff(times) > 0):
        return "duplicate_or_unordered_times"
    if not np.allclose(times / step_sec, np.round(times / step_sec), atol=1e-5):
        return "invalid_snapshot_step"
    features = [c for c in df.columns if c not in COLUMNS]
    if not features:
        return "no_features"
    try:
        values = df[features].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    except (ValueError, TypeError):
        return "nonnumeric_features"
    if not np.isfinite(values).all():
        return "nonfinite_features"
    return None


def build_dataset(raw_root: Path, out_dir: Path, *,
                  max_files: int | None = None,
                  snapshot_step_sec: int = 15,
                  early_cut_sec: int = 120,
                  min_match_sec: int = 180,
                  max_match_sec: int = 1800,
                  bridge=None) -> dict:
    out_dir = out_dir.resolve()
    raw_root = raw_root.resolve()
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing outputs: {out_dir}")
    if raw_root in out_dir.parents or out_dir == raw_root:
        raise ValueError("Output must not reside inside scanned raw replay directory")
    if out_dir.name in FROZEN_NAMES:
        raise ValueError("Cannot overwrite frozen dataset names")
    if max_files is not None and max_files <= 0:
        raise ValueError("max_files must be positive or omitted")
    if not (0 <= early_cut_sec < min_match_sec < max_match_sec):
        raise ValueError("Invalid temporal filtering parameters")
    if snapshot_step_sec <= 0:
        raise ValueError("Snapshot step must be positive")

    catalog, excluded_by_sampling = source_catalog(raw_root, max_files)
    processor = bridge if bridge is not None else V4ReplayBridge(
        snapshot_step_sec=snapshot_step_sec,
        early_cut_sec=early_cut_sec,
        min_match_sec=min_match_sec,
        max_match_sec=max_match_sec,
    )
    audit = []
    frames = []
    unique_seen: set[str] = set()
    for item in catalog:
        source = item["_path"]
        digest = item["sha256"]
        record = {k: v for k, v in item.items() if k != "_path"}
        record["replay_id"] = digest + ".SC2Replay"
        record["n_snapshots"] = 0
        if digest in unique_seen:
            record.update(status="excluded", reason="duplicate_content")
            audit.append(record)
            continue
        unique_seen.add(digest)
        if not item["valid_mpq_magic"]:
            reason = "appledouble_metadata" if item["appledouble"] else "invalid_mpq_magic"
            record.update(status="excluded", reason=reason)
            audit.append(record)
            continue
        try:
            parsed = processor.process_replay(source)
            record["bridge_status"] = parsed.status
            if parsed.status != "ok" or parsed.dataframe is None:
                record.update(status="excluded", reason=parsed.reason or "parser_failed")
            else:
                frame = parsed.dataframe.copy()
                frame["replay_id"] = record["replay_id"]
                issue = validate_replay_frame(
                    frame, record["replay_id"], early_cut_sec, snapshot_step_sec
                )
                if issue:
                    record.update(status="excluded", reason=issue)
                else:
                    record.update(
                        status="included", reason="",
                        n_snapshots=len(frame),
                    )
                    frames.append(frame)
        except Exception as error:
            record.update(
                status="excluded", reason="parser_exception:" + type(error).__name__
            )
        audit.append(record)

    for item in excluded_by_sampling:
        audit.append({
            **{k: v for k, v in item.items() if k != "_path"},
            "replay_id": item["sha256"] + ".SC2Replay",
            "status": "not_selected", "reason": "max_files_limit",
            "n_snapshots": 0,
        })
    if not frames:
        raise ValueError("No valid replay snapshots: inspect source and exclusions")
    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.sort_values(["replay_id", "time_sec"], kind="stable").reset_index(drop=True)
    if dataset.duplicated(["replay_id", "time_sec"]).any():
        raise ValueError("Duplicate replay/time keys after combining source files")
    if dataset.groupby("replay_id")["p1_wins"].nunique().max() != 1:
        raise ValueError("Contradictory replay winner labels")
    ordered_columns = list(COLUMNS) + sorted(c for c in dataset.columns if c not in COLUMNS)
    dataset = dataset[ordered_columns]
    content = dataset.to_csv(index=False).encode("utf-8")
    zipinfo = zipfile.ZipInfo("sc2_v4_clean.csv", date_time=(1980, 1, 1, 0, 0, 0))
    zipinfo.compress_type = zipfile.ZIP_DEFLATED

    # Outputs are created only once all source data have been validated.
    out_dir.mkdir(parents=True, exist_ok=False)
    dest = out_dir / "sc2_v4_clean.zip"
    with zipfile.ZipFile(dest, "w") as archive:
        archive.writestr(zipinfo, content, compress_type=zipfile.ZIP_DEFLATED,
                         compresslevel=6)
    audit_path = out_dir / "per_replay_source_audit.csv"
    audit_columns = sorted(set().union(*(row.keys() for row in audit)))
    with audit_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=audit_columns)
        writer.writeheader()
        writer.writerows(audit)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    status_counts = Counter(row["reason"] if row["reason"] else "included" for row in audit)
    report = {
        "schema": "ml_sc2.v4.raw_replay_to_dataset.v1",
        "dataset_version": "v4_clean_independent",
        "historical_frozen_data_used_for_build": False,
        "original_v31_pipeline_attested": False,
        "raw_directory": str(raw_root),
        "git_commit": commit,
        "python_version": sys.version,
        "platform": platform.platform(),
        "code_sha256": code_version_sha256(),
        "raw_file_paths_scanned": len(catalog) + len(excluded_by_sampling),
        "raw_unique_hashes_selected": len(unique_seen),
        "raw_duplicate_paths": status_counts["duplicate_content"],
        "excluded_by_max_files": status_counts["max_files_limit"],
        "included_replays": int(dataset["replay_id"].nunique()),
        "excluded_replays": sum(row["status"] == "excluded" for row in audit),
        "exclusion_reason_counts": dict(sorted(status_counts.items())),
        "n_snapshots": len(dataset),
        "n_features": len(ordered_columns) - len(COLUMNS),
        "feature_columns": ordered_columns[len(COLUMNS):],
        "min_max_time_sec": [float(dataset.time_sec.min()), float(dataset.time_sec.max())],
        "source_snapshot_step_sec": snapshot_step_sec,
        "source_early_cut_sec": early_cut_sec,
        "source_min_match_sec": min_match_sec,
        "source_max_match_sec": max_match_sec,
        "dataset_csv_sha256": hashlib.sha256(content).hexdigest(),
        "dataset_zip_sha256": file_sha256(dest),
        "per_replay_audit_sha256": file_sha256(audit_path),
        "warning": "New independent v4 data. Not parity/reproduction of frozen 23,241-replay data.",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report
