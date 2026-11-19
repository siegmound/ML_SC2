#!/usr/bin/env python3
"""B3: diagnose unmatched original SC2Replays without inventing historical exclusions.

Reads the nine SOURCE_PRESENT_FROZEN_ABSENT_REASON_UNKNOWN rows from a
previous inventory. Inspects only original replay file headers via sc2reader
load_level=0. Optionally runs the *current* corrected ReplayBridge, which is
NOT the historical v3_counter120 executable and cannot attest past reasons.
Never writes to source directories, the frozen data, or existing audit paths.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

CANDIDATE_STATUS = "SOURCE_PRESENT_FROZEN_ABSENT_REASON_UNKNOWN"
MIN_DURATION = 180
MAX_DURATION = 1800


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def candidate_ids(unresolved_csv: Path) -> list[str]:
    require(unresolved_csv.is_file(), f"Missing inventory: {unresolved_csv}")
    with unresolved_csv.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames is not None and
                {"replay_id", "classification"} <= set(reader.fieldnames),
                "Missing required inventory columns")
        names = [row["replay_id"] for row in reader
                 if row["classification"] == CANDIDATE_STATUS]
    require(bool(names), "No raw-present/frozen-absent replay IDs")
    require(len(names) == len(set(names)), "Duplicate unmatched replay IDs")
    require(all(n and Path(n).name == n for n in names),
            "Expected basename-only replay IDs")
    return sorted(names)


def original_replay_files(raw_root: Path, selected: list[str]) -> dict[str, Path]:
    require(raw_root.is_dir(), f"Missing raw root: {raw_root}")
    wanted = set(selected)
    matches: dict[str, list[Path]] = defaultdict(list)
    for path in raw_root.rglob("*"):
        if path.is_file() and not path.is_symlink() and path.name in wanted:
            matches[path.name].append(path)
    for name in selected:
        require(len(matches.get(name, [])) == 1,
                f"Expected exactly one real source for {name}; found {len(matches.get(name, []))}")
        require(matches[name][0].suffix.lower() == ".sc2replay",
                f"Not an SC2Replay file: {name}")
    return {name: matches[name][0] for name in selected}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def duration_seconds(replay) -> float | None:
    value = getattr(replay, "length", None)
    if value is None:
        return None
    if hasattr(value, "total_seconds"):
        return float(value.total_seconds())
    if hasattr(value, "seconds"):
        return float(value.seconds)
    return None


def duration_category(duration: float | None) -> str:
    if duration is None:
        return "UNKNOWN_DURATION"
    if duration < MIN_DURATION:
        return "BELOW_HISTORICAL_MIN_180"
    if duration > MAX_DURATION:
        return "ABOVE_HISTORICAL_MAX_1800"
    return "WITHIN_HISTORICAL_DURATION_BAND"


def probe_header(path: Path, load_replay) -> dict:
    """Level-0 header probe: not the original acceptance/exclusion decision."""
    result = {
        "header_load_status": "ERROR", "header_error_type": "",
        "duration_seconds": None, "duration_category": "UNKNOWN_DURATION",
    }
    try:
        replay = load_replay(str(path), load_level=0)
        duration = duration_seconds(replay)
        result.update({
            "header_load_status": "OK",
            "duration_seconds": duration,
            "duration_category": duration_category(duration),
        })
    except Exception as exc:
        result["header_error_type"] = type(exc).__name__
    return result


def diagnostic(raw_root: Path, unresolved_csv: Path,
               *, run_current_bridge=False) -> tuple[dict, list[dict]]:
    from sc2reader import load_replay

    selected = candidate_ids(unresolved_csv)
    paths = original_replay_files(raw_root, selected)
    bridge = None
    if run_current_bridge:
        from sc2proj.bridge import ReplayBridge
        bridge = ReplayBridge()
    rows: list[dict] = []
    for name in selected:
        path = paths[name]
        row = {
            "replay_id": name, "raw_relpath": path.relative_to(raw_root).as_posix(),
            "raw_size_bytes": path.stat().st_size, "raw_sha256": file_sha256(path),
            "source_membership": CANDIDATE_STATUS,
            **probe_header(path, load_replay),
            "current_bridge_status": "NOT_RUN",
            "current_bridge_reason": "",
            "current_bridge_snapshots": None,
        }
        if bridge is not None:
            try:
                parsed = bridge.process_replay(path)
                row.update({
                    "current_bridge_status": parsed.metadata.get("parse_status", parsed.status),
                    "current_bridge_reason": parsed.metadata.get("failure_reason", parsed.reason),
                    "current_bridge_snapshots": parsed.metadata.get("num_snapshots"),
                })
            except Exception as exc:
                row.update({
                    "current_bridge_status": "PROBE_EXCEPTION",
                    "current_bridge_reason": type(exc).__name__,
                    "current_bridge_snapshots": None,
                })
        rows.append(row)
    duration_counts = dict(sorted(Counter(r["duration_category"] for r in rows).items()))
    return {
        "schema": "ml_sc2.b3.unmatched_replay_header_diagnostics.v1",
        "n_raw_present_frozen_absent": len(selected),
        "historical_exclusion_reasons_verified": False,
        "historical_executable_parity_verified": False,
        "historical_duration_band_sec": [MIN_DURATION, MAX_DURATION],
        "duration_counts": duration_counts,
        "header_errors": sum(r["header_load_status"] == "ERROR" for r in rows),
        "current_bridge_was_run": run_current_bridge,
        "current_bridge_is_historical": False,
        "disposition": "PARTIAL_DIAGNOSTIC_NOT_HISTORICAL_EXCLUSION_AUDIT",
    }, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/replay_subsets"))
    parser.add_argument("--inventory-dir", type=Path,
                        default=Path("results/b3_source_inventory_subset_20261119"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/b3_unmatched_diagnostics_20261119"))
    parser.add_argument("--current-bridge", action="store_true",
                        help="Optional slower *current* parser run; NEVER original provenance")
    args = parser.parse_args()
    root = args.raw_root.resolve()
    inventory = args.inventory_dir.resolve()
    out = args.output_dir.resolve()
    require(not out.exists(), f"Refusing to overwrite: {out}")
    require(not (out == root or root in out.parents),
            "Cannot write diagnostics inside raw source root")
    require(not (out == inventory or inventory in out.parents),
            "Cannot write diagnostics inside original inventory")
    report, rows = diagnostic(
        root, inventory / "unresolved_ids.csv",
        run_current_bridge=args.current_bridge,
    )
    report.update({"raw_root": str(root), "source_inventory": str(inventory)})
    out.mkdir(parents=True, exist_ok=False)
    with (out / "unmatched_replay_headers.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    for row in rows:
        print(row["replay_id"], row["header_load_status"],
              row["duration_seconds"], row["duration_category"],
              row["current_bridge_status"], row["current_bridge_reason"])


if __name__ == "__main__":
    main()
