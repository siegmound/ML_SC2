#!/usr/bin/env python3
"""B3: prospectively inspect raw-only 816 and frozen-matched control replays.

Original outer ZIP and SHA-pinned v3.1 dataset are read-only. Replay
members are CRC-checked and materialized one at a time in temp storage,
then inspected via current sc2reader metadata. This is a *modern
diagnostic*, not an attested historical exclusion log or a repair
of the original executed parser. No training/feature-tuning occurs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import random
import tempfile
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import sc2reader

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_RAW_SHA = "8dcb826aeb17481d592f2d564b8a3af72bcb2f11fba9421bdbd350641970241d"
EXPECTED_V31_SHA = "65e402884f43c03bf5450d6975e91db3d61ee7e28edf1a56483a37f15d944517"


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def census_module():
    spec = importlib.util.spec_from_file_location(
        "recovered_raw_census", ROOT / "scripts/93_b3_reconcile_original_replays_zip.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def classify_replay_metadata(replay, *, min_sec: int = 180, max_sec: int = 1800) -> dict:
    """Observations from contemporary sc2reader; NOT historical reasons."""
    duration = int(getattr(getattr(replay, "length", None), "seconds", 0) or 0)
    players = getattr(replay, "players", []) or []
    n_players = len(players)
    winning_team = getattr(replay, "winner", None)
    winner_players = getattr(winning_team, "players", None) or []
    known_pids = [getattr(p, "pid", None) for p in players]
    winner_pid = getattr(winner_players[0], "pid", None) if len(winner_players) == 1 else None
    winner_valid = len(winner_players) == 1 and winner_pid in known_pids and winner_pid is not None
    conditions = {
        "duration_below_min": duration < min_sec,
        "duration_above_max": duration > max_sec,
        "not_one_vs_one": n_players != 2,
        "winner_not_single_known_player": not winner_valid,
    }
    active = [key for key, value in conditions.items() if value]
    return {
        "observed_duration_sec": duration,
        "observed_player_count": n_players,
        "observed_single_valid_winner": winner_valid,
        "current_metadata_conditions": ";".join(active) if active else "none_of_tested_conditions",
        "all_tested_metadata_conditions_pass": not active,
        "diagnostic_source": "current_sc2reader_load_level_4",
    }


def inspect_member(archive, member, *, min_sec: int, max_sec: int,
                   loader=sc2reader.load_replay):
    row = {
        "source_zip_member": member.filename,
        "crc32_hex": f"{member.CRC:08x}",
        "uncompressed_bytes": member.file_size,
        "member_sha256": "",
        "header": "",
        "observed_duration_sec": "",
        "observed_player_count": "",
        "observed_single_valid_winner": "",
        "current_metadata_conditions": "",
        "all_tested_metadata_conditions_pass": False,
        "diagnostic_source": "current_sc2reader_load_level_4",
    }
    with tempfile.TemporaryDirectory(prefix="b3_rawonly_") as tmpdir:
        target = Path(tmpdir) / "candidate.SC2Replay"
        h = hashlib.sha256()
        initial = b""
        written = 0
        with archive.open(member) as stream, target.open("xb") as out:
            while chunk := stream.read(1024 * 1024):
                if not initial:
                    initial = chunk[:4]
                h.update(chunk)
                written += len(chunk)
                out.write(chunk)
        require(written == member.file_size, "Replay ZIP CRC/extraction byte count mismatch")
        row["member_sha256"] = h.hexdigest()
        if initial in (b"MPQ\x1a", b"MPQ\x1b"):
            row["header"] = "MPQ_CANDIDATE"
        elif initial == b"\x00\x05\x16\x07":
            row["header"] = "APPLEDOUBLE_METADATA"
            row["current_metadata_conditions"] = "non_game_appledouble"
            return row
        else:
            row["header"] = "UNRECOGNIZED"
            row["current_metadata_conditions"] = "invalid_mpq_header"
            return row
        try:
            replay = loader(str(target), load_level=4)
            row.update(classify_replay_metadata(
                replay, min_sec=min_sec, max_sec=max_sec
            ))
        except Exception as error:
            row["current_metadata_conditions"] = "load_error:" + type(error).__name__
    return row


def choose_sample(raw_only: list[str], shared: list[str], *,
                  max_raw_only: int | None, controls: int, seed: int):
    require(raw_only and not (set(raw_only) & set(shared)),
            "Raw-only and frozen-matched controls must be disjoint")
    require(max_raw_only is None or max_raw_only > 0,
            "Raw-only limit must be positive or omitted")
    require(controls >= 1 and len(shared) >= controls,
            "Not enough independently selected positive controls")
    rng = random.Random(seed)
    raw_candidates = sorted(raw_only)
    if max_raw_only is not None and max_raw_only < len(raw_candidates):
        raw_candidates = sorted(rng.sample(raw_candidates, max_raw_only))
    frozen_control_candidates = sorted(rng.sample(sorted(shared), controls))
    return [(replay_id, "RAW_NOT_IN_FROZEN")
            for replay_id in raw_candidates] + [
                (replay_id, "FROZEN_MATCHED_CONTROL")
                for replay_id in frozen_control_candidates
            ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--frozen-v31-zip", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-raw-only", type=int, default=30,
                   help="Smoke sample count. Use 816 to inspect all observed unmatched.")
    p.add_argument("--controls", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-duration", type=int, default=180)
    p.add_argument("--max-duration", type=int, default=1800)
    a = p.parse_args()
    require(0 < a.min_duration < a.max_duration, "Invalid historical duration band")
    source = a.archive.resolve()
    frozen_path = a.frozen_v31_zip.resolve()
    out = a.output_dir.resolve()
    require(not out.exists(), "Refusing to overwrite prior evidence")
    require(source.is_file() and frozen_path.is_file(), "Missing source or frozen ZIP")
    mod = census_module()
    require(mod.sha256_file(source) == EXPECTED_RAW_SHA,
            "Recovered original source ZIP SHA-256 differs from operator-verified archive")
    require(mod.sha256_file(frozen_path) == EXPECTED_V31_SHA,
            "Frozen v3.1 ZIP SHA-256 differs from frozen source identity")
    _, raw_by_id, _ = mod.inventory_raw_zip(source)
    frozen_ids, frozen_rows, _ = mod.frozen_replay_ids(frozen_path)
    source_only = sorted(set(raw_by_id) - frozen_ids)
    shared = sorted(set(raw_by_id) & frozen_ids)
    require(len(source_only) == 816 and len(shared) == 23241
            and not (frozen_ids - set(raw_by_id)),
            "Recover and reconcile source identity census before this experiment")
    choices = choose_sample(
        source_only, shared, max_raw_only=a.max_raw_only,
        controls=a.controls, seed=a.seed
    )
    rows = []
    with zipfile.ZipFile(source) as archive:
        by_member_path = {x.filename: x for x in archive.infolist()}
        for idx, (replay_id, membership) in enumerate(choices, start=1):
            member = by_member_path[raw_by_id[replay_id]["zip_member"]]
            current = inspect_member(
                archive, member, min_sec=a.min_duration, max_sec=a.max_duration
            )
            rows.append({
                "replay_id": replay_id,
                "source_membership": membership,
                **current,
            })
            if idx % 100 == 0 or idx == len(choices):
                print(f"Current metadata probe: {idx}/{len(choices)}", flush=True)

    counts = {
        group: dict(sorted(Counter(
            row["current_metadata_conditions"] for row in rows
            if row["source_membership"] == group
        ).items()))
        for group in ("RAW_NOT_IN_FROZEN", "FROZEN_MATCHED_CONTROL")
    }
    summary = {
        "schema": "ml_sc2.b3.raw_only_metadata_diagnostic.v1",
        "source_zip_sha256": EXPECTED_RAW_SHA,
        "frozen_v31_zip_sha256": EXPECTED_V31_SHA,
        "frozen_snapshot_rows": frozen_rows,
        "source_raw_replay_count": len(raw_by_id),
        "source_only_replay_count": len(source_only),
        "matched_frozen_replays": len(shared),
        "selected_raw_only_replays": sum(x["source_membership"] == "RAW_NOT_IN_FROZEN" for x in rows),
        "selected_frozen_matched_controls": sum(x["source_membership"] == "FROZEN_MATCHED_CONTROL" for x in rows),
        "seed": a.seed,
        "current_duration_band_sec": [a.min_duration, a.max_duration],
        "modern_diagnostic_condition_counts": counts,
        "all_816_probe_complete": sum(x["source_membership"] == "RAW_NOT_IN_FROZEN" for x in rows) == 816,
        "raw_replay_bytes_extracted_to_persistent_storage": False,
        "original_historical_exclusion_reasons_attested": False,
        "historical_source_execution_attested": False,
        "historical_b3_gate_closed": False,
        "warning": (
            "Contemporary sc2reader diagnostics cannot establish original "
            "builder execution reasons; matched controls are essential to "
            "assess specificity and validity of inferred filters."
        ),
    }
    out.mkdir(parents=True, exist_ok=False)
    with (out / "per_replay_diagnostic.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary["per_replay_diagnostic_sha256"] = mod.sha256_file(
        out / "per_replay_diagnostic.csv"
    )
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
