#!/usr/bin/env python3
"""Full V4 parser preflight on the 16 metadata-valid raw-only games + controls.

Reads SHA-pinned replay source outer ZIP, and the already sealed full-816
modern metadata diagnostic. Streams only selected ZIP replay members into
temporary files, preserves original basename, executes the CURRENT V4 bridge
and validates output snapshot feature frames. Writes independently identified
test and control results. NEVER infers attested HISTORICAL exclusion causes;
does not overwrite source files, train models, or extract the full corpus.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import tempfile
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from sc2proj.v4 import V4ReplayBridge
from sc2proj.v4_dataset import validate_replay_frame

EXPECTED_RAW_SHA = "8dcb826aeb17481d592f2d564b8a3af72bcb2f11fba9421bdbd350641970241d"
EXPECTED_DIAG_SHA = "0160faec6ccaae5cc95cae4589e06225ee9b5e4e692bb788135494bdf80adf57"
COLUMNS = [
    "replay_id", "source_membership", "metadata_condition",
    "source_zip_member", "source_crc32_hex", "source_uncompressed_bytes",
    "observed_source_sha256", "v4_status", "v4_reason",
    "v4_snapshots", "v4_feature_count", "v4_validation_issue",
    "v4_match_duration_sec", "v4_player_count", "v4_winner_known",
    "v4_unknown_unit_count", "v4_alias_unit_count",
]


def require(condition: bool, description: str):
    if not condition:
        raise ValueError(description)


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def select_metadata_survivors(diagnostic_csv: Path, controls: int, seed: int):
    require(controls >= 1, "At least one frozen-matched control is required")
    rows = []
    with diagnostic_csv.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames is not None and {
            "replay_id", "source_membership", "current_metadata_conditions",
            "source_zip_member", "member_sha256",
        } <= set(reader.fieldnames), "Bad metadata-diagnostic schema")
        for row in reader:
            rows.append(row)
    require(len(rows) == 916, "Diagnostic must contain all 816 and 100 controls")
    require(len({r["replay_id"] for r in rows}) == len(rows),
            "Repeated replay IDs in metadata diagnostic")
    raw_only = [r for r in rows if r["source_membership"] == "RAW_NOT_IN_FROZEN"]
    available_controls = [
        r for r in rows if r["source_membership"] == "FROZEN_MATCHED_CONTROL"
    ]
    require(len(raw_only) == 816 and len(available_controls) == 100,
            "Expected complete 816/100 contemporary diagnostic")
    require(controls <= len(available_controls), "Too many requested controls")
    survivors = sorted(
        (r for r in raw_only
         if r["current_metadata_conditions"] == "none_of_tested_conditions"),
        key=lambda x: x["replay_id"],
    )
    require(len(survivors) == 16,
            "Expected exactly 16 metadata-valid raw-only survivors; re-check input")
    rng = random.Random(seed)
    selected_controls = sorted(
        rng.sample(sorted(available_controls, key=lambda x: x["replay_id"]), controls),
        key=lambda x: x["replay_id"],
    )
    return survivors + selected_controls


def probe_selected_member(archive, member, expected_member_sha: str,
                          parser: V4ReplayBridge, *, tmpdir: Path):
    name = PurePosixPath(member.filename).name
    dest = tmpdir / name
    h = hashlib.sha256()
    bytes_written = 0
    # ZipFile.open verifies member CRC when fully consumed.
    with archive.open(member) as inp, dest.open("xb") as out:
        while chunk := inp.read(1024 * 1024):
            h.update(chunk)
            bytes_written += len(chunk)
            require(bytes_written <= 50 * 1024 * 1024,
                    "Raw member exceeds declared per-replay safety cap")
            out.write(chunk)
    require(bytes_written == member.file_size,
            f"Member size mismatch for {name}")
    digest = h.hexdigest()
    require(digest == expected_member_sha,
            f"Input raw member bytes changed relative to prior B3 evidence: {name}")
    record = {
        "observed_source_sha256": digest,
        "v4_status": "error",
        "v4_reason": "",
        "v4_snapshots": 0,
        "v4_feature_count": 0,
        "v4_validation_issue": "",
        "v4_match_duration_sec": "",
        "v4_player_count": "",
        "v4_winner_known": "",
        "v4_unknown_unit_count": "",
        "v4_alias_unit_count": "",
    }
    try:
        result = parser.process_replay(dest)
        record["v4_status"] = result.status
        record["v4_reason"] = result.reason
        m = result.metadata
        for key in (
            "match_duration_sec", "player_count", "winner_known",
            "unknown_unit_count", "alias_unit_count"
        ):
            record["v4_" + key] = m.get(key, "")
        if result.status == "ok" and result.dataframe is not None:
            frame = result.dataframe
            record["v4_snapshots"] = len(frame)
            record["v4_feature_count"] = len(frame.columns) - 3
            issue = validate_replay_frame(
                frame, name, parser.early_cut_sec, parser.snapshot_step_sec
            )
            if issue:
                record["v4_validation_issue"] = issue
                record["v4_status"] = "invalid_features"
                record["v4_reason"] = issue
        elif result.status == "ok":
            record["v4_status"] = "invalid_features"
            record["v4_validation_issue"] = "ok_without_dataframe"
            record["v4_reason"] = "ok_without_dataframe"
    except Exception as error:
        record["v4_reason"] = f"parser_exception:{type(error).__name__}"
        record["v4_validation_issue"] = str(error)[:300]
    finally:
        dest.unlink(missing_ok=True)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--metadata-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--controls", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    out_dir = a.output_dir.resolve()
    require(not out_dir.exists(), "Refusing overwrite of existing output")
    archive_path = a.archive.resolve()
    metadata_csv = a.metadata_csv.resolve()
    require(archive_path.is_file() and metadata_csv.is_file(), "Input missing")
    require(hash_file(archive_path) == EXPECTED_RAW_SHA,
            "Recovered original ZIP SHA mismatch")
    require(hash_file(metadata_csv) == EXPECTED_DIAG_SHA,
            "Previous full816 per-replay diagnostic SHA mismatch")
    selected = select_metadata_survivors(metadata_csv, a.controls, a.seed)
    results = []
    parser = V4ReplayBridge()
    with zipfile.ZipFile(archive_path) as archive:
        members = {m.filename: m for m in archive.infolist()}
        with tempfile.TemporaryDirectory(prefix="sc2_b3_v4_preflight_") as td:
            temp = Path(td)
            for index, selected_row in enumerate(selected, start=1):
                original = selected_row["source_zip_member"]
                require(original in members, f"Missing selected ZIP member: {original}")
                member = members[original]
                require(PurePosixPath(original).name == selected_row["replay_id"],
                        "Source member does not match replay ID")
                result = probe_selected_member(
                    archive, member, selected_row["member_sha256"],
                    parser, tmpdir=temp,
                )
                results.append({
                    "replay_id": selected_row["replay_id"],
                    "source_membership": selected_row["source_membership"],
                    "metadata_condition": selected_row["current_metadata_conditions"],
                    "source_zip_member": original,
                    "source_crc32_hex": f"{member.CRC:08x}",
                    "source_uncompressed_bytes": member.file_size,
                    **result,
                })
                print(
                    f"V4 preflight {index}/{len(selected)} "
                    f"[{selected_row['source_membership']}] "
                    f"{selected_row['replay_id']}: {result['v4_status']} "
                    f"{result['v4_reason']}",
                    flush=True,
                )
    counts = {
        group: dict(sorted(Counter(
            f"{r['v4_status']}:{r['v4_reason']}"
            for r in results if r["source_membership"] == group
        ).items()))
        for group in ("RAW_NOT_IN_FROZEN", "FROZEN_MATCHED_CONTROL")
    }
    report = {
        "schema": "ml_sc2.b3.current_v4_parser_preflight_survivors.v1",
        "input_raw_zip_sha256": EXPECTED_RAW_SHA,
        "input_full816_diagnostic_csv_sha256": EXPECTED_DIAG_SHA,
        "selected_unexplained_raw_only": 16,
        "selected_frozen_matched_controls": a.controls,
        "seed": a.seed,
        "current_v4_parser_outcomes": counts,
        "source_replay_bytes_extracted_to_persistent_storage": False,
        "historical_exclusion_causality_proven": False,
        "historical_raw_to_v31_feature_parity_proven": False,
        "original_b3_release_gate_closed": False,
    }
    out_dir.mkdir(parents=True, exist_ok=False)
    csv_path = out_dir / "v4_parser_preflight.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(results)
    report["v4_parser_preflight_csv_sha256"] = hash_file(csv_path)
    (out_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
