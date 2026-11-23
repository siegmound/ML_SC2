#!/usr/bin/env python3
"""Read-only, SHA-pinned census of recovered raw replay ZIP vs frozen v3.1 IDs.

Does not extract or parse replays; does not guess why any game is absent.
The single outer ZIP has 24,057 directly stored .SC2Replay file members
and 71 separate embedded tournament ZIPs. Never double-count contents of
embedded ZIPs. This is inventory provenance, not historical builder parity.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath

EXPECTED_RAW_SHA = "8dcb826aeb17481d592f2d564b8a3af72bcb2f11fba9421bdbd350641970241d"
EXPECTED_V31_SHA = "65e402884f43c03bf5450d6975e91db3d61ee7e28edf1a56483a37f15d944517"


def ensure(test: bool, description: str):
    if not test:
        raise ValueError(description)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def valid_zip_path(member: zipfile.ZipInfo) -> bool:
    name = member.filename
    normalized = PurePosixPath(name)
    mode = member.external_attr >> 16
    return (
        not member.is_dir()
        and not name.startswith("/")
        and "\\" not in name and "\x00" not in name
        and not any(x in (".", "..", "") for x in name.split("/"))
        and not stat.S_ISLNK(mode)
        and not normalized.is_absolute()
    )


def inventory_raw_zip(path: Path) -> tuple[dict, dict[str, dict], list[dict]]:
    ensure(zipfile.is_zipfile(path), "Not a valid outer raw ZIP")
    raw_by_id: dict[str, dict] = {}
    other_entries = []
    seen_member_names = set()
    nested_zip_count = 0
    json_count = 0
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            if item.is_dir():
                continue
            ensure(valid_zip_path(item), f"Unsafe ZIP member path/type: {item.filename}")
            ensure(item.filename not in seen_member_names,
                   f"Repeated central-directory member path: {item.filename}")
            seen_member_names.add(item.filename)
            ext = PurePosixPath(item.filename).suffix.lower()
            if ext == ".sc2replay":
                basename = PurePosixPath(item.filename).name
                ensure(basename not in raw_by_id,
                       f"Ambiguous repeated replay basename: {basename}")
                raw_by_id[basename] = {
                    "replay_id": basename,
                    "zip_member": item.filename,
                    "crc32_hex": f"{item.CRC:08x}",
                    "uncompressed_bytes": item.file_size,
                    "compressed_bytes": item.compress_size,
                }
            else:
                if ext == ".zip":
                    nested_zip_count += 1
                if ext == ".json":
                    json_count += 1
                row = {
                    "member": item.filename, "suffix": ext,
                    "uncompressed_bytes": item.file_size,
                    "compressed_bytes": item.compress_size,
                    "crc32_hex": f"{item.CRC:08x}",
                    "json_top_level_type": "",
                    "json_top_level_keys": "",
                    "json_preview_status": "NOT_JSON",
                }
                # Embedded JSON can be crucial original provenance.
                # Read only bounded JSON, refuse oversized decompression.
                if ext == ".json":
                    if item.file_size > 2_000_000:
                        row["json_preview_status"] = "OVERSIZE_NOT_READ"
                    else:
                        with archive.open(item) as stream:
                            data = stream.read(2_000_001)
                        ensure(len(data) <= 2_000_000,
                               "JSON decompression exceeds declared safety cap")
                        row["json_sha256"] = hashlib.sha256(data).hexdigest()
                        try:
                            parsed = json.loads(data.decode("utf-8-sig"))
                            row["json_top_level_type"] = type(parsed).__name__
                            if isinstance(parsed, dict):
                                row["json_top_level_keys"] = ";".join(sorted(
                                    str(k) for k in parsed.keys()
                                )[:60])
                            row["json_preview_status"] = "OK"
                        except (ValueError, UnicodeError):
                            row["json_preview_status"] = "INVALID_JSON"
                other_entries.append(row)
    ensure(raw_by_id, "No directly accessible raw replay members")
    return {
        "direct_raw_replay_entries": len(raw_by_id),
        "nonreplay_file_entries": len(other_entries),
        "nested_zip_entries_not_recursed": nested_zip_count,
        "json_entries": json_count,
        "all_file_entries": len(seen_member_names),
        "total_raw_replay_uncompressed_bytes": sum(
            row["uncompressed_bytes"] for row in raw_by_id.values()
        ),
    }, raw_by_id, other_entries


def frozen_replay_ids(path: Path) -> tuple[set[str], int, str]:
    import pandas as pd

    seen_ids: set[str] = set()
    total_rows = 0
    with zipfile.ZipFile(path) as archive:
        csv_members = [
            m.filename for m in archive.infolist()
            if not m.is_dir() and m.filename.lower().endswith(".csv")
        ]
        ensure(len(csv_members) == 1, "Expected a single frozen v3.1 CSV")
        with archive.open(csv_members[0]) as stream:
            for part in pd.read_csv(
                stream, usecols=["replay_id"],
                dtype={"replay_id": str}, chunksize=50000
            ):
                ensure(part["replay_id"].notna().all(), "Null frozen replay ID")
                seen_ids.update(part["replay_id"].tolist())
                total_rows += len(part)
    return seen_ids, total_rows, csv_members[0]


def compare_identities(raw_by_id: dict[str, dict], frozen: set[str]):
    raw_ids = set(raw_by_id)
    shared = sorted(raw_ids & frozen)
    only_raw = sorted(raw_ids - frozen)
    only_frozen = sorted(frozen - raw_ids)
    result = {
        "raw_unique_replay_basenames": len(raw_ids),
        "frozen_unique_replay_ids": len(frozen),
        "raw_ids_also_present_frozen": len(shared),
        "raw_ids_absent_frozen_reason_unknown": len(only_raw),
        "frozen_ids_absent_raw_archive": len(only_frozen),
        "raw_minus_frozen_arithmetic": len(raw_ids)-len(frozen),
        "source_minus_final_is_exact_original_816": (
            len(only_raw) == 816 and not only_frozen
        ),
        "upstream_exclusion_causes_attested": False,
        "historical_parser_executed_version_attested": False,
        "original_b3_release_gate_closed": False,
    }
    return result, only_raw, only_frozen


def save_csv(path: Path, fields: list[str], rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-zip", required=True, type=Path)
    p.add_argument("--frozen-v31-zip", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--raw-sha256", default=EXPECTED_RAW_SHA)
    p.add_argument("--frozen-sha256", default=EXPECTED_V31_SHA)
    p.add_argument("--expected-raw-replays", type=int, default=24057)
    a = p.parse_args()
    destination = a.output_dir.resolve()
    ensure(not destination.exists(), "Refusing to overwrite any existing evidence")
    raw_zip = a.raw_zip.resolve()
    frozen_zip = a.frozen_v31_zip.resolve()
    ensure(raw_zip.is_file() and frozen_zip.is_file(), "Input ZIP is missing")
    ensure(destination not in (raw_zip, frozen_zip), "Cannot overwrite input")
    raw_digest = sha256_file(raw_zip)
    frozen_digest = sha256_file(frozen_zip)
    ensure(raw_digest == a.raw_sha256, "Raw archive SHA-256 does not match pinned source")
    ensure(frozen_digest == a.frozen_sha256,
           "Frozen v3.1 ZIP SHA-256 does not match pinned immutable source")
    inventory, raw_by_id, metadata = inventory_raw_zip(raw_zip)
    ensure(inventory["direct_raw_replay_entries"] == a.expected_raw_replays,
           "Raw replay count differs from expected source census")
    frozen, rows, member_name = frozen_replay_ids(frozen_zip)
    comparison, unmatched, final_missing = compare_identities(raw_by_id, frozen)
    report = {
        "schema": "ml_sc2.b3.recovered_original_raw_zip_vs_frozen.v1",
        "raw_zip": str(raw_zip),
        "raw_zip_sha256": raw_digest,
        "frozen_zip_sha256": frozen_digest,
        "frozen_csv_member": member_name,
        "frozen_snapshots": rows,
        **inventory,
        **comparison,
        "nested_tournament_zip_members_not_double_counted": True,
        "raw_replay_bytes_extracted_or_parsed": False,
        "exclusion_reason_policy": (
            "UNKNOWN without independently validated historical exclusion log "
            "or replay-level failure evidence"
        ),
    }
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8"
    )
    save_csv(
        destination / "replay_identity_reconciliation.csv",
        ["replay_id", "zip_member", "crc32_hex", "uncompressed_bytes",
         "compressed_bytes", "in_frozen_v31", "classification"],
        [{
            **row, "in_frozen_v31": replay_id in frozen,
            "classification": (
                "SOURCE_PRESENT_FROZEN" if replay_id in frozen
                else "SOURCE_PRESENT_FROZEN_ABSENT_REASON_UNKNOWN"
            ),
        } for replay_id, row in sorted(raw_by_id.items())],
    )
    save_csv(
        destination / "raw_not_in_frozen.csv",
        ["replay_id", "zip_member", "crc32_hex", "uncompressed_bytes",
         "compressed_bytes", "reason"],
        [{
            **raw_by_id[replay_id], "reason": "UNKNOWN"
        } for replay_id in unmatched],
    )
    save_csv(
        destination / "frozen_not_in_raw.csv",
        ["replay_id"],
        [{"replay_id": name} for name in final_missing],
    )
    save_csv(
        destination / "outer_zip_other_members.csv",
        ["member", "suffix", "uncompressed_bytes", "compressed_bytes",
         "crc32_hex", "json_top_level_type", "json_top_level_keys",
         "json_preview_status", "json_sha256"],
        metadata,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
