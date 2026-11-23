#!/usr/bin/env python3
"""Safely materialize only .SC2Replay members from a pinned original outer ZIP.

Never extracts embedded tournament ZIPs. Every copied replay is streamed
through ZipFile.open (CRC-checked), SHA-256 hashed, stored under a new output
root by original unique filename and mapped back to its exact ZIP member.
Always refuses to overwrite existing extraction outputs. Does not parse
games or assign original historical rejection reasons.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import stat
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

PINNED_ARCHIVE_SHA = "8dcb826aeb17481d592f2d564b8a3af72bcb2f11fba9421bdbd350641970241d"


def require(condition, description):
    if not condition:
        raise ValueError(description)


def digest_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for b in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def source_members(archive: zipfile.ZipFile, *,
                   max_member_bytes: int, max_total_bytes: int):
    items = []
    seen_paths = set()
    seen_names = set()
    total = 0
    for item in archive.infolist():
        if item.is_dir() or not item.filename.lower().endswith(".sc2replay"):
            continue
        pieces = item.filename.split("/")
        require(
            not item.filename.startswith("/") and "\\" not in item.filename
            and all(x not in ("", ".", "..") for x in pieces)
            and not stat.S_ISLNK(item.external_attr >> 16),
            f"Unsafe replay ZIP member: {item.filename}",
        )
        basename = PurePosixPath(item.filename).name
        require(item.filename not in seen_paths, "Duplicate original ZIP member path")
        require(basename not in seen_names, f"Ambiguous basename in original ZIP: {basename}")
        seen_paths.add(item.filename)
        seen_names.add(basename)
        require(4 <= item.file_size <= max_member_bytes,
                f"Replay exceeds per-file safety limits: {basename}")
        total += item.file_size
        require(total <= max_total_bytes,
                "Total declared raw replay bytes exceed safety limit")
        items.append(item)
    require(bool(items), "Original ZIP has no direct .SC2Replay members")
    return sorted(items, key=lambda x: x.filename), total


def extract_replays(archive_path: Path, output_dir: Path, *,
                    expected_sha: str, expected_count: int | None = 24057,
                    max_member_bytes: int = 50 * 1024 * 1024,
                    max_total_bytes: int = 50 * 1024 * 1024 * 1024,
                    execute: bool = False):
    source = archive_path.resolve()
    output_dir = output_dir.resolve()
    require(source.is_file(), f"Raw source ZIP missing: {source}")
    require(not output_dir.exists(), f"Refusing overwrite of existing extraction: {output_dir}")
    require(expected_count is None or expected_count >= 1,
            "Invalid expected replay count")
    actual_sha = digest_file(source)
    require(actual_sha == expected_sha,
            "The source raw ZIP SHA-256 changed since the original census")
    with zipfile.ZipFile(source) as archive:
        members, byte_count = source_members(
            archive, max_member_bytes=max_member_bytes,
            max_total_bytes=max_total_bytes
        )
        require(expected_count is None or len(members) == expected_count,
                f"Expected {expected_count} raw replay members, found {len(members)}")
        plan = {
            "schema": "ml_sc2.v4.original_outer_zip_safe_extraction.v1",
            "source_archive": str(source),
            "source_archive_sha256": actual_sha,
            "selected_direct_raw_replay_members": len(members),
            "declared_raw_replay_bytes": byte_count,
            "embedded_tournament_zip_members_extracted": 0,
            "game_events_parsed": False,
            "historical_exclusion_reasons_attested": False,
        }
        if not execute:
            return {"status": "PLAN_ONLY_NO_OUTPUTS_CREATED", **plan}
        output_dir.mkdir(parents=True, exist_ok=False)
        raw_dir = output_dir / "replays"
        raw_dir.mkdir()
        records = []
        for idx, member in enumerate(members, start=1):
            name = PurePosixPath(member.filename).name
            target = raw_dir / name
            h = hashlib.sha256()
            size = 0
            first_bytes = b""
            # Read to end to force zipfile's CRC check.
            with archive.open(member) as source_stream, target.open("xb") as destination:
                while data := source_stream.read(1024 * 1024):
                    if size == 0:
                        first_bytes = data[:4]
                    h.update(data)
                    destination.write(data)
                    size += len(data)
                    require(size <= max_member_bytes, "Streaming replay size exceeded limit")
            require(size == member.file_size,
                    f"Uncompressed bytes disagree with central directory: {name}")
            if first_bytes in (b"MPQ\x1a", b"MPQ\x1b"):
                file_type = "MPQ_CANDIDATE"
            elif first_bytes == b"\x00\x05\x16\x07":
                file_type = "APPLEDOUBLE_NOT_GAME"
            else:
                file_type = "UNRECOGNIZED_MAGIC"
            records.append({
                "replay_id_original_basename": name,
                "outer_zip_member": member.filename,
                "uncompressed_bytes": size,
                "crc32_hex": f"{member.CRC:08x}",
                "sha256": h.hexdigest(),
                "header_type": file_type,
            })
            if idx % 3000 == 0 or idx == len(members):
                print(f"Verified raw ZIP replay extraction: {idx}/{len(members)}", flush=True)

        unique_contents = {row["sha256"] for row in records}
        classifications = Counter(row["header_type"] for row in records)
        plan.update({
            "status": "EXTRACTED_AND_CRC_VERIFIED",
            "extracted_replay_paths": len(records),
            "byte_unique_replay_contents": len(unique_contents),
            "duplicate_binary_content_count": len(records)-len(unique_contents),
            "header_type_counts": dict(sorted(classifications.items())),
            "extracted_raw_root": str(raw_dir),
        })
        record_path = output_dir / "original_replay_source_manifest.csv"
        with record_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        plan["replay_source_manifest_sha256"] = digest_file(record_path)
        (output_dir / "manifest.json").write_text(
            json.dumps(plan, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--expected-sha256", default=PINNED_ARCHIVE_SHA)
    p.add_argument("--expected-replays", type=int, default=24057)
    p.add_argument("--max-member-mb", type=int, default=50)
    p.add_argument("--max-total-gb", type=int, default=50)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--plan", action="store_true", help="Default; no files written")
    group.add_argument("--extract", action="store_true", help="Copy replay members into new root")
    a = p.parse_args()
    require(0 < a.max_member_mb <= 1024, "Invalid max member size")
    require(0 < a.max_total_gb <= 1000, "Invalid total replay cap")
    report = extract_replays(
        a.archive, a.output_dir,
        expected_sha=a.expected_sha256,
        expected_count=a.expected_replays,
        max_member_bytes=a.max_member_mb * 1024 * 1024,
        max_total_bytes=a.max_total_gb * 1024 * 1024 * 1024,
        execute=a.extract,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
