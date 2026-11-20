#!/usr/bin/env python3
"""B3: inventory real SC2Replay sources and compare to frozen replay identities.

Read-only against inputs. ZIP members are listed using their central directory:
replay bytes are never extracted or parsed. An absent replay is NOT assigned
an inferred parse/exclusion reason; the original per-replay parser log is
required for that claim. Rejects incomplete/raw-absent scans and overwrites.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "data/raw/replaypack_inventory.csv"
DEFAULT_FROZEN = ROOT / "data/processed/starcraft_full_dataset_v3_1_fixed.zip"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_ids(archive: Path) -> tuple[dict, set[str]]:
    spec = importlib.util.spec_from_file_location(
        "b3_existing_replay_census", ROOT / "scripts/69_audit_replay_lineage.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.processed_census(archive)


def catalogued_packs(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None and "replaypack_filename" in reader.fieldnames,
                f"Invalid replaypack catalogue: {path}")
        names = [row["replaypack_filename"] for row in reader]
    require(bool(names) and len(set(names)) == len(names), "Empty or duplicate replaypack catalogue")
    return set(names)


def collect_sources(root: Path) -> tuple[list[dict], dict]:
    require(root.is_dir(), f"Missing raw directory: {root}")
    entries: list[dict] = []
    packs: list[dict] = []
    nested_zips: list[str] = []
    visited_zip_count = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        suffix = path.suffix.lower()
        if suffix == ".sc2replay":
            entries.append({
                "replay_id": path.name,
                "source_kind": "loose_replay",
                "source": relative,
                "size_bytes": path.stat().st_size,
                "crc32_hex": "",
            })
        elif suffix == ".zip":
            visited_zip_count += 1
            try:
                with zipfile.ZipFile(path) as archive:
                    records = archive.infolist()
            except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
                raise ValueError(f"Unreadable ZIP source: {relative}") from exc
            raw = [item for item in records
                   if not item.is_dir() and item.filename.lower().endswith(".sc2replay")]
            nested = [item for item in records
                      if not item.is_dir() and item.filename.lower().endswith(".zip")]
            if nested:
                nested_zips.append(relative)
            if raw:
                packs.append({
                    "path": relative, "name": path.name,
                    "size_bytes": path.stat().st_size,
                    "replay_entries": len(raw),
                })
            for item in raw:
                entries.append({
                    "replay_id": item.filename.replace("\\", "/").rsplit("/", 1)[-1],
                    "source_kind": "zip_member",
                    "source": relative + "!/" + item.filename,
                    "size_bytes": item.file_size,
                    "crc32_hex": f"{item.CRC:08x}",
                })
    require(entries, "No actual .SC2Replay entries found in raw directory or ZIPs")
    return entries, {
        "scanned_zip_files": visited_zip_count,
        "replay_bearing_packs": packs,
        "archives_with_nested_zip_members_not_recursed": nested_zips,
    }


def audit(entries: list[dict], final_ids: set[str], expected_packs: set[str],
          pack_info: dict, documented_upstream: int) -> tuple[dict, list[dict], list[dict]]:
    by_id: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        require(bool(entry["replay_id"]), "Blank replay basename in source inventory")
        by_id[entry["replay_id"]].append(entry)
    observed = set(by_id)
    raw_not_final = sorted(observed - final_ids)
    missing_final = sorted(final_ids - observed)
    duplicate_names = sorted(name for name, sources in by_id.items() if len(sources) > 1)
    packs = {p["name"] for p in pack_info["replay_bearing_packs"]}
    names = sorted(expected_packs - packs)
    unexpected = sorted(packs - expected_packs)
    identity = []
    for replay_id, sources in sorted(by_id.items()):
        identity.append({
            "replay_id": replay_id,
            "source_entries": len(sources),
            "in_frozen_v3_1": replay_id in final_ids,
            "classification": (
                "SOURCE_PRESENT_FROZEN" if replay_id in final_ids
                else "SOURCE_PRESENT_FROZEN_ABSENT_REASON_UNKNOWN"
            ),
            "source_paths": ";".join(item["source"] for item in sources),
        })
    unresolved = [{
        "replay_id": replay_id, "classification": "SOURCE_PRESENT_FROZEN_ABSENT_REASON_UNKNOWN"
    } for replay_id in raw_not_final]
    unresolved.extend({
        "replay_id": replay_id, "classification": "FROZEN_ID_MISSING_FROM_SCANNED_SOURCES"
    } for replay_id in missing_final)
    result = {
        "schema": "ml_sc2.b3.raw_inventory.v1",
        "historical_raw_to_feature_certified": False,
        "exclusion_reasons_certified": False,
        "exclusion_reason_policy": "UNKNOWN until original per-replay parser audit/log is recovered",
        "source_replay_entries": len(entries),
        "source_unique_replay_basenames": len(observed),
        "source_basename_collisions": len(duplicate_names),
        "duplicate_basename_examples": duplicate_names[:20],
        "unique_source_ids_also_frozen": len(observed & final_ids),
        "unique_source_ids_absent_frozen_unattributed": len(raw_not_final),
        "frozen_ids_missing_from_scanned_sources": len(missing_final),
        "documented_upstream_not_verified": documented_upstream,
        "source_entries_minus_documented": len(entries) - documented_upstream,
        "source_unique_minus_documented": len(observed) - documented_upstream,
        "frozen_replay_ids": len(final_ids),
        "catalogued_packs": len(expected_packs),
        "catalogued_packs_with_raw_entries": len(packs & expected_packs),
        "catalogued_packs_not_observed": names,
        "uncatalogued_replay_bearing_packs": unexpected,
        "zip_files_scanned": pack_info["scanned_zip_files"],
        "zip_files_with_uninspected_nested_archives":
            pack_info["archives_with_nested_zip_members_not_recursed"],
        "replay_bearing_packs": pack_info["replay_bearing_packs"],
        "disposition": "INVENTORY_ONLY_NO_EXCLUSION_CAUSE_ATTESTATION",
    }
    return result, identity, unresolved


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True,
                        help="Folder with original .SC2Replay or downloaded replaypack ZIPs")
    parser.add_argument("--processed-zip", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--pack-catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--documented-upstream", type=int, default=24057)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/b3_source_inventory_20261119"))
    args = parser.parse_args()
    require(args.documented_upstream > 0, "Invalid upstream count")
    destination = args.output_dir.resolve()
    require(not destination.exists(), f"Refusing to overwrite B3 audit output: {destination}")
    root = args.raw_root.resolve()
    require(not (destination == root or root in destination.parents),
            "Output directory must not be inside the raw source directory")
    require(not (destination == args.processed_zip.resolve()), "Output overlaps input")
    census, final_ids = frozen_ids(args.processed_zip)
    pack_catalog = catalogued_packs(args.pack_catalog)
    raw_entries, scanned = collect_sources(root)
    report, identity, unresolved = audit(
        raw_entries, final_ids, pack_catalog, scanned, args.documented_upstream
    )
    report["frozen_census"] = census
    report["processed_zip_sha256"] = sha256(args.processed_zip)
    report["pack_catalog_sha256"] = sha256(args.pack_catalog)
    report["raw_root"] = str(root)
    report["raw_zip_member_data_not_extracted"] = True
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_csv(destination / "raw_entries.csv",
              ["replay_id", "source_kind", "source", "size_bytes", "crc32_hex"],
              raw_entries)
    write_csv(destination / "identity_reconciliation.csv",
              ["replay_id", "source_entries", "in_frozen_v3_1",
               "classification", "source_paths"], identity)
    write_csv(destination / "unresolved_ids.csv",
              ["replay_id", "classification"], unresolved)
    print(json.dumps({
        "output_dir": str(destination),
        "source_entries": len(raw_entries),
        "unique_source_ids": report["source_unique_replay_basenames"],
        "source_basenames_also_frozen": report["unique_source_ids_also_frozen"],
        "source_unique_absent_frozen_no_reason": report["unique_source_ids_absent_frozen_unattributed"],
        "frozen_missing_in_sources": report["frozen_ids_missing_from_scanned_sources"],
        "observed_catalogue_packs": report["catalogued_packs_with_raw_entries"],
        "catalogue_size": report["catalogued_packs"],
        "duplicate_basenames": report["source_basename_collisions"],
        "status": report["disposition"],
    }, indent=2))


if __name__ == "__main__":
    main()
