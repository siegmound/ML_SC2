#!/usr/bin/env python3
"""B3 forensic discovery for original replay folders, build logs and manifests.

Source roots and ZIP contents are read-only. ZIP central directories are
inspected without extracting replay bytes. Printed evidence does not establish
original-generation provenance; it only locates source candidates on FAIR.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import zipfile
from collections import defaultdict
from pathlib import Path

SKIP_DIRS = frozenset({
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache",
    ".ruff_cache", ".pytest_cache", ".tox", ".cache", "site-packages",
    "dist-packages", ".conda", ".npm", "miniconda3", "anaconda3",
})
CANDIDATE_RE = re.compile(
    r"(sc2.*(build|parser|bridge|counter|constant)|buildf|"
    r"replay.*(manifest|audit|error|fail|exclu|log|processed|list|source)|"
    r"(?:raw|source).*replay|processed.failed|extract.*log|dataset.*manifest)", re.I
)
TEXT_SUFFIXES = frozenset({
    ".txt", ".log", ".json", ".jsonl", ".md", ".tex", ".py", ".csv", ".sh",
})
WITNESS = re.compile(
    r"(?<!\d)(?:24[.,_ ]?057|23[.,_ ]?241|892[.,_ ]?048|892[.,_ ]?047)(?!\d)"
    r"|Avvio estrazione massiva su|Snapshot totali\s*:|File saltati per errore\s*:"
    r"|failedToProcess|processedFiles|excluded_replays", re.I
)
ARCHIVE_RE = re.compile(r"replay|sc2|tournament|freeze|source", re.I)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def replay_magic(path: Path) -> str:
    with path.open("rb") as f:
        header = f.read(4)
    if header == b"\x00\x05\x16\x07":
        return "APPLEDOUBLE_NOT_REPLAY"
    if header in (b"MPQ\x1a", b"MPQ\x1b"):
        return "MPQ_CANDIDATE"
    return "INVALID_OR_UNKNOWN"


def replaypack_catalog(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        if not rd.fieldnames or "replaypack_filename" not in rd.fieldnames:
            raise ValueError(f"Bad replaypack catalogue: {path}")
        names = [x["replaypack_filename"] for x in rd]
    if not names or len(names) != len(set(names)):
        raise ValueError("Empty or duplicate catalogue entries")
    return set(names)


def scan(roots: list[Path], expected: set[str], *,
         max_files: int = 500000, max_archives: int = 2000,
         max_text_bytes: int = 1048576,
         on_limit: str = "error") -> tuple[dict, dict]:
    if max_files < 1 or max_archives < 1:
        raise ValueError("Invalid scan caps")
    if on_limit not in ("error", "partial"):
        raise ValueError("on_limit must be 'error' or 'partial'")
    count = 0
    seen = set()
    raw, archives, candidates, witnesses = [], [], [], []
    missing = []
    inspected_roots = []
    limit_reached = None
    interrupted_at = None
    for original in roots:
        if limit_reached is not None:
            break
        root = original.expanduser().resolve()
        if not root.is_dir():
            missing.append(str(root))
            continue
        inspected_roots.append(str(root))
        for base, dirs, names in os.walk(root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS
                             and not (Path(base) / d).is_symlink())
            for name in sorted(names):
                path = Path(base) / name
                if path.is_symlink() or not path.is_file():
                    continue
                try:
                    st = path.stat()
                except OSError:
                    continue
                identity = (st.st_dev, st.st_ino)
                if identity in seen:
                    continue
                if count >= max_files:
                    if on_limit == "error":
                        raise ValueError("File scan cap reached; narrow --roots")
                    limit_reached = "max_files"
                    interrupted_at = str(path)
                    break
                seen.add(identity)
                count += 1
                suffix = path.suffix.lower()
                if suffix == ".sc2replay":
                    try:
                        magic = replay_magic(path)
                    except OSError:
                        magic = "READ_ERROR"
                    raw.append({
                        "path": str(path), "basename": name,
                        "directory": str(path.parent), "bytes": st.st_size,
                        "magic": magic,
                    })
                    continue
                if suffix == ".zip" and (name in expected or ARCHIVE_RE.search(str(path))):
                    if len(archives) >= max_archives:
                        if on_limit == "error":
                            raise ValueError("ZIP scan cap reached; narrow --roots")
                        limit_reached = "max_archives"
                        interrupted_at = str(path)
                        break
                    entry = {
                        "path": str(path), "filename": name, "bytes": st.st_size,
                        "catalogued_zenodo_pack": name in expected,
                        "replay_members": None, "unique_replay_members": None,
                        "nested_zip_members": None, "status": "BAD_OR_UNREADABLE",
                    }
                    try:
                        with zipfile.ZipFile(path) as z:
                            names_in_zip = [x.filename for x in z.infolist() if not x.is_dir()]
                        replay_names = [
                            x.replace("\\", "/").rsplit("/", 1)[-1] for x in names_in_zip
                            if x.lower().endswith(".sc2replay")
                        ]
                        entry.update({
                            "replay_members": len(replay_names),
                            "unique_replay_members": len(set(replay_names)),
                            "nested_zip_members": sum(x.lower().endswith(".zip") for x in names_in_zip),
                            "status": "OK",
                        })
                    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile):
                        pass
                    archives.append(entry)
                if suffix in TEXT_SUFFIXES and CANDIDATE_RE.search(name):
                    candidates.append({
                        "path": str(path), "filename": name, "bytes": st.st_size,
                        "sha256": sha256(path) if st.st_size <= max_text_bytes else "NOT_HASHED_OVERSIZE",
                    })
                if suffix not in TEXT_SUFFIXES or st.st_size > max_text_bytes:
                    continue
                try:
                    data = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                for match in list(WITNESS.finditer(data))[:20]:
                    witnesses.append({
                        "path": str(path), "match": match.group(0),
                        "excerpt": " ".join(data[
                            max(0, match.start()-70):match.end()+85
                        ].split()),
                    })
            if limit_reached is not None:
                break
    groups = defaultdict(list)
    for item in raw:
        groups[item["directory"]].append(item)
    dirs = [{
        "directory": path,
        "n_extension_files": len(items),
        "n_unique_names": len({x["basename"] for x in items}),
        "n_mpq_candidates": sum(x["magic"] == "MPQ_CANDIDATE" for x in items),
        "n_appledouble": sum(x["magic"] == "APPLEDOUBLE_NOT_REPLAY" for x in items),
    } for path, items in sorted(groups.items(), key=lambda x: (-len(x[1]), x[0]))]
    pack_names = {x["filename"] for x in archives if x["catalogued_zenodo_pack"]
                  and x["status"] == "OK"}
    result = {
        "schema": "ml_sc2.b3.historical_generation_discovery.v1",
        "searched_roots": [str(x.expanduser().resolve()) for x in roots],
        "roots_started": inspected_roots,
        "roots_not_started_due_to_limit": [
            str(x.expanduser().resolve()) for x in roots
            if str(x.expanduser().resolve()) not in inspected_roots
            and str(x.expanduser().resolve()) not in missing
        ],
        "missing_roots": missing,
        "scan_complete": limit_reached is None,
        "scan_status": (
            "COMPLETE_ALL_REQUESTED_ROOTS" if limit_reached is None
            else "PARTIAL_INCOMPLETE_LIMIT_REACHED"
        ),
        "limit_reached": limit_reached,
        "interrupted_at_path": interrupted_at,
        "scan_caps": {"max_files": max_files, "max_archives": max_archives},
        "pruned_directory_names": sorted(SKIP_DIRS),
        "n_visited_files": count,
        "loose_sc2replay_files": len(raw),
        "loose_mpq_candidates": sum(x["magic"] == "MPQ_CANDIDATE" for x in raw),
        "loose_appledouble": sum(x["magic"] == "APPLEDOUBLE_NOT_REPLAY" for x in raw),
        "candidate_raw_replay_folders": dirs,
        "zip_files_examined": len(archives),
        "catalogued_zenodo_packs_found": len(pack_names),
        "catalogued_zenodo_packs_expected": len(expected),
        "catalogued_zenodo_packs_missing": (
            sorted(expected-pack_names) if limit_reached is None else None
        ),
        "direct_zip_replay_entries_not_deduplicated":
            sum(x["replay_members"] or 0 for x in archives),
        "candidate_source_files": len(candidates),
        "targeted_build_log_matches": len(witnesses),
        "no_replays_extracted_or_parsed": True,
        "historical_build_execution_attested": False,
        "historical_816_exclusions_explained": False,
    }
    return result, {
        "raw_files": raw, "zip_archives": archives,
        "source_candidates": candidates, "text_witnesses": witnesses,
    }


def save_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--roots", type=Path, nargs="+", required=True)
    p.add_argument("--pack-catalog", type=Path,
                   default=Path("data/raw/replaypack_inventory.csv"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/b3_historical_build_discovery_20261120"))
    p.add_argument("--max-files", type=int, default=500000)
    p.add_argument("--max-archives", type=int, default=2000)
    p.add_argument("--on-limit", choices=("error", "partial"),
                   default="partial",
                   help="If a scan cap is hit, save explicit incomplete evidence"
                        " rather than discard all prior discoveries")
    args = p.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        raise ValueError(f"Refusing to overwrite any prior evidence: {out}")
    catalogue = replaypack_catalog(args.pack_catalog)
    result, details = scan(args.roots, catalogue,
                           max_files=args.max_files, max_archives=args.max_archives,
                           on_limit=args.on_limit)
    result["pack_catalog_sha256"] = sha256(args.pack_catalog)
    out.mkdir(parents=True, exist_ok=False)
    (out / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    schemas = {
        "raw_files": ["path", "basename", "directory", "bytes", "magic"],
        "zip_archives": ["path", "filename", "bytes", "catalogued_zenodo_pack",
                         "replay_members", "unique_replay_members", "nested_zip_members", "status"],
        "source_candidates": ["path", "filename", "bytes", "sha256"],
        "text_witnesses": ["path", "match", "excerpt"],
    }
    for key, columns in schemas.items():
        save_csv(out / f"{key}.csv", details[key], columns)
    print(json.dumps({k: result[k] for k in (
        "scan_status", "scan_complete", "limit_reached",
        "interrupted_at_path", "roots_not_started_due_to_limit",
        "n_visited_files", "loose_sc2replay_files", "loose_mpq_candidates",
        "loose_appledouble", "candidate_raw_replay_folders",
        "catalogued_zenodo_packs_found", "zip_files_examined",
        "candidate_source_files", "targeted_build_log_matches", "missing_roots"
    )}, indent=2))
    print("Saved evidence to:", out)


if __name__ == "__main__":
    main()
