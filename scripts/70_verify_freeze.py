"""Check archived experiment Git blobs and dataset Git LFS integrity.

`--mode pointer` is fast and works with Git LFS smudge disabled.
`--mode payload` additionally hashes both downloaded dataset archives and checks
ZIP headers against saved XGBoost model feature lists (no GPU or retraining).
Git HEAD objects are checked, so uncommitted edits do not spoof frozen history.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
import zipfile
from pathlib import Path

MANIFEST = "docs/audit/b0_baseline_manifest_2026-11-17.json"
PROTECTED_TABLES = {"tables/deep_final_summary.csv", "tables/rigorous_calibration_summary.csv"}
FEATURE_CONFIGS = {
    "v3_1_fixed": "results/xgb_full/real_v3_1_fixed_fullgpu_clean/seed_42/config_used.json",
    "v3_2_combatfix": "results/xgb_full/real_v3_2_combatfix_fullgpu_clean/seed_42/config_used.json",
}
POINTER_RE = re.compile(
    rb"version https://git-lfs.github.com/spec/v1\r?\n"
    rb"oid sha256:([0-9a-f]{64})\r?\n"
    rb"size ([0-9]+)\r?\n?\Z"
)


def git(repo: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, check=False,
    )
    if result.returncode:
        raise ValueError(f"git {' '.join(args)} failed: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout


def protected_paths(manifest: dict) -> dict[str, str]:
    tracked = manifest["tracked_artifact_git_blob_sha1"]
    return {
        path: sha for path, sha in tracked.items()
        if path.startswith(("data/", "results/")) or path in PROTECTED_TABLES
    }


def verify_git_objects(repo: Path, manifest: dict) -> dict:
    """Verify commit HEAD still contains the exact original protected Git blobs."""
    objects = {}
    for line in git(repo, "ls-tree", "-rz", "HEAD").split(b"\0"):
        if not line:
            continue
        metadata, path = line.split(b"\t", 1)
        mode, kind, sha = metadata.decode().split(" ")
        if kind == "blob":
            objects[path.decode()] = sha
    protected = protected_paths(manifest)
    if not protected:
        raise ValueError("Baseline manifest does not contain protected artifacts")
    mismatches = {
        path: {"expected": sha, "observed": objects.get(path)}
        for path, sha in protected.items()
        if objects.get(path) != sha
    }
    if mismatches:
        raise ValueError(f"Frozen Git object changed or missing: {mismatches}")
    return {"protected_artifacts": len(protected), "git_object_integrity": "PASS"}


def verify_pointers(repo: Path, manifest: dict) -> dict:
    result = {}
    for relpath, meta in manifest["datasets"].items():
        committed = git(repo, "show", f"HEAD:{relpath}")
        match = POINTER_RE.fullmatch(committed)
        if match is None:
            raise ValueError(f"Not a canonical Git LFS pointer: {relpath}")
        oid, size = match[1].decode(), int(match[2])
        if oid != meta["lfs_sha256"] or size != meta["size_bytes"]:
            raise ValueError(f"LFS pointer changed from B0: {relpath}")
        result[relpath] = {"sha256": oid, "expected_bytes": size, "pointer": "PASS"}
    if not result:
        raise ValueError("No LFS dataset pointers documented in baseline")
    return result


def archive_header(path: Path) -> list[str]:
    try:
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(members) != 1:
                raise ValueError(f"Expected exactly one CSV in {path}, got {len(members)}")
            with archive.open(members[0]) as stream:
                header = next(csv.reader(io.TextIOWrapper(stream, encoding="utf-8-sig")))
    except (zipfile.BadZipFile, EOFError, StopIteration) as exc:
        raise ValueError(f"Bad or incomplete dataset archive: {path}") from exc
    if len(header) != len(set(header)):
        raise ValueError(f"Duplicate ZIP CSV columns: {path}")
    return header


def verify_payloads(repo: Path, manifest: dict, pointers: dict) -> dict:
    details = {}
    for relpath, data in pointers.items():
        archive = repo / relpath
        if not archive.is_file():
            raise ValueError(f"LFS payload not downloaded: {archive}")
        hasher = hashlib.sha256()
        byte_count = 0
        with archive.open("rb") as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                byte_count += len(block)
                hasher.update(block)
        if byte_count != data["expected_bytes"] or hasher.hexdigest() != data["sha256"]:
            raise ValueError(f"LFS payload size/hash mismatch or pointer-only checkout: {relpath}")
        if "v3_1_fixed" in relpath:
            key = "v3_1_fixed"
        elif "v3_2_combatfix" in relpath:
            key = "v3_2_combatfix"
        else:
            raise ValueError(f"Unrecognized final dataset version: {relpath}")
        config = json.loads((repo / FEATURE_CONFIGS[key]).read_text(encoding="utf-8"))
        header = archive_header(archive)
        names = [name for name in header if name not in {"replay_id", "time_sec", "p1_wins"}]
        if names != config["features"] or len(names) != int(manifest["datasets"][relpath]["features"]):
            raise ValueError(f"Dataset ZIP header differs from published {key} feature schema")
        details[relpath] = {"sha256": data["sha256"], "bytes": byte_count,
                            "features": len(names), "zip_header": "PASS"}
    return details


def verify(repo: Path, mode: str = "pointer", manifest_path: str = MANIFEST) -> dict:
    root = Path(repo).resolve()
    manifest = json.loads((root / manifest_path).read_text(encoding="utf-8"))
    if not (root / ".git").exists():
        raise ValueError("Git checkout required; exported ZIP cannot verify commit objects")
    blobs = verify_git_objects(root, manifest)
    pointers = verify_pointers(root, manifest)
    result = {"mode": mode, **blobs, "lfs_pointers": pointers}
    if mode == "payload":
        result["lfs_payloads"] = verify_payloads(root, manifest, pointers)
    elif mode != "pointer":
        raise ValueError(f"Unknown mode: {mode}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--mode", choices=("pointer", "payload"), default="pointer")
    ap.add_argument("--manifest", default=MANIFEST)
    args = ap.parse_args()
    try:
        result = verify(args.repo_root, args.mode, args.manifest)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(json.dumps(result, indent=2))
    print(f"PASS: B0 protected Git blobs and LFS {args.mode} integrity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
