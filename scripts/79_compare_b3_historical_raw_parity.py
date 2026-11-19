#!/usr/bin/env python3
"""B3: compare historical v3.1 bridge output from real replays to frozen rows.

The historical parser/bridge/constants are *exact byte-verified copies*
from the supplied freeze archive, NOT proof of the originally used executable.
The sample is seeded from already inventoried present-in-frozen raw replay
IDs. No original training, dataset, splits or replay bytes are modified.
An observed mismatch is evidence of reconstruction difference, not evidence
of what code was used at original creation time.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import random
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_NAMES = {
    "sc2_parser_v3_counter120.py": "80ed2d83835c85fc03b4707c5431a2a29aeeb4166fc35777d3ca1bd590b3da1c",
    "sc2_bridge_v3_counter120.py": "89886649bfffba88a938f86518dd395ecc7caf0731c10520e3f96a52d12086e4",
    "sc2_constants.py": "1e568db34d1cbf8ec8ad4f6a7d8fbc8889c90d939c3289b077a74cd5232bb7d0",
    "sc2_counters.py": "ab4fad87512363d096f89d3e305d9c970aa9c0b13354eacb7d39ae64e658ce7b",
}
EXPECTED_V31_ZIP_SHA256 = "65e402884f43c03bf5450d6975e91db3d61ee7e28edf1a56483a37f15d944517"
KEYS = ("replay_id", "time_sec", "p1_wins")
REPLAY_IN_FROZEN = "SOURCE_PRESENT_FROZEN"
MAX_EXAMPLES = 80


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            d.update(block)
    return d.hexdigest()


def load_historical_source(source_zip: Path, destination: Path) -> dict[str, dict]:
    """Safely materialize only four expected scripts, with exact original SHAs."""
    require(zipfile.is_zipfile(source_zip), f"Missing/invalid historical source ZIP: {source_zip}")
    manifest = {}
    with zipfile.ZipFile(source_zip) as archive:
        for name, expected_sha in SCRIPT_NAMES.items():
            matched = [member for member in archive.namelist()
                       if Path(member).name == name and not member.endswith("/")]
            require(len(matched) == 1, f"Historical source ZIP missing/duplicates: {name}")
            payload = archive.read(matched[0])
            actual_sha = hashlib.sha256(payload).hexdigest()
            require(actual_sha == expected_sha, f"Historical source SHA mismatch: {name}")
            (destination / name).write_bytes(payload)
            manifest[name] = {"sha256": actual_sha, "bytes": len(payload)}
    return manifest


def load_historical_bridge(source_dir: Path):
    """Import isolated recovered historical files; do NOT run malformed builder."""
    sys.path.insert(0, str(source_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            "sc2_bridge_v3_counter120", source_dir / "sc2_bridge_v3_counter120.py"
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.SC2Bridge()
    finally:
        sys.path.remove(str(source_dir))


def select_present_replays(inventory_csv: Path, raw_root: Path, limit: int,
                           seed: int) -> list[tuple[str, Path]]:
    require(limit > 0, "max-replays must be positive")
    require(raw_root.is_dir(), f"Missing raw replay directory: {raw_root}")
    require(inventory_csv.is_file(), f"Missing identity inventory: {inventory_csv}")
    options = []
    seen = set()
    with inventory_csv.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames is not None and
                {"replay_id", "source_entries", "in_frozen_v3_1", "classification"} <= set(reader.fieldnames),
                "Wrong B3 identity inventory CSV schema")
        for row in reader:
            if row["classification"] != REPLAY_IN_FROZEN:
                continue
            replay_id = row["replay_id"]
            require(replay_id and replay_id not in seen, "Duplicate/blank included replay ID")
            seen.add(replay_id)
            require(row["source_entries"] == "1" and row["in_frozen_v3_1"] == "True",
                    f"Uncertain raw source membership or duplicate name: {replay_id}")
            require(Path(replay_id).name == replay_id and
                    replay_id.lower().endswith(".sc2replay"),
                    f"Invalid replay basename: {replay_id}")
            path = raw_root / replay_id
            require(path.is_file() and not path.is_symlink(), f"Raw file absent: {path}")
            with path.open("rb") as stream:
                require(stream.read(3) == b"MPQ", f"Not a plausible MPQ replay: {path}")
            options.append((replay_id, path))
    require(options, "No eligible original raw replays identified")
    rng = random.Random(seed)
    return sorted(rng.sample(sorted(options), min(limit, len(options))))


def frozen_selected(dataset_zip: Path, selected_ids: set[str]) -> pd.DataFrame:
    require(zipfile.is_zipfile(dataset_zip), f"Not a frozen CSV ZIP: {dataset_zip}")
    frames = []
    with zipfile.ZipFile(dataset_zip) as archive:
        files = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        require(len(files) == 1, "Expected exactly one CSV inside frozen dataset ZIP")
        with archive.open(files[0]) as source:
            for frame in pd.read_csv(source, dtype={"replay_id": str}, chunksize=20000):
                require(set(KEYS) <= set(frame), "Missing frozen snapshot key columns")
                match = frame[frame.replay_id.isin(selected_ids)]
                if not match.empty:
                    frames.append(match.copy())
    require(frames, "No sampled replay snapshots in frozen data")
    result = pd.concat(frames, ignore_index=True)
    require(set(result["replay_id"]) == selected_ids,
            "Missing selected replays in frozen archive")
    require(not result.duplicated(["replay_id", "time_sec"]).any(),
            "Duplicate frozen replay/time keys")
    return result


def compare_replay(replay_id: str, frozen: pd.DataFrame, generated: pd.DataFrame,
                   *, atol: float, rtol: float) -> tuple[dict, list[dict], list[dict]]:
    """Report all shared feature discrepancies, not only an aggregate PASS."""
    report = {
        "replay_id": replay_id,
        "status": "NOT_COMPARED",
        "frozen_rows": len(frozen),
        "reconstructed_rows": len(generated),
        "matched_rows": 0,
        "time_keys_only_frozen": 0,
        "time_keys_only_reconstructed": 0,
        "label_mismatches": 0,
        "feature_comparisons": 0,
        "feature_mismatches": 0,
        "missing_historical_features": "",
        "extra_historical_features": "",
    }
    require(not frozen.duplicated(["time_sec"]).any(), f"Frozen duplicate time keys: {replay_id}")
    require(not generated.duplicated(["time_sec"]).any(), f"Generated duplicate time keys: {replay_id}")
    require("time_sec" in generated and "p1_wins" in generated,
            f"Missing historical replay/time/winner columns: {replay_id}")
    actual_cols = set(generated.columns)
    expected_cols = set(frozen.columns)
    missing = sorted((expected_cols - actual_cols) - {"replay_id"})
    extra = sorted((actual_cols - expected_cols) - {"replay_id"})
    report["missing_historical_features"] = ";".join(missing)
    report["extra_historical_features"] = ";".join(extra)
    original_time = set(frozen["time_sec"])
    generated_time = set(generated["time_sec"])
    report["time_keys_only_frozen"] = len(original_time - generated_time)
    report["time_keys_only_reconstructed"] = len(generated_time - original_time)
    joined = frozen.merge(generated, on="time_sec", how="inner", suffixes=("_frozen", "_historical"),
                          validate="one_to_one")
    report["matched_rows"] = len(joined)
    features = sorted((expected_cols & actual_cols) - set(KEYS))
    per_feature = []
    examples = []
    for field in features:
        left = pd.to_numeric(joined[f"{field}_frozen"], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(joined[f"{field}_historical"], errors="coerce").to_numpy(dtype=float)
        okay = np.isclose(left, right, atol=atol, rtol=rtol, equal_nan=False)
        bad = np.flatnonzero(~okay)
        report["feature_comparisons"] += len(left)
        report["feature_mismatches"] += len(bad)
        per_feature.append({
            "replay_id": replay_id, "feature": field,
            "comparisons": len(left), "mismatches": len(bad),
        })
        for i in bad[:max(0, MAX_EXAMPLES-len(examples))]:
            examples.append({
                "replay_id": replay_id, "time_sec": joined.iloc[i]["time_sec"],
                "feature": field, "frozen_value": left[i],
                "reconstructed_value": right[i],
            })
    left_labels = pd.to_numeric(joined["p1_wins_frozen"], errors="coerce").to_numpy()
    right_labels = pd.to_numeric(joined["p1_wins_historical"], errors="coerce").to_numpy()
    report["label_mismatches"] = int((left_labels != right_labels).sum())
    clean = (
        report["frozen_rows"] == report["reconstructed_rows"]
        and report["matched_rows"] == report["frozen_rows"]
        and report["time_keys_only_frozen"] == 0
        and report["time_keys_only_reconstructed"] == 0
        and report["label_mismatches"] == 0
        and report["feature_mismatches"] == 0
        and not missing and not extra
    )
    report["status"] = "SAMPLED_PARITY" if clean else "RECONSTRUCTION_MISMATCH"
    return report, per_feature, examples


def save_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-source-zip", type=Path, required=True,
                        help="User-supplied freeze ZIP or the derived four-script audit bundle")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/replay_subsets"))
    parser.add_argument("--inventory-csv", type=Path,
                        default=Path("results/b3_source_inventory_subset_20261119/identity_reconciliation.csv"))
    parser.add_argument("--processed-zip", type=Path,
                        default=Path("data/processed/starcraft_full_dataset_v3_1_fixed.zip"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/b3_raw_to_v31_parity_20261119"))
    parser.add_argument("--max-replays", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-6)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    raw = args.raw_root.resolve()
    require(not out.exists(), f"Refusing to overwrite source or results: {out}")
    require(not (out == raw or raw in out.parents), "Output cannot be placed inside raw replay directory")
    require(math.isfinite(args.atol) and math.isfinite(args.rtol) and
            0 <= args.atol <= 1e-3 and 0 <= args.rtol <= 1e-3,
            "Invalid or too loose numerical tolerance")
    dataset = args.processed_zip.resolve()
    dataset_sha = sha256(dataset)
    require(dataset_sha == EXPECTED_V31_ZIP_SHA256,
            "Frozen v3.1 SHA-256 mismatch; aborting comparison")
    selected = select_present_replays(args.inventory_csv, raw, args.max_replays, args.seed)
    frozen = frozen_selected(dataset, {item[0] for item in selected})
    rows, per_features, examples, sources = [], [], [], []
    with tempfile.TemporaryDirectory(prefix="sc2_b3_history_") as directory:
        historical_dir = Path(directory)
        source_manifest = load_historical_source(args.historical_source_zip, historical_dir)
        historical = load_historical_bridge(historical_dir)
        for replay_id, file in selected:
            sources.append({
                "replay_id": replay_id, "raw_sha256": sha256(file),
                "raw_size_bytes": file.stat().st_size,
            })
            try:
                reconstructed = historical.process_single_replay(str(file))
                if reconstructed is None or reconstructed.empty:
                    rows.append({
                        "replay_id": replay_id, "status": "HISTORICAL_BRIDGE_RETURNED_NO_ROWS",
                        "frozen_rows": len(frozen[frozen.replay_id == replay_id]),
                        "reconstructed_rows": 0,
                    })
                    continue
                # Recreate the historical builder's post-120-second cutoff.
                reconstructed = reconstructed[reconstructed["time_sec"] > 120].copy()
                report, features, bad = compare_replay(
                    replay_id, frozen[frozen.replay_id == replay_id],
                    reconstructed, atol=args.atol, rtol=args.rtol,
                )
                rows.append(report)
                per_features.extend(features)
                examples.extend(bad)
            except Exception as exc:
                rows.append({
                    "replay_id": replay_id, "status": "RECONSTRUCTION_EXCEPTION",
                    "error_type": type(exc).__name__,
                    "error_detail": str(exc)[:300],
                })
    n_pass = sum(r.get("status") == "SAMPLED_PARITY" for r in rows)
    report = {
        "schema": "ml_sc2.b3.historical_raw_to_v31_sample.v1",
        "historical_source_copy_sha256": source_manifest,
        "historical_source_zip_sha256": sha256(args.historical_source_zip),
        "frozen_v31_sha256": dataset_sha,
        "sample_seed": args.seed, "sample_size": len(selected),
        "sampled_replay_ids": [x[0] for x in selected],
        "sampled_raw_file_sha256": sources,
        "numerical_atol": args.atol, "numerical_rtol": args.rtol,
        "sampled_parity_count": n_pass,
        "sampled_reconstruction_difference_count": len(rows)-n_pass,
        "total_feature_comparisons": sum(r.get("feature_comparisons", 0) for r in rows),
        "total_feature_mismatches": sum(r.get("feature_mismatches", 0) for r in rows),
        "historical_original_execution_attested": False,
        "whole_corpus_raw_to_final_provenance_certified": False,
        "scope_warning": "Subset bridge reconstruction using historical-source copies, not proof of original build execution or v3.2 formulas.",
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    save_csv(out / "per_replay.csv", rows, sorted({k for row in rows for k in row}))
    save_csv(out / "per_feature.csv", per_features, ["replay_id", "feature", "comparisons", "mismatches"])
    save_csv(out / "mismatch_examples.csv", examples,
             ["replay_id", "time_sec", "feature", "frozen_value", "reconstructed_value"])
    print(json.dumps(report, indent=2))
    for row in rows:
        print(row["replay_id"], row["status"],
              "rows=", row.get("matched_rows", "N/A"),
              "feature_mismatches=", row.get("feature_mismatches", "N/A"),
              "exception=", row.get("error_type", ""))


if __name__ == "__main__":
    main()
