#!/usr/bin/env python3
"""B3 static incompatibility audit: historical UNIT_STATS vs frozen tech levels.

Read-only original archives; do not reconstruct/repair original definitions.
Tech values 1..3 in frozen CSV are incompatible with an archived parser
whose historical UNIT_STATS lack tech_level entirely. This independently
checks the three-replay mismatch totals without rerunning sc2reader or ML.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED_CONSTANTS_SHA = "1e568db34d1cbf8ec8ad4f6a7d8fbc8889c90d939c3289b077a74cd5232bb7d0"
EXPECTED_V31_SHA = "65e402884f43c03bf5450d6975e91db3d61ee7e28edf1a56483a37f15d944517"
TECH_FIELDS = ("p1_tech_level", "p2_tech_level", "tech_diff")
FAMILIES = {
    "tech_levels": set(TECH_FIELDS),
    "combat_primitive": {"p1_combat", "p2_combat", "diff_combat_score"},
    "counter_primitive": {"counter_advantage_p1", "counter_advantage_p2", "counter_advantage_diff"},
    "combat_derived": {
        "delta_combat_p1", "delta_combat_p2", "combat_momentum_15",
        "diff_combat_60s", "diff_combat_120s", "diff_combat_vol_120",
    },
    "counter_derived": {
        "counter_advantage_diff_60s", "counter_advantage_diff_120s",
        "counter_advantage_vol_120",
    },
}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def file_sha256(path: Path) -> str:
    hashobj = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            hashobj.update(chunk)
    return hashobj.hexdigest()


def constants_from_historical_zip(path: Path) -> dict:
    require(zipfile.is_zipfile(path), f"Invalid historical source zip: {path}")
    with zipfile.ZipFile(path) as zf:
        matched = [name for name in zf.namelist()
                   if Path(name).name == "sc2_constants.py" and not name.endswith("/")]
        require(len(matched) == 1, "Exactly one historical sc2_constants.py is required")
        raw = zf.read(matched[0])
    sha = hashlib.sha256(raw).hexdigest()
    require(sha == EXPECTED_CONSTANTS_SHA, "Historical constants hash does not match audited source")
    tree = ast.parse(raw.decode("utf-8"), filename="sc2_constants.py")
    definitions = [
        node.value for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "UNIT_STATS" for t in node.targets)
    ]
    require(len(definitions) == 1, "Exactly one UNIT_STATS assignment required")
    values = ast.literal_eval(definitions[0])
    require(isinstance(values, dict) and bool(values), "Invalid UNIT_STATS literal")
    require(all(isinstance(d, dict) for d in values.values()), "Invalid unit metadata")
    return {
        "sha256": sha,
        "units": len(values),
        "units_with_explicit_tech_level": sum("tech_level" in d for d in values.values()),
        "units_with_upg_cat": sum("upg_cat" in d for d in values.values()),
        "parser_fallback_for_absent_tech_level": 0,
    }


def selected_ids(path: Path) -> set[str]:
    require(path.is_file(), f"Missing historical per-replay report: {path}")
    with path.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    require(bool(rows) and "replay_id" in rows[0], "Invalid per-replay report")
    ids = [r["replay_id"] for r in rows]
    require(all(ids) and len(ids) == len(set(ids)), "Duplicate or blank replay ID")
    return set(ids)


def frozen_tech_rows(archive_path: Path, ids: set[str]) -> dict:
    require(zipfile.is_zipfile(archive_path), "Missing frozen v3.1 CSV ZIP")
    sha = file_sha256(archive_path)
    require(sha == EXPECTED_V31_SHA, "Frozen v3.1 payload SHA-256 mismatch")
    counts = Counter()
    values = defaultdict(Counter)
    rows_per_replay = Counter()
    with zipfile.ZipFile(archive_path) as archive:
        members = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        require(len(members) == 1, "Expected exactly one frozen CSV")
        with archive.open(members[0]) as byte_stream:
            reader = csv.DictReader(io.TextIOWrapper(byte_stream, encoding="utf-8-sig"))
            require(reader.fieldnames is not None and
                    set(TECH_FIELDS + ("replay_id",)) <= set(reader.fieldnames),
                    "Frozen tech fields are missing")
            for row in reader:
                replay_id = row["replay_id"]
                if replay_id not in ids:
                    continue
                rows_per_replay[replay_id] += 1
                for feature in TECH_FIELDS:
                    val = float(row[feature])
                    require(val == val and abs(val) != float("inf"),
                            f"Nonfinite {feature} for {replay_id}")
                    values[feature][str(val)] += 1
                    if val != 0:
                        counts[feature] += 1
    require(set(rows_per_replay) == ids, "Not all sampled replays found in frozen source")
    return {
        "sha256": sha,
        "sampled_rows": sum(rows_per_replay.values()),
        "sampled_replays": len(ids),
        "nonzero_feature_counts": dict(counts),
        "value_distributions": {key: dict(sorted(values[key].items()))
                                for key in TECH_FIELDS},
    }


def mismatches_by_family(path: Path, ids: set[str]) -> dict:
    with path.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    require(bool(rows) and {"replay_id", "feature", "comparisons", "mismatches"} <= set(rows[0]),
            "Invalid per-feature mismatch CSV")
    groups = Counter()
    counts = Counter()
    per_replay_seen = set()
    for row in rows:
        require(row["replay_id"] in ids, f"Unexpected replay in feature CSV: {row['replay_id']}")
        identity = (row["replay_id"], row["feature"])
        require(identity not in per_replay_seen, "Repeated feature/replay evidence")
        per_replay_seen.add(identity)
        total, bad = int(row["comparisons"]), int(row["mismatches"])
        require(0 <= bad <= total, "Bad feature comparison totals")
        matches = [name for name, fields in FAMILIES.items() if row["feature"] in fields]
        require(len(matches) <= 1, "Overlapping feature family")
        group = matches[0] if matches else "unaffected_other"
        groups[group] += bad
        counts[row["feature"]] += bad
    return {"mismatch_counts_by_family": dict(sorted(groups.items())),
            "mismatch_counts_by_feature": dict(sorted(counts.items())),
            "total_mismatches": sum(groups.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-source-zip", type=Path,
                        default=Path("sc2_b3_historical_v31_sources.zip"))
    parser.add_argument("--processed-zip", type=Path,
                        default=Path("data/processed/starcraft_full_dataset_v3_1_fixed.zip"))
    parser.add_argument("--parity-dir", type=Path,
                        default=Path("results/b3_raw_to_v31_parity_sample3_20261119"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/b3_source_constant_incompatibility_20261119"))
    args = parser.parse_args()
    out = args.output_dir.resolve()
    require(not out.exists(), f"Refusing to overwrite: {out}")
    constants = constants_from_historical_zip(args.historical_source_zip)
    ids = selected_ids(args.parity_dir / "per_replay.csv")
    frozen = frozen_tech_rows(args.processed_zip, ids)
    mismatches = mismatches_by_family(args.parity_dir / "per_feature.csv", ids)
    incompatible = constants["units_with_explicit_tech_level"] == 0
    match = {
        feature: frozen["nonzero_feature_counts"].get(feature, 0)
        == mismatches["mismatch_counts_by_feature"].get(feature, 0)
        for feature in TECH_FIELDS
    }
    report = {
        "schema": "ml_sc2.b3.historical_unit_stats_incompatibility.v1",
        "historical_constants": constants,
        "frozen_sample": frozen,
        "mismatch_evidence": mismatches,
        "all_historical_units_lack_tech_level": incompatible,
        "tech_mismatches_equal_nonzero_frozen_levels": match,
        "proved_tech_feature_incompatibility":
            incompatible and all(match.values()) and
            any(frozen["nonzero_feature_counts"].values()),
        "counter_combat_root_cause_resolved": False,
        "historical_original_execution_attested": False,
        "b3_full_provenance_certified": False,
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
