#!/usr/bin/env python3
"""B3 conservative residual profiler for script-82 controlled counter experiment.

Uses previously frozen 81/82 CSVs only. Tests whether the 144 mismatches
after relation reversal cluster by sign, zero/nonzero support, replay,
army composition or upgrades. No new numerical model, no fitting,
no source mutation and no claim of original counter algorithm.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ABS_TOL = 1e-6
REL_TOL = 1e-6
KINDS = {"combat", "counter_advantage"}


def require(ok: bool, msg: str) -> None:
    if not ok:
        raise ValueError(msg)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path: Path, required: set[str]) -> list[dict]:
    require(path.is_file(), f"Required input file absent: {path}")
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        require(reader.fieldnames is not None and required <= set(reader.fieldnames),
                f"Required source CSV columns missing: {path}")
        rows = list(reader)
    require(bool(rows), f"No source observations: {path}")
    return rows


def parse_bool(v: str) -> bool:
    require(v in ("True", "False"), f"Unexpected boolean: {v}")
    return v == "True"


def counter_category(frozen: float, predicted: float) -> str:
    near_zero_f = abs(frozen) <= ABS_TOL
    near_zero_p = abs(predicted) <= ABS_TOL
    if near_zero_f and near_zero_p:
        return "both_zero"
    if near_zero_f:
        return "frozen_zero_predicted_nonzero"
    if near_zero_p:
        return "predicted_zero_frozen_nonzero"
    if (frozen > 0) != (predicted > 0):
        return "opposite_sign"
    return "both_nonzero_same_sign"


def numeric_stats(values: list[float]) -> dict:
    require(values and all(math.isfinite(x) for x in values), "Invalid summary values")
    ordered = sorted(values)
    def percentile(frac: float) -> float:
        pos = (len(ordered)-1)*frac
        low, high = math.floor(pos), math.ceil(pos)
        return ordered[low] + (pos-low)*(ordered[high]-ordered[low])
    return {
        "n": len(values),
        "min": ordered[0],
        "p10": percentile(0.1),
        "median": statistics.median(ordered),
        "p90": percentile(0.9),
        "max": ordered[-1],
        "mean": statistics.mean(values),
    }


def grouped_profile(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "mismatches": 0}
    original_errs = [abs(x["frozen"]-x["as_archived"]) for x in rows]
    reverse_errs = [abs(x["frozen"]-x["counter_relation_reversed"]) for x in rows]
    mismatches = [x for x in rows if x["is_reversed_mismatch"]]
    factors = [
        x["frozen"]/x["counter_relation_reversed"]
        for x in rows
        if x["kind"] == "counter_advantage"
        and abs(x["counter_relation_reversed"]) > ABS_TOL
        and abs(x["frozen"]) > ABS_TOL
    ]
    return {
        "n": len(rows),
        "mismatches": len(mismatches),
        "mean_absolute_error_archived": statistics.mean(original_errs),
        "mean_absolute_error_reversed": statistics.mean(reverse_errs),
        "reversed_minus_archived_mae": (
            statistics.mean(reverse_errs)-statistics.mean(original_errs)
        ),
        "reversed_residual_signed_stats": numeric_stats(
            [x["frozen"]-x["counter_relation_reversed"] for x in rows]
        ),
        "mismatch_category_counts": dict(sorted(Counter(
            counter_category(x["frozen"], x["counter_relation_reversed"])
            for x in mismatches
        ).items())),
        "mismatch_frozen_gt_reversed": sum(
            x["frozen"] > x["counter_relation_reversed"] for x in mismatches
        ),
        "mismatch_frozen_lt_reversed": sum(
            x["frozen"] < x["counter_relation_reversed"] for x in mismatches
        ),
        "nonzero_frozen_to_reversed_ratios": (
            numeric_stats(factors) if factors else None
        ),
    }


def profile(
    hypothesis_csv: Path, raw_inventory_csv: Path, expected_count: int | None = 234
) -> tuple[dict, list[dict]]:
    hypotheses = csv_rows(
        hypothesis_csv,
        {"replay_id", "time_sec", "player", "kind", "feature", "frozen",
         "as_archived", "counter_relation_reversed",
         "archived_formula_reproduced_from_captured_inventory"},
    )
    inventory = csv_rows(
        raw_inventory_csv,
        {"replay_id", "time_sec", "player", "army_names_and_counts",
         "upgrade_events_completed", "raw_base_power_before_counters_upgrades"},
    )
    by_inv = {}
    for row in inventory:
        key = (row["replay_id"], int(float(row["time_sec"])), row["player"])
        require(key not in by_inv, f"Duplicate inventory row: {key}")
        by_inv[key] = row
    grouped = defaultdict(list)
    seen = set()
    detailed = []
    for row in hypotheses:
        kind = row["kind"]
        require(kind in KINDS, f"Unknown primitive kind: {kind}")
        key = (row["replay_id"], int(float(row["time_sec"])), row["player"])
        identity = (*key, kind)
        require(identity not in seen, f"Duplicate hypothesis comparison: {identity}")
        seen.add(identity)
        require(key in by_inv, f"Missing source inventory: {key}")
        require(row["player"] in ("p1", "p2"), f"Invalid player {row['player']}")
        require(row["feature"] == (
            f"{row['player']}_combat" if kind == "combat"
            else f"counter_advantage_{row['player']}"
        ), f"Unexpected feature mapping: {identity}")
        included = parse_bool(row["archived_formula_reproduced_from_captured_inventory"])
        orig = float(row["frozen"])
        as_archived = float(row["as_archived"])
        reversed_counter = float(row["counter_relation_reversed"])
        require(all(math.isfinite(x) for x in (orig, as_archived, reversed_counter)),
                f"Nonfinite comparison: {identity}")
        require(kind != "counter_advantage" or included,
                f"Counter formula fails independent archive baseline: {identity}")
        member = {
            "replay_id": key[0], "time_sec": key[1],
            "player": key[2], "kind": kind,
            "frozen": orig, "as_archived": as_archived,
            "counter_relation_reversed": reversed_counter,
            "reversed_residual": orig-reversed_counter,
            "abs_reversed_residual": abs(orig-reversed_counter),
            "is_reversed_mismatch": not math.isclose(
                orig, reversed_counter, abs_tol=ABS_TOL, rel_tol=REL_TOL
            ),
            "eligible": included,
            "army_names_and_counts": by_inv[key]["army_names_and_counts"],
            "upgrade_events_completed": by_inv[key]["upgrade_events_completed"],
            "raw_base_power_before_counters_upgrades": float(
                by_inv[key]["raw_base_power_before_counters_upgrades"]
            ),
        }
        detailed.append(member)
        if included:
            grouped[kind].append(member)
            grouped[f"{kind}/{key[0]}"].append(member)
            grouped[f"{kind}/{key[2]}"].append(member)
    if expected_count is not None:
        for kind in KINDS:
            require(len(grouped[kind]) <= expected_count,
                    f"Excess records for {kind}: {len(grouped[kind])}")
        require(len(grouped["counter_advantage"]) == expected_count,
                "Saved counter evidence is incomplete")
    summary = {
        "schema": "ml_sc2.b3.counter_residual_profile.v1",
        "no_training_or_formula_selection": True,
        "comparisons_in_original_hypothesis_csv": len(detailed),
        "counter_summary": grouped_profile(grouped["counter_advantage"]),
        "combat_included_summary": grouped_profile(grouped["combat"]),
        "combat_excluded_incomplete_evidence": sum(
            x["kind"] == "combat" and not x["eligible"] for x in detailed
        ),
        "counter_by_replay": {
            group.removeprefix("counter_advantage/"): grouped_profile(data)
            for group, data in sorted(grouped.items())
            if group.startswith("counter_advantage/")
            and group.removeprefix("counter_advantage/") not in ("p1", "p2")
        },
        "counter_by_player": {
            player: grouped_profile(grouped[f"counter_advantage/{player}"])
            for player in ("p1", "p2")
        },
        "raw_replay_source_hashes_not_recomputed": True,
        "historical_counter_formula_attested": False,
        "b3_release_gate_closed": False,
    }
    worst = sorted(
        (x for x in detailed if x["kind"] == "counter_advantage" and x["is_reversed_mismatch"]),
        key=lambda x: (-x["abs_reversed_residual"], x["replay_id"],
                       x["time_sec"], x["player"]),
    )
    return summary, worst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orientation-dir", type=Path, default=Path(
        "results/b3_counter_orientation_hypothesis_20261119"
    ))
    parser.add_argument("--forensic-dir", type=Path, default=Path(
        "results/b3_combat_counter_forensics_sample3_20261119"
    ))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "results/b3_counter_residual_profile_20261119"
    ))
    args = parser.parse_args()
    dest = args.output_dir.resolve()
    require(not dest.exists(), f"Refusing overwrite: {dest}")
    source = args.orientation_dir.resolve()
    forensic = args.forensic_dir.resolve()
    require(not (dest == source or source in dest.parents
                 or dest == forensic or forensic in dest.parents),
            "Output directory cannot overlap source evidence")
    with (source / "summary.json").open(encoding="utf-8") as file:
        manifest = json.load(file)
    require(manifest["schema"] == "ml_sc2.b3.counter_orientation_hypothesis.v1",
            "Unexpected source hypothesis manifest")
    inventory_path = forensic / "raw_player_inventories_and_upgrades.csv"
    hypothesis_path = source / "per_player_snapshot_hypotheses.csv"
    require(manifest["forensic_input_sha256"]["raw_player_inventories_and_upgrades"]
            == sha256(inventory_path), "Changed script-81 inventory since counter analysis")
    expected = manifest["player_snapshot_count"]
    result, worst = profile(hypothesis_path, inventory_path, expected_count=expected)
    result["source_manifest_sha256"] = sha256(source / "summary.json")
    result["hypothesis_csv_sha256"] = sha256(hypothesis_path)
    result["inventory_csv_sha256"] = sha256(inventory_path)
    require(
        result["counter_summary"]["n"]
        == manifest["by_primitive_family"]["counter_advantage"]["as_archived"]["comparisons"],
        "Counter source comparison count differs"
    )
    require(
        result["counter_summary"]["mismatches"]
        == manifest["by_primitive_family"]["counter_advantage"]["counter_relation_reversed"]["mismatches"],
        "Counter source mismatch count differs"
    )
    dest.mkdir(parents=True, exist_ok=False)
    (dest / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    columns = [
        "replay_id", "time_sec", "player", "frozen", "as_archived",
        "counter_relation_reversed", "reversed_residual", "abs_reversed_residual",
        "army_names_and_counts", "upgrade_events_completed",
        "raw_base_power_before_counters_upgrades"
    ]
    with (dest / "top_counter_residuals.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for item in worst[:35]:
            writer.writerow({key: item[key] for key in columns})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
