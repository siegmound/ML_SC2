#!/usr/bin/env python3
"""B3: decompose single-unit-type reverse-counter residuals without fitting.

Reuses *saved* script-81 army inventories and script-82 counter hypotheses.
For player snapshots containing exactly one known army-unit type, compute
the exact opponent countered-unit fraction and the effective weight
frozen_counter / fraction. This effective weight is conditional on the
captured inventory being identical to the inventory used in the frozen
historical build; it is NOT an attested original UNIT_STATS or a proposed
repair. No re-parsing, no optimization and no modification of sources.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ABS_TOL = 1e-6
REL_TOL = 1e-6


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def module82():
    spec = importlib.util.spec_from_file_location(
        "b3_archived_orientation_for_decomposition",
        HERE / "82_test_b3_counter_orientation.py",
    )
    require(spec is not None and spec.loader is not None, "Missing script 82")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def csv_rows(path: Path, expected: set[str]) -> list[dict]:
    require(path.is_file(), f"Missing input evidence CSV: {path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None and expected <= set(reader.fieldnames),
                f"Unexpected input CSV schema: {path}")
        rows = list(reader)
    require(bool(rows), f"No input rows: {path}")
    return rows


def single_type_evidence(
    hypotheses: list[dict], inventory: dict, stats: dict, counters: dict,
    *, atol: float = ABS_TOL, rtol: float = REL_TOL,
) -> tuple[dict, list[dict]]:
    """Derive implied unit weights using *fixed* saved opponent compositions."""
    require(0 <= atol <= 1e-3 and 0 <= rtol <= 1e-3,
            "Tolerance too loose or negative")
    rows = []
    totals = {"counter_rows": 0, "non_single_type": 0, "zero_counter_fraction": 0}
    seen = set()
    for h in hypotheses:
        if h["kind"] != "counter_advantage":
            continue
        key = (h["replay_id"], int(float(h["time_sec"])), h["player"])
        require(key not in seen, f"Repeated counter observation: {key}")
        seen.add(key)
        require(key[2] in ("p1", "p2"), f"Invalid player in evidence: {key}")
        require(h["feature"] == f"counter_advantage_{key[2]}",
                f"Counter feature belongs to wrong player: {key}")
        require(key[:2] in inventory, f"Missing recorded snapshot inventory: {key}")
        players = inventory[key[:2]]
        require(set(players) == {"p1", "p2"}, f"Partial player inventory: {key}")
        own_army = players[key[2]]["army"]
        opp_army = players["p2" if key[2] == "p1" else "p1"]["army"]
        for n in own_army.values():
            require(n > 0, f"Nonpositive unit count: {key}")
        totals["counter_rows"] += 1
        if len(own_army) != 1:
            totals["non_single_type"] += 1
            continue
        (unit, unit_count), = own_army.items()
        require(unit in stats and stats[unit].get("type") == "army",
                f"Unknown army source unit: {unit}")
        opponent_total = sum(opp_army.values())
        matched_count = sum(
            count for other, count in opp_army.items()
            if unit in counters.get(other, ())
        )
        fraction = matched_count / opponent_total if opponent_total else 0.0
        frozen = float(h["frozen"])
        reversed_value = float(h["counter_relation_reversed"])
        archived = float(h["as_archived"])
        require(all(math.isfinite(x) for x in (frozen, reversed_value, archived)),
                f"Nonfinite recorded value: {key}")
        power = float(stats[unit]["power"])
        archived_weight = 0.75 + 0.25 * min(4.0, power / 20.0)
        require(math.isclose(reversed_value, fraction * archived_weight,
                             rel_tol=rtol, abs_tol=atol),
                f"Captured reverse-counter value not reproducible from inventory: {key}")
        if fraction <= 0:
            totals["zero_counter_fraction"] += 1
            # A positive frozen value here may signal omitted/changed
            # opponent context: do not manufacture an effective weight.
            continue
        implied_weight = frozen / fraction
        ratio = frozen / reversed_value
        rows.append({
            "replay_id": key[0],
            "time_sec": key[1],
            "player": key[2],
            "unit": unit,
            "unit_count": unit_count,
            "enemy_army_names_and_counts": ";".join(
                f"{name}:{opp_army[name]}" for name in sorted(opp_army)
            ),
            "enemy_unit_count": opponent_total,
            "countered_enemy_unit_count": matched_count,
            "enemy_countered_fraction": fraction,
            "archived_unit_power": power,
            "archived_unit_weight": archived_weight,
            "frozen_implied_effective_unit_weight": implied_weight,
            "frozen_to_reversed_ratio": ratio,
            "frozen_counter": frozen,
            "reconstructed_reversed_counter": reversed_value,
            "signed_frozen_minus_reversed": frozen-reversed_value,
            "mismatch_at_original_tolerance": not math.isclose(
                frozen, reversed_value, rel_tol=rtol, abs_tol=atol
            ),
        })
    require(totals["counter_rows"] == len(seen),
            "Incorrect counter observation accounting")
    require(totals["counter_rows"] > 0, "No saved counter rows")
    by_unit = defaultdict(list)
    for row in rows:
        by_unit[row["unit"]].append(row)
    per_unit = []
    for unit, subset in sorted(by_unit.items()):
        values = [r["frozen_implied_effective_unit_weight"] for r in subset]
        ratios = [r["frozen_to_reversed_ratio"] for r in subset]
        per_unit.append({
            "unit": unit,
            "n_single_type_contexts": len(subset),
            "n_mismatches": sum(r["mismatch_at_original_tolerance"] for r in subset),
            "n_distinct_enemy_army_compositions": len(set(
                r["enemy_army_names_and_counts"] for r in subset
            )),
            "archived_unit_weight": subset[0]["archived_unit_weight"],
            "implied_weight_min": min(values),
            "implied_weight_median": statistics.median(values),
            "implied_weight_max": max(values),
            "frozen_to_reverse_ratio_median": statistics.median(ratios),
        })
    queen = next((r for r in per_unit if r["unit"] == "Queen"), None)
    return {
        "schema": "ml_sc2.b3.single_type_counter_decomposition.v1",
        **totals,
        "single_type_nonzero_counter_fraction": len(rows),
        "single_type_mismatches": sum(r["mismatch_at_original_tolerance"] for r in rows),
        "by_unit": per_unit,
        "queen_only": queen,
        "interpretation": (
            "frozen_implied_effective_unit_weight=frozen/fraction_of_countered_"
            "opponent_army using script-81 recorded inventories; differences "
            "are conditional observations, NOT attested historical unit weights."
        ),
        "historical_source_execution_attested": False,
        "b3_release_gate_closed": False,
    }, rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--historical-source-zip", type=Path, default=Path(
        "sc2_b3_historical_v31_sources.zip"))
    p.add_argument("--orientation-dir", type=Path, default=Path(
        "results/b3_counter_orientation_hypothesis_20261119"))
    p.add_argument("--forensic-dir", type=Path, default=Path(
        "results/b3_combat_counter_forensics_sample3_20261119"))
    p.add_argument("--output-dir", type=Path, default=Path(
        "results/b3_single_type_counter_decomposition_20261119"))
    args = p.parse_args()
    out = args.output_dir.resolve()
    orientation_dir = args.orientation_dir.resolve()
    forensic_dir = args.forensic_dir.resolve()
    require(not out.exists(), f"Refusing to overwrite: {out}")
    require(not (
        out == orientation_dir or orientation_dir in out.parents
        or out == forensic_dir or forensic_dir in out.parents
    ), "Output inside source evidence directory")
    a = module82()
    sources = a.original_definitions(args.historical_source_zip)
    with (orientation_dir / "summary.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    require(manifest["schema"] == "ml_sc2.b3.counter_orientation_hypothesis.v1",
            "Wrong script-82 source summary schema")
    require(manifest["source_hashes"] == sources["hashes"],
            "Historical source SHAs differ from saved orientation evidence")
    inventory_path = forensic_dir / "raw_player_inventories_and_upgrades.csv"
    hypothesis_path = orientation_dir / "per_player_snapshot_hypotheses.csv"
    require(
        manifest["forensic_input_sha256"]["raw_player_inventories_and_upgrades"]
        == sha256(inventory_path),
        "Modified script-81 inventoried source evidence",
    )
    require(
        manifest["player_snapshot_count"] * 2
        == len(csv_rows(hypothesis_path, {"kind"})),
        "Unexpected player snapshot hypothesis row count",
    )
    inventories = a.parse_inventory(
        inventory_path, sources["definitions"]["UNIT_STATS"]
    )
    rows = csv_rows(hypothesis_path, {
        "replay_id", "time_sec", "player", "kind", "feature",
        "frozen", "as_archived", "counter_relation_reversed",
    })
    report, observations = single_type_evidence(
        rows, inventories, sources["definitions"]["UNIT_STATS"],
        sources["definitions"]["UNIT_COUNTERS"],
    )
    require(report["counter_rows"] == manifest["player_snapshot_count"],
            "Counter candidate row count mismatches source manifest")
    report["historical_source_zip_sha256"] = sha256(args.historical_source_zip)
    report["hypothesis_manifest_sha256"] = sha256(orientation_dir / "summary.json")
    report["hypothesis_rows_sha256"] = sha256(hypothesis_path)
    report["inventory_sha256"] = sha256(inventory_path)
    report["historical_source_hashes"] = sources["hashes"]
    out.mkdir(parents=True, exist_ok=False)
    (out / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if observations:
        with (out / "single_type_weight_evidence.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(observations[0]))
            writer.writeheader()
            writer.writerows(observations)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
