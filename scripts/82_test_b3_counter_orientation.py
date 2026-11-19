#!/usr/bin/env python3
"""B3 controlled *diagnostic* counter-direction hypotheses (read-only).

Uses already captured script-81 raw inventories/upgrades and primitive
residuals; does not reparse replay events or re-train models. Evaluates
the archived UNIT_COUNTERS direction versus its reversal for BOTH
counter and combat scores, keeping all unit powers/upgrade rules fixed.
A closer fit is a hypothesis, never proof of the historical generator.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED = {
    "sc2_constants.py": "1e568db34d1cbf8ec8ad4f6a7d8fbc8889c90d939c3289b077a74cd5232bb7d0",
    "sc2_counters.py": "ab4fad87512363d096f89d3e305d9c970aa9c0b13354eacb7d39ae64e658ce7b",
}
FEATURES = ("p1_combat", "p2_combat", "counter_advantage_p1", "counter_advantage_p2")
TOL = 1e-6


def require(ok: bool, description: str) -> None:
    if not ok:
        raise ValueError(description)


def file_hash(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def literal_assignments(raw: bytes, names: set[str], label: str) -> dict:
    tree = ast.parse(raw.decode("utf-8"), filename=label)
    found = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in names:
                require(target.id not in found, f"Duplicate {target.id} in {label}")
                found[target.id] = ast.literal_eval(node.value)
    require(set(found) == names, f"Expected definitions {sorted(names)} in {label}")
    require(all(isinstance(x, dict) for x in found.values()), f"Invalid dicts in {label}")
    return found


def original_definitions(archive: Path) -> dict:
    require(zipfile.is_zipfile(archive), f"Missing historical source ZIP: {archive}")
    source = {}
    manifest = {}
    with zipfile.ZipFile(archive) as z:
        for name, target_sha in EXPECTED.items():
            members = [member for member in z.namelist()
                       if Path(member).name == name and not member.endswith("/")]
            require(len(members) == 1, f"Missing/duplicated historical {name}")
            raw = z.read(members[0])
            digest = hashlib.sha256(raw).hexdigest()
            require(digest == target_sha, f"Historical {name} SHA mismatch")
            source[name] = raw
            manifest[name] = digest
    definitions = literal_assignments(
        source["sc2_constants.py"], {"UNIT_STATS", "UNIT_TECH_MULTIPLIERS"},
        "sc2_constants.py",
    )
    definitions.update(literal_assignments(
        source["sc2_counters.py"], {"UNIT_COUNTERS"}, "sc2_counters.py"
    ))
    return {"definitions": definitions, "hashes": manifest}


def parse_army(value: str, stats: dict) -> dict[str, int]:
    army = {}
    if not value:
        return army
    for part in value.split(";"):
        require(":" in part, f"Invalid recorded army item: {part}")
        unit, text = part.rsplit(":", 1)
        require(unit in stats and stats[unit].get("type") == "army",
                f"Unrecognized known army unit: {unit}")
        require(unit not in army, f"Duplicated army unit: {unit}")
        number = int(text)
        require(number > 0, f"Invalid army count: {unit}={number}")
        army[unit] = number
    return army


def parse_inventory(path: Path, stats: dict) -> dict[tuple[str, int], dict]:
    require(path.is_file(), f"Missing script-81 inventory: {path}")
    by_snapshot: dict[tuple[str, int], dict] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {"replay_id", "time_sec", "player", "army_names_and_counts",
                    "raw_base_power_before_counters_upgrades",
                    "upgrade_events_completed"}
        require(reader.fieldnames is not None and required <= set(reader.fieldnames),
                "Incomplete script-81 inventory columns")
        for row in reader:
            key = (row["replay_id"], int(float(row["time_sec"])))
            player = row["player"]
            require(player in {"p1", "p2"} and player not in by_snapshot[key],
                    f"Missing/duplicate player inventory for {key}")
            army = parse_army(row["army_names_and_counts"], stats)
            base = sum(stats[u]["power"] * n for u, n in army.items()) / 100.0
            require(math.isclose(base, float(row["raw_base_power_before_counters_upgrades"]),
                                 rel_tol=0, abs_tol=1e-8),
                    f"Army base power does not reproduce script-81 evidence: {key} {player}")
            by_snapshot[key][player] = {
                "army": army,
                "upgrades": set(filter(None, row["upgrade_events_completed"].split(";"))),
            }
    require(by_snapshot, "No recorded inventory rows")
    require(all(set(players) == {"p1", "p2"} for players in by_snapshot.values()),
            "Missing one player at a recorded time")
    return dict(by_snapshot)


def relation(own: str, enemy: str, counters: dict, *, reverse: bool) -> bool:
    """Table entries list opponents effective against the keyed unit."""
    if reverse:
        return own in counters.get(enemy, ())
    return enemy in counters.get(own, ())


def counter_score(own: dict, enemy: dict, stats: dict,
                  counters: dict, *, reverse: bool) -> float:
    own_total, enemy_total = sum(own.values()), sum(enemy.values())
    if own_total <= 0 or enemy_total <= 0:
        return 0.0
    score = 0.0
    for unit, count in own.items():
        countered_enemy = sum(
            n for other, n in enemy.items()
            if relation(unit, other, counters, reverse=reverse)
        )
        if not countered_enemy:
            continue
        power = stats.get(unit, {}).get("power", 1.0)
        weight = 0.75 + 0.25 * min(4.0, power / 20.0)
        score += count * (countered_enemy / enemy_total) * weight
    return score / own_total


def upgrade_multiplier(unit: str, upgrades: set[str],
                       stats: dict, multipliers: dict) -> float:
    # Exact archived _get_upgrade_multiplier implementation (not repaired).
    bonus = 0.0
    category = stats.get(unit, {}).get("upg_cat", "None")
    for upg in upgrades:
        if (category != "None" and category in upg) or ("Shields" in upg):
            if any(level in upg for level in ("Level1", "Level2", "Level3")):
                bonus += 0.10
    if unit in multipliers:
        for tech in multipliers[unit]:
            if any(tech in upgrade for upgrade in upgrades):
                bonus += 0.15
    return 1.0 + bonus


def combat_score(own: dict, enemy: dict, upgrades: set[str],
                 stats: dict, counters: dict, multipliers: dict,
                 *, reverse: bool) -> float:
    total = 0.0
    for unit, count in own.items():
        if count <= 0:
            continue
        penalty = sum(
            0.15 for enemy_unit, n in enemy.items()
            if n > 0 and relation(unit, enemy_unit, counters, reverse=reverse)
        )
        counter = max(0.4, 1.0 - penalty)
        total += (stats[unit]["power"] * count * counter *
                  upgrade_multiplier(unit, upgrades, stats, multipliers))
    return total / 100.0


def read_primitive_evidence(path: Path) -> dict[tuple[str, int, str], tuple[float, float]]:
    require(path.is_file(), f"Missing script-81 residual file: {path}")
    evidence = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"replay_id", "time_sec", "feature", "frozen", "historical_reconstructed"}
        require(reader.fieldnames is not None and required <= set(reader.fieldnames),
                "Invalid script-81 primitive residual CSV")
        for row in reader:
            if row["feature"] not in FEATURES:
                continue
            key = (row["replay_id"], int(float(row["time_sec"])), row["feature"])
            require(key not in evidence, f"Duplicate primitive residual: {key}")
            nums = (float(row["frozen"]), float(row["historical_reconstructed"]))
            require(all(math.isfinite(v) for v in nums), f"Nonfinite primitive: {key}")
            evidence[key] = nums
    require(evidence, "No primitive residuals")
    return evidence


def metrics(actual: list[float], predicted: list[float]) -> dict:
    require(len(actual) == len(predicted) and bool(actual), "Empty/mismatched metrics input")
    absolute = [abs(a-b) for a, b in zip(actual, predicted)]
    mismatches = sum(
        not math.isclose(a, b, rel_tol=TOL, abs_tol=TOL)
        for a, b in zip(actual, predicted)
    )
    return {
        "comparisons": len(actual), "mismatches": mismatches,
        "mean_absolute_error": sum(absolute) / len(absolute),
        "max_absolute_error": max(absolute),
    }


def evaluate(inventories: dict, evidence: dict, definitions: dict) -> tuple[dict, list[dict]]:
    stats = definitions["UNIT_STATS"]
    counters = definitions["UNIT_COUNTERS"]
    upgrades = definitions["UNIT_TECH_MULTIPLIERS"]
    expected = set((r, t, field) for r, t in inventories for field in FEATURES)
    require(set(evidence) == expected,
            "Script-81 primitive residual keys differ from the captured player inventory snapshot keys")
    rows = []
    for (replay, time), sides in sorted(inventories.items()):
        for p1_or_p2 in ("p1", "p2"):
            other = "p2" if p1_or_p2 == "p1" else "p1"
            owned, enemy = sides[p1_or_p2], sides[other]
            army, enemy_army = owned["army"], enemy["army"]
            frozen_vals = {}
            for kind in ("combat", "counter_advantage"):
                feature = f"{p1_or_p2}_{kind}" if kind == "combat" else f"counter_advantage_{p1_or_p2}"
                frozen, archived = evidence[(replay, time, feature)]
                if kind == "combat":
                    direct = combat_score(army, enemy_army, owned["upgrades"],
                                          stats, counters, upgrades, reverse=False)
                    reversed_score = combat_score(army, enemy_army, owned["upgrades"],
                                                  stats, counters, upgrades, reverse=True)
                else:
                    direct = counter_score(army, enemy_army, stats, counters, reverse=False)
                    reversed_score = counter_score(army, enemy_army, stats, counters, reverse=True)
                archived_match = math.isclose(direct, archived, rel_tol=TOL, abs_tol=TOL)
                # The captured script-81 army metadata excludes non-army/unknown
                # opponent types. Historical combat penalty loops ALL raw opponent
                # types, e.g. Overseer, so some combat rows cannot be reconstructed
                # from this compressed evidence. Exclude, never score such rows.
                if kind == "counter_advantage":
                    require(archived_match,
                            f"Recomputed archived {feature} differs from script-81 reconstruction: {replay}@{time}")
                upgraded = any(
                    any(level in u for level in ("Level1", "Level2", "Level3"))
                    and "Shields" not in u for u in owned["upgrades"]
                )
                rows.append({
                    "replay_id": replay, "time_sec": time, "player": p1_or_p2,
                    "kind": kind, "feature": feature,
                    "frozen": frozen, "as_archived": direct,
                    "counter_relation_reversed": reversed_score,
                    "archive_abs_error": abs(frozen - direct),
                    "reversed_abs_error": abs(frozen - reversed_score),
                    "archived_formula_reproduced_from_captured_inventory": archived_match,
                    "generic_level_upgrade_observed": upgraded,
                    "army_unit_count": sum(army.values()),
                    "army_types": len(army),
                })
    report = {}
    for kind in ("combat", "counter_advantage"):
        subset = [x for x in rows if x["kind"] == kind]
        eligible = [x for x in subset
                    if x["archived_formula_reproduced_from_captured_inventory"]]
        truth = [x["frozen"] for x in eligible]
        archived = [x["as_archived"] for x in eligible]
        reversed_values = [x["counter_relation_reversed"] for x in eligible]
        report[kind] = {
            "as_archived": metrics(truth, archived) if eligible else None,
            "counter_relation_reversed": metrics(truth, reversed_values) if eligible else None,
            "excluded_due_to_incomplete_captured_inventory_context": len(subset)-len(eligible),
            "generic_level_upgrade_observed": sum(x["generic_level_upgrade_observed"] for x in eligible),
            "mismatch_with_no_generic_level_upgrade": sum(
                not math.isclose(x["frozen"], x["as_archived"], rel_tol=TOL, abs_tol=TOL)
                and not x["generic_level_upgrade_observed"] for x in eligible
            ),
        }
    return report, rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--historical-source-zip", type=Path,
                   default=Path("sc2_b3_historical_v31_sources.zip"))
    p.add_argument("--forensic-dir", type=Path, default=Path(
        "results/b3_combat_counter_forensics_sample3_20261119"))
    p.add_argument("--output-dir", type=Path, default=Path(
        "results/b3_counter_orientation_hypothesis_20261119"))
    args = p.parse_args()
    out = args.output_dir.resolve()
    forensic = args.forensic_dir.resolve()
    require(not out.exists(), f"Refusing to overwrite output: {out}")
    require(out != forensic and forensic not in out.parents,
            "Cannot place new output inside original evidence folder")
    source = original_definitions(args.historical_source_zip)
    inventories = parse_inventory(
        forensic / "raw_player_inventories_and_upgrades.csv",
        source["definitions"]["UNIT_STATS"],
    )
    residual_path = forensic / "per_snapshot_primitive_residual.csv"
    residuals = read_primitive_evidence(residual_path)
    with (forensic / "summary.json").open(encoding="utf-8") as f:
        original_manifest = json.load(f)
    require(original_manifest["original_source_sha256"]["sc2_constants.py"]["sha256"]
            == source["hashes"]["sc2_constants.py"], "Script-81 constants provenance mismatch")
    require(original_manifest["original_source_sha256"]["sc2_counters.py"]["sha256"]
            == source["hashes"]["sc2_counters.py"], "Script-81 counter provenance mismatch")
    comparison, rows = evaluate(inventories, residuals, source["definitions"])
    result = {
        "schema": "ml_sc2.b3.counter_orientation_hypothesis.v1",
        "evidence_scope": "three historically inspected replay snapshots, not independent pristine test",
        "hypotheses": {
            "as_archived": "Own unit counters list contains an enemy unit type",
            "counter_relation_reversed": "Own unit appears in the enemy unit's counters list",
        },
        "source_hashes": source["hashes"],
        "source_zip_sha256": file_hash(args.historical_source_zip),
        "forensic_input_sha256": {
            "summary": file_hash(forensic / "summary.json"),
            "raw_player_inventories_and_upgrades": file_hash(
                forensic / "raw_player_inventories_and_upgrades.csv"),
            "per_snapshot_primitive_residual": file_hash(residual_path),
        },
        "replayed_raw_files": original_manifest["selected_raw_sha256"],
        "snapshot_count": len(inventories),
        "player_snapshot_count": len(inventories) * 2,
        "archived_counter_formula_reproduces_saved_script81_predictions": True,
        "combat_scored_only_where_captured_inventories_recreate_archived_combat": True,
        "by_primitive_family": comparison,
        "historical_original_generator_attested": False,
        "counter_formula_proven_from_original_build": False,
        "release_gate_b3_closed": False,
        "interpretation_warning": (
            "Reverse-counter is a controlled semantic hypothesis only; "
            "better test fit does not certify actual historical executable or source version."
        ),
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (out / "per_player_snapshot_hypotheses.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
