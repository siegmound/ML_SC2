#!/usr/bin/env python3
"""B3 read-only primitive combat/counter forensic reconstruction.

Re-use exactly pinned original v3.1 source copies + existing seed-42 raw
replay membership evidence. Record source inventories and upgrades, then
compare frozen vs reconstructed combat/counter primitive feature values.
Never claim original executable attestation, guess missing unit metadata,
change historical source files or alter the frozen datasets.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRIMITIVES = (
    "p1_combat", "p2_combat", "counter_advantage_p1",
    "counter_advantage_p2", "p1_tech_level", "p2_tech_level",
)
EPSILON = 1e-6


def previous_auditor():
    spec = importlib.util.spec_from_file_location(
        "b3_pinned_historical_parser", ROOT / "scripts/79_compare_b3_historical_raw_parity.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def freeze_values(frame: pd.DataFrame, replay_id: str) -> dict:
    part = frame[frame.replay_id == replay_id]
    assert len(part) > 0
    return {int(row.time_sec): row for row in part.itertuples(index=False)}


def summarize_player(inventory: dict, pid: int, unit_stats: dict,
                     unit_counters: dict, upgrades: dict) -> dict:
    """Static player evidence extracted from actual replay inventory events."""
    army = {u: int(count) for u, count in inventory.get(pid, {}).items()
            if count > 0 and unit_stats.get(u, {}).get("type") == "army"}
    n_own = sum(army.values())
    raw_power = sum(unit_stats[u]["power"] * count for u, count in army.items())
    return {
        "army_units": n_own,
        "army_types": len(army),
        "army_names_and_counts": ";".join(f"{u}:{army[u]}" for u in sorted(army)),
        "raw_base_power_before_counters_upgrades": float(raw_power) / 100.0,
        "units_missing_tech_definition": sum(
            count for u, count in army.items() if "tech_level" not in unit_stats[u]
        ),
        "units_missing_upgrade_category": sum(
            count for u, count in army.items() if "upg_cat" not in unit_stats[u]
        ),
        "upgrade_events_completed": ";".join(sorted(str(x) for x in upgrades.get(pid, set()))),
        "counter_rules_for_army_types": sum(
            len(unit_counters.get(unit, ())) for unit in army
        ),
    }


def compare_primitives(frozen: pd.DataFrame, generated: pd.DataFrame,
                       *, atol: float = EPSILON, rtol: float = EPSILON
                       ) -> tuple[list[dict], dict]:
    """Compare aligned replay+timestamp rows and expose signed residuals."""
    required = set(PRIMITIVES) | {"replay_id", "time_sec", "p1_wins"}
    if not required.issubset(frozen) or not required.issubset(generated):
        raise ValueError("Missing primitive field(s) in frozen/reconstructed data")
    if frozen.duplicated(["replay_id", "time_sec"]).any():
        raise ValueError("Frozen duplicate replay/time")
    if generated.duplicated(["replay_id", "time_sec"]).any():
        raise ValueError("Reconstructed duplicate replay/time")
    merge = frozen.merge(generated, on=["replay_id", "time_sec"], how="outer",
                         suffixes=("_frozen", "_rebuilt"), indicator=True,
                         validate="one_to_one")
    if not (merge["_merge"] == "both").all():
        raise ValueError("Replay/time keys differ; inspect with script 79")
    if (merge.p1_wins_frozen != merge.p1_wins_rebuilt).any():
        raise ValueError("Historical replay winner differs from frozen")
    rows = []
    for field in PRIMITIVES:
        original = pd.to_numeric(merge[f"{field}_frozen"], errors="coerce").to_numpy(dtype=float)
        new = pd.to_numeric(merge[f"{field}_rebuilt"], errors="coerce").to_numpy(dtype=float)
        good = np.isclose(original, new, atol=atol, rtol=rtol, equal_nan=False)
        for i, (f, g, ok) in enumerate(zip(original, new, good)):
            rows.append({
                "replay_id": str(merge.iloc[i]["replay_id"]),
                "time_sec": int(merge.iloc[i]["time_sec"]),
                "feature": field,
                "frozen": float(f),
                "historical_reconstructed": float(g),
                "signed_frozen_minus_reconstructed": float(f-g),
                "absolute_error": float(abs(f-g)),
                "mismatch": not bool(ok),
            })
    summary = []
    for feature in PRIMITIVES:
        subset = [r for r in rows if r["feature"] == feature]
        errors = [r["absolute_error"] for r in subset]
        bad = [r for r in subset if r["mismatch"]]
        summary.append({
            "feature": feature,
            "comparisons": len(subset),
            "mismatches": len(bad),
            "mean_absolute_error": sum(errors) / len(errors) if errors else None,
            "max_absolute_error": max(errors) if errors else None,
            "first_mismatch": (
                f"{min(bad, key=lambda r: (r['replay_id'],r['time_sec']))['replay_id']}"
                f"@{min(bad, key=lambda r: (r['replay_id'],r['time_sec']))['time_sec']}"
                if bad else ""
            ),
        })
    return rows, {"by_primitive": summary,
                  "mismatches": sum(r["mismatch"] for r in rows),
                  "comparisons": len(rows)}


def writer(path: Path, rows: list[dict], headers: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        w = csv.DictWriter(stream, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--historical-source-zip", type=Path, required=True)
    p.add_argument("--raw-root", type=Path, default=Path("data/raw/replay_subsets"))
    p.add_argument("--inventory-csv", type=Path, default=Path(
        "results/b3_source_inventory_subset_20261119/identity_reconciliation.csv"))
    p.add_argument("--processed-zip", type=Path, default=Path(
        "data/processed/starcraft_full_dataset_v3_1_fixed.zip"))
    p.add_argument("--max-replays", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path(
        "results/b3_combat_counter_forensics_sample3_20261119"))
    args = p.parse_args()
    dest = args.output_dir.resolve()
    raw_root = args.raw_root.resolve()
    if dest.exists() or dest == raw_root or raw_root in dest.parents:
        raise ValueError("Refusing overwrite or output inside raw replay root")
    audit = previous_auditor()
    expected_sha = audit.EXPECTED_V31_ZIP_SHA256
    observed_sha = audit.sha256(args.processed_zip)
    if observed_sha != expected_sha:
        raise ValueError("Wrong frozen v3.1 ZIP hash")
    selected = audit.select_present_replays(
        args.inventory_csv, raw_root, args.max_replays, args.seed
    )
    frozen = audit.frozen_selected(args.processed_zip, {x[0] for x in selected})
    all_reconstructed, metadata, errors = [], [], []
    source_files = []
    with tempfile.TemporaryDirectory(prefix="b3_forensics_history_") as temp:
        folder = Path(temp)
        manifests = audit.load_historical_source(args.historical_source_zip, folder)
        bridge = audit.load_historical_bridge(folder)
        parser = bridge.parser
        original_create = parser._create_snapshot
        for name, path in selected:
            raw_sha = audit.sha256(path)
            source_files.append({"replay_id": name, "sha256": raw_sha})
            snapshots = []
            def traced_create(*positional, **kw):
                snap = original_create(*positional, **kw)
                if snap is not None and float(snap["time_sec"]) > 120:
                    inventories = kw["inventories"]
                    upgrades = kw["upgrades_completed"]
                    for label, pid in (("p1", kw["p1"]), ("p2", kw["p2"])):
                        item = summarize_player(
                            inventories, pid, parser.UNIT_STATS,
                            parser.UNIT_COUNTERS, upgrades,
                        )
                        snapshots.append({
                            "replay_id": name, "time_sec": snap["time_sec"],
                            "player": label, **item,
                        })
                return snap
            parser._create_snapshot = traced_create
            try:
                reconstructed = bridge.process_single_replay(str(path))
                if reconstructed is None or reconstructed.empty:
                    raise ValueError(f"Historical bridge returned no rows: {name}")
                reconstructed = reconstructed[reconstructed["time_sec"] > 120].copy()
                part = frozen[frozen.replay_id == name]
                audit.compare_replay(name, part, reconstructed, atol=EPSILON, rtol=EPSILON)
                all_reconstructed.append(reconstructed)
                metadata.extend(snapshots)
            finally:
                parser._create_snapshot = original_create
    actual = pd.concat(all_reconstructed, ignore_index=True)
    residuals, summary = compare_primitives(frozen, actual)
    # A direct source inventory is recorded as contextual evidence. It is not
    # proof that equal unit *counts* mean equal per-unit inventory composition.
    left_evidence = {(r["replay_id"], float(r["time_sec"]), r["player"])
                     for r in metadata}
    expected_evidence = len(frozen) * 2
    if len(left_evidence) != expected_evidence:
        raise ValueError("Instrumented player-level raw inventory coverage missing")
    manifest = {
        "schema": "ml_sc2.b3.combat_counter_primitive_forensics.v1",
        "frozen_sha256": observed_sha,
        "original_source_sha256": manifests,
        "historical_source_zip_sha256": audit.sha256(args.historical_source_zip),
        "selected_raw_sha256": source_files,
        "seed": args.seed, "n_replays": len(selected),
        "n_aligned_snapshots": len(frozen),
        "raw_player_snapshot_rows": len(metadata),
        "primitive_residuals": summary,
        "all_historical_unit_stats_have_no_tech_or_upg_cat": all(
            "tech_level" not in x and "upg_cat" not in x
            for x in parser.UNIT_STATS.values()
        ),
        "original_executable_attested": False,
        "actual_historical_original_counter_formula_attested": False,
        "root_causes_for_combat_counter_resolved": False,
        "note": ("No modifications to historical source copies; descriptive "
                 "signed errors only, not optimization or guessed replacement formulas."),
    }
    dest.mkdir(parents=True, exist_ok=False)
    (dest / "summary.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    writer(dest / "per_snapshot_primitive_residual.csv", residuals,
           list(residuals[0]))
    writer(dest / "raw_player_inventories_and_upgrades.csv", metadata,
           list(metadata[0]))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
