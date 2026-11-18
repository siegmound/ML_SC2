#!/usr/bin/env python3
"""S8: offline 25/50/75/90%-of-game accuracy curves from frozen GRU predictions.

No new fitting, no test partition, no future inputs to the GRU.
By default compare all three S4/S7/S6 data variants after S7 completes.
Use --arms v31_original64 for an immediate v3.1-only analysis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sc2proj.sequential.baselines import endpoint_frame
from sc2proj.sequential.comparison import VALIDATION_ENDPOINTS, load_verified_gru
from sc2proj.sequential.data import load_index, load_replay_partitions
from sc2proj.sequential.progress import (
    ARMS, CHECKPOINTS, join_checked_predictions, score_progress,
    select_exact_checkpoints,
)
from sc2proj.sequential.variants import (
    V31_SHA256, combatfix_common64_view, verify_combatfix_parity,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/sequential"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", choices=(42, 43, 44), default=[42, 43, 44])
    ap.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS),
                    help="Choose v31_original64 alone to evaluate S4 before S7 is complete")
    ap.add_argument("--bootstrap-reps", type=int, default=1000)
    ap.add_argument("--bootstrap-seed", type=int, default=2026)
    ap.add_argument("--reference-index-dir", type=Path,
                    default=RESULTS / "v3_1_fixed_w9_s1")
    ap.add_argument("--combatfix-index-dir", type=Path,
                    default=RESULTS / "v3_2_combatfix_w9_s1")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Fresh directory under results/sequential; never overwrite")
    args = ap.parse_args()
    if args.bootstrap_reps < 0 or len(args.seeds) != len(set(args.seeds)) or len(
        args.arms
    ) != len(set(args.arms)):
        ap.error("Negative bootstrap reps or duplicate seeds/arms")
    output = None
    if args.output_dir is not None:
        output = args.output_dir.resolve()
        base = RESULTS.resolve()
        if output == base or not output.is_relative_to(base) or output.exists():
            ap.error("Output must be a fresh subdirectory of results/sequential")

    ref = load_index(args.reference_index_dir)
    if (ref.manifest.get("source_zip_sha256") != V31_SHA256 or
            int(ref.manifest["window_length"]) != 9 or
            int(ref.manifest["stride"]) != 1 or len(ref.features) != 64 or
            ref.manifest.get("subset_max_replays") is not None):
        ap.error("S8 requires complete frozen 9-snapshot v3.1 S1 index")
    combatfix = None
    common = None
    parity = None
    if any(arm != "v31_original64" for arm in args.arms):
        combatfix = load_index(args.combatfix_index_dir)
        parity = verify_combatfix_parity(ref, combatfix, require_full=True)
        common = combatfix_common64_view(ref, combatfix)

    records = []
    coverage_rows = []
    checkpoint_rows = []
    per_seed = []
    configs = {}
    for seed in sorted(args.seeds):
        split_file = ROOT / f"data/processed/splits/real_v3_1_fixed_split_seed_{seed}.json"
        partition = load_replay_partitions(split_file, ref)
        if len(partition.window_positions["val"]) != VALIDATION_ENDPOINTS[seed]:
            raise ValueError(f"S8 seed {seed}: wrong frozen validation window count")
        if combatfix is not None:
            p32 = load_replay_partitions(split_file, combatfix)
            for part in ("train", "val", "test"):
                if not np.array_equal(partition.row_masks[part], p32.row_masks[part]) or not np.array_equal(
                    partition.window_positions[part], p32.window_positions[part]
                ):
                    raise ValueError(f"S8 seed {seed}: {part} split mismatch")
        all_endpoints = endpoint_frame(ref, partition, "val")
        checkpoints, coverage = select_exact_checkpoints(ref, partition)
        predictions = {}
        paths = {
            "v31_original64": RESULTS / f"s4_gru_w9_seed{seed}_v1",
            "v32_common64": RESULTS / f"s7_combatfix_common64_gru_w9_seed{seed}_v1",
            "v32_full73": RESULTS / f"s6_combatfix_gru_w9_seed{seed}_v1",
        }
        for arm in args.arms:
            if arm == "v31_original64":
                index, dataset, status = ref, "v3_1_fixed", "S4_VALIDATION_PASS"
            elif arm == "v32_common64":
                index, dataset, status = common, "v3_2_combatfix_common64", "S7_VALIDATION_PASS"
            else:
                index, dataset, status = combatfix, "v3_2_combatfix", "S6_VALIDATION_PASS"
            summary = json.loads((paths[arm] / "summary.json").read_text(encoding="utf-8"))
            config = json.dumps(summary.get("config"), sort_keys=True)
            if config == "null" or (configs and config != next(iter(configs.values()))):
                raise ValueError(f"S8 inconsistent GRU training configuration: {arm} seed={seed}")
            configs[f"{seed}:{arm}"] = config
            if arm == "v32_common64" and (
                summary.get("feature_view") != "v3_2_common64" or
                summary.get("feature_names") != list(ref.features) or
                summary.get("reference_zip_sha256") != ref.manifest["source_zip_sha256"]
            ):
                raise ValueError("S8 common64 provenance mismatch")
            predictions[arm] = load_verified_gru(
                paths[arm], seed, index, all_endpoints,
                dataset_variant=dataset, expected_status=status,
            )
        matched = join_checked_predictions(checkpoints, predictions)
        metrics, gains = score_progress(
            matched, arms=tuple(args.arms), checkpoints=CHECKPOINTS,
            reps=args.bootstrap_reps, rng_seed=args.bootstrap_seed + seed,
        )
        per_seed.append({"seed": seed, "coverage": coverage,
                         "checkpoint_metrics": metrics, "progression": gains})
        coverage_rows.append({"seed": seed, **{
            k: v for k, v in coverage.items() if not isinstance(v, (dict, list))
        }, **{f"raw_eligible_{p}": coverage["raw_eligible_replays_by_checkpoint"][str(p)]
             for p in CHECKPOINTS}})
        for progress in CHECKPOINTS:
            for arm in args.arms:
                checkpoint_rows.append({
                    "seed": seed, "progress_percent": progress,
                    "arm": arm, "replays": metrics[str(progress)]["replays"],
                    **metrics[str(progress)]["models"][arm],
                })
        records.append(matched.assign(seed=seed))
        print(
            f"[S8] seed={seed} full_val_replays={coverage['validation_replays_total']:,} "
            f"balanced_replays={coverage['balanced_replays']:,}: PASS",
            flush=True,
        )
    mean_checkpoint_metrics = {
        arm: {
            str(p): {
                metric: float(np.mean([
                    entry["checkpoint_metrics"][str(p)]["models"][arm][metric]
                    for entry in per_seed
                ]))
                for metric in ("accuracy", "balanced_accuracy", "roc_auc",
                               "log_loss", "brier_score")
            }
            for p in CHECKPOINTS
        }
        for arm in args.arms
    }
    report = {
        "schema": "sc2.sequential.progress_curve.v1",
        "status": "S8_VALIDATION_PROGRESS_PASS",
        "data_partition": "val",
        "source_zip_sha256": ref.manifest["source_zip_sha256"],
        "source_variants_parity": parity,
        "model_arms": args.arms,
        "seeds": sorted(args.seeds),
        "checkpoints_percent": list(CHECKPOINTS),
        "checkpoint_selection": (
            "ceil(percent * total_source_replay_snapshots / 100) exact endpoint; "
            "no nearest future/backward substitution"
        ),
        "population": "same complete-case replay cohort at every checkpoint within seed",
        "model_input": "only previous 9 snapshots at selected endpoint; no future or game duration",
        "bootstrap": {
            "unit": "replay_id", "reps": args.bootstrap_reps, "seed": args.bootstrap_seed,
            "interpretation": "conditional, retrospective and exploratory",
        },
        "per_seed": per_seed,
        "unweighted_seed_mean_curve": mean_checkpoint_metrics,
        "caveats": [
            "Final replay snapshot count is used only by offline evaluator to choose checkpoints; not deployable online schedule.",
            "Complete-case cohort excludes replay(s) with any missing nine-snapshot checkpoint and may bias results toward longer/regular games.",
            "S4/S6/S7 validation checkpoint selection reused validation; not a fresh prospective performance estimate.",
            "Seed splits reuse one corpus; their aggregate is not three independent external replications.",
            "This evaluates already-trained fixed-9 GRUs; it is NOT an experiment with an expanding sequence prefix.",
        ],
    }
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
        (output / "summary.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        pd.DataFrame(coverage_rows).to_csv(output / "cohort_coverage.csv", index=False)
        pd.DataFrame(checkpoint_rows).to_csv(output / "checkpoint_metrics.csv", index=False)
        pd.concat(records, ignore_index=True).to_csv(
            output / "matched_checkpoint_predictions.csv", index=False
        )
        print(f"Saved S8 validation progress curves to {output}", flush=True)
    print(json.dumps({
        "status": report["status"], "arms": args.arms, "seeds": sorted(args.seeds),
        "cohorts": coverage_rows, "unweighted_seed_mean_curve": mean_checkpoint_metrics,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
