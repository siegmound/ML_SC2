#!/usr/bin/env python3
"""S7: three-arm GRU ablation on exactly matched validation endpoints.

A: v3.1 64 original, B: v3.2 corrected common64, C: v3.2 full73.
Contrasts: B-A (corrections), C-B (nine added features), C-A (total).
No models are trained and no test predictions are accessed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sc2proj.sequential.baselines import endpoint_frame
from sc2proj.sequential.comparison import (
    METRICS, VALIDATION_ENDPOINTS, load_verified_gru, model_metrics,
    paired_replay_bootstrap,
)
from sc2proj.sequential.data import load_index, load_replay_partitions
from sc2proj.sequential.variants import verify_combatfix_parity, combatfix_common64_view

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/sequential"
ARMS = ("v31_original64", "v32_common64", "v32_full73")
CONTRASTS = (
    ("corrections", "v32_common64", "v31_original64"),
    ("nine_added", "v32_full73", "v32_common64"),
    ("total_variant", "v32_full73", "v31_original64"),
)


def compare_seed(seed: int, reference, combatfix, view, reps: int, rng_seed: int) -> dict:
    split_path = ROOT / f"data/processed/splits/real_v3_1_fixed_split_seed_{seed}.json"
    p31 = load_replay_partitions(split_path, reference)
    p32 = load_replay_partitions(split_path, combatfix)
    if p31.seed != p32.seed or p31.seed != seed:
        raise ValueError("S7 seed/split mismatch")
    for part in ("train", "val", "test"):
        if not np.array_equal(p31.window_positions[part], p32.window_positions[part]):
            raise ValueError(f"S7 {part} windows differ")
        if not np.array_equal(p31.row_masks[part], p32.row_masks[part]):
            raise ValueError(f"S7 {part} rows differ")
    endpoints = endpoint_frame(reference, p31, "val")
    if len(endpoints) != VALIDATION_ENDPOINTS[seed]:
        raise ValueError(f"S7 seed {seed}: unexpected validation endpoint count")
    dirs = {
        "v31_original64": RESULTS / f"s4_gru_w9_seed{seed}_v1",
        "v32_common64": RESULTS / f"s7_combatfix_common64_gru_w9_seed{seed}_v1",
        "v32_full73": RESULTS / f"s6_combatfix_gru_w9_seed{seed}_v1",
    }
    pred = {
        "v31_original64": load_verified_gru(dirs["v31_original64"], seed, reference, endpoints),
        "v32_common64": load_verified_gru(
            dirs["v32_common64"], seed, view, endpoints,
            dataset_variant="v3_2_combatfix_common64",
            expected_status="S7_VALIDATION_PASS"),
        "v32_full73": load_verified_gru(
            dirs["v32_full73"], seed, combatfix, endpoints,
            dataset_variant="v3_2_combatfix",
            expected_status="S6_VALIDATION_PASS"),
    }
    summaries = {arm: json.loads((path / "summary.json").read_text(encoding="utf-8"))
                 for arm, path in dirs.items()}
    reference_config = json.dumps(summaries["v31_original64"]["config"], sort_keys=True)
    if any(json.dumps(summaries[arm]["config"], sort_keys=True) != reference_config
           for arm in ARMS):
        raise ValueError("S7 GRU training hyperparameters must match across all three arms")
    common = summaries["v32_common64"]
    if (common.get("feature_view") != "v3_2_common64" or
            common.get("feature_names") != list(reference.features) or
            common.get("reference_zip_sha256") != reference.manifest["source_zip_sha256"]):
        raise ValueError("S7 common64 feature/view provenance mismatch")

    y = endpoints.p1_wins.to_numpy(dtype=np.int8)
    estimates = {}
    for arm in ARMS:
        scores = pred[arm]
        if (not np.array_equal(scores.y_true.to_numpy(dtype=np.int8), y) or
                not np.array_equal(scores.replay_id.to_numpy(dtype=str),
                                   endpoints.replay_id.to_numpy(dtype=str)) or
                not np.array_equal(scores.time_sec.to_numpy(dtype=np.int64),
                                   endpoints.time_sec.to_numpy(dtype=np.int64))):
            raise ValueError(f"S7 {arm}: inconsistent matched endpoint rows")
        estimates[arm] = model_metrics(
            y, scores.y_prob.to_numpy(dtype=np.float64),
            scores.y_pred.to_numpy(dtype=np.int8))
    contrasts = {}
    for i, (label, final, initial) in enumerate(CONTRASTS):
        difference = {
            key: float(estimates[final][key] - estimates[initial][key])
            for key in METRICS
        }
        paired = endpoints.copy()
        paired["gru_prob"] = pred[final].y_prob.to_numpy(dtype=np.float64)
        paired["gru_pred"] = pred[final].y_pred.to_numpy(dtype=np.int8)
        paired["v31_prob"] = pred[initial].y_prob.to_numpy(dtype=np.float64)
        paired["v31_pred"] = pred[initial].y_pred.to_numpy(dtype=np.int8)
        interval = paired_replay_bootstrap(paired, "v31", reps=reps,
                                           seed=rng_seed + 13 * seed + i)
        if (not np.isclose(interval["accuracy_delta"], difference["accuracy"], atol=1e-12, rtol=0)
                or not np.isclose(interval["log_loss_delta"], difference["log_loss"],
                                   atol=1e-9, rtol=0)):
            raise ValueError(f"S7 {label}: inconsistent paired delta")
        contrasts[label] = {
            "final_arm": final, "initial_arm": initial,
            "deltas": difference, "paired_replay_bootstrap": interval,
        }
    return {
        "seed": seed, "n_endpoints": len(endpoints),
        "n_replays": int(endpoints.replay_id.nunique()),
        "metrics": estimates, "contrasts": contrasts,
        "checkpoint_epochs": {arm: summaries[arm]["selected_epoch"] for arm in ARMS},
        "config": summaries["v31_original64"]["config"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", choices=(42, 43, 44), default=[42, 43, 44])
    ap.add_argument("--bootstrap-reps", type=int, default=1000)
    ap.add_argument("--bootstrap-seed", type=int, default=2026)
    ap.add_argument("--reference-index-dir", type=Path, default=RESULTS / "v3_1_fixed_w9_s1")
    ap.add_argument("--index-dir", type=Path, default=RESULTS / "v3_2_combatfix_w9_s1")
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()
    if args.bootstrap_reps < 0 or len(args.seeds) != len(set(args.seeds)):
        ap.error("Negative bootstrap reps or duplicate seeds")
    output = None
    if args.output_dir is not None:
        output = args.output_dir.resolve()
        base = RESULTS.resolve()
        if output == base or not output.is_relative_to(base) or output.exists():
            ap.error("Output directory must be fresh and under results/sequential/")
    reference = load_index(args.reference_index_dir)
    combatfix = load_index(args.index_dir)
    parity = verify_combatfix_parity(reference, combatfix, require_full=True)
    view = combatfix_common64_view(reference, combatfix)
    per_seed = [
        compare_seed(seed, reference, combatfix, view,
                     reps=args.bootstrap_reps, rng_seed=args.bootstrap_seed)
        for seed in sorted(args.seeds)
    ]
    mean_contrasts = {
        label: {
            metric: float(np.mean([entry["contrasts"][label]["deltas"][metric]
                                   for entry in per_seed]))
            for metric in METRICS
        }
        for label, _, _ in CONTRASTS
    }
    report = {
        "schema": "sc2.sequential.common64_ablation.v1",
        "status": "S7_VALIDATION_ABLATION_PASS",
        "partition": "val", "seeds": sorted(args.seeds), "parity": parity,
        "experiment_arms": list(ARMS), "per_seed": per_seed,
        "mean_unweighted_contrast_deltas": mean_contrasts,
        "bootstrap": {"unit": "replay_id", "reps": args.bootstrap_reps,
                      "base_seed": args.bootstrap_seed},
        "caveat": (
            "Exploratory validation-only ablation. v3.2-common64 vs v3.1-original64 tests corrected "
            "values in common predictor names; v3.2-full73 vs v3.2-common64 tests added predictor "
            "channels with increased input-parameter count. Replay bootstrap intervals are "
            "conditional; seeds reuse one replay corpus. Validation selected all checkpoints "
            "and historical test was previously inspected. No test accessed in S7."
        ),
    }
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
        (output / "summary.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        pd.DataFrame([{
            "seed": entry["seed"], "comparison": label,
            "n_endpoints": entry["n_endpoints"],
            **{f"delta_{m}": entry["contrasts"][label]["deltas"][m] for m in METRICS},
            "accuracy_ci_low": entry["contrasts"][label]["paired_replay_bootstrap"]["accuracy_delta_ci95"][0]
                if args.bootstrap_reps else None,
            "accuracy_ci_high": entry["contrasts"][label]["paired_replay_bootstrap"]["accuracy_delta_ci95"][1]
                if args.bootstrap_reps else None,
            "log_loss_ci_low": entry["contrasts"][label]["paired_replay_bootstrap"]["log_loss_delta_ci95"][0]
                if args.bootstrap_reps else None,
            "log_loss_ci_high": entry["contrasts"][label]["paired_replay_bootstrap"]["log_loss_delta_ci95"][1]
                if args.bootstrap_reps else None,
        } for entry in per_seed for label, _, _ in CONTRASTS]).to_csv(
            output / "paired_ablation_deltas.csv", index=False)
        print(f"Saved S7 ablation report to {output}", flush=True)
    for item in per_seed:
        print(f"[S7] seed={item['seed']} endpoints={item['n_endpoints']:,} "
              f"replays={item['n_replays']:,}: PASS", flush=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
