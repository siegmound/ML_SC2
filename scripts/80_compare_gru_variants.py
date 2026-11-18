#!/usr/bin/env python3
"""S6: validation-only paired comparison of fixed GRU on v3.1 vs combatfix.

Retrospective; no test predictions/scores are read, no model is fitted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sc2proj.sequential.baselines import endpoint_frame
from sc2proj.sequential.comparison import (
    METRICS, VALIDATION_ENDPOINTS, load_verified_gru,
    model_metrics, paired_replay_bootstrap,
)
from sc2proj.sequential.data import load_index, load_replay_partitions
from sc2proj.sequential.variants import verify_combatfix_parity

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/sequential"


def compare(seed: int, reference, candidate, reps: int, bootstrap_seed: int) -> dict:
    split_file = ROOT / f"data/processed/splits/real_v3_1_fixed_split_seed_{seed}.json"
    split_v31 = load_replay_partitions(split_file, reference)
    split_v32 = load_replay_partitions(split_file, candidate)
    for part in ("train", "val", "test"):
        if not np.array_equal(split_v31.window_positions[part], split_v32.window_positions[part]):
            raise ValueError(f"S6 seed {seed}: {part} window split mismatch")
    endpoints = endpoint_frame(reference, split_v31, "val")
    if len(endpoints) != VALIDATION_ENDPOINTS[seed]:
        raise ValueError(f"S6 seed {seed}: unexpected validation endpoint count")
    run31 = RESULTS / f"s4_gru_w9_seed{seed}_v1"
    run32 = RESULTS / f"s6_combatfix_gru_w9_seed{seed}_v1"
    p31 = load_verified_gru(run31, seed, reference, endpoints)
    p32 = load_verified_gru(run32, seed, candidate, endpoints,
                            dataset_variant="v3_2_combatfix",
                            expected_status="S6_VALIDATION_PASS")
    meta31 = json.loads((run31 / "summary.json").read_text(encoding="utf-8"))
    meta32 = json.loads((run32 / "summary.json").read_text(encoding="utf-8"))
    if json.dumps(meta31["config"], sort_keys=True) != json.dumps(meta32["config"], sort_keys=True):
        raise ValueError("S6 and S4 hyperparameter configurations differ")
    if not (np.array_equal(p31["replay_id"].to_numpy(), p32["replay_id"].to_numpy()) and
            np.array_equal(p31["time_sec"].to_numpy(), p32["time_sec"].to_numpy()) and
            np.array_equal(p31["y_true"].to_numpy(), p32["y_true"].to_numpy())):
        raise ValueError("S6 GRU variants do not share exact validation labels and endpoints")
    paired = endpoints.copy()
    paired["v31_prob"] = p31.y_prob.to_numpy(dtype=np.float64)
    paired["v31_pred"] = p31.y_pred.to_numpy(dtype=np.int8)
    paired["gru_prob"] = p32.y_prob.to_numpy(dtype=np.float64)
    paired["gru_pred"] = p32.y_pred.to_numpy(dtype=np.int8)
    y = paired.p1_wins.to_numpy(dtype=np.int8)
    measures = {}
    for name in ("v31", "gru"):
        measures[name] = model_metrics(
            y, paired[f"{name}_prob"].to_numpy(dtype=np.float64),
            paired[f"{name}_pred"].to_numpy(dtype=np.int8))
    delta = {
        metric: float(measures["gru"][metric] - measures["v31"][metric])
        for metric in METRICS
    }
    ci = paired_replay_bootstrap(paired, "v31", reps=reps, seed=bootstrap_seed + seed)
    if not (np.isclose(ci["accuracy_delta"], delta["accuracy"], atol=1e-12, rtol=0)
            and np.isclose(ci["log_loss_delta"], delta["log_loss"], atol=1e-9, rtol=0)):
        raise ValueError("S6 paired bootstrap estimates disagree with metrics")
    return {
        "seed": seed, "endpoints": len(paired),
        "replays": int(paired.replay_id.nunique()),
        "v3_1": measures["v31"], "v3_2_combatfix": measures["gru"],
        "delta_v3_2_minus_v3_1": delta, "paired_bootstrap": ci,
        "v3_1_selected_epoch": meta31["selected_epoch"],
        "v3_2_selected_epoch": meta32["selected_epoch"],
        "config": meta31["config"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", nargs="+", type=int, choices=(42, 43, 44), default=[42, 43, 44])
    ap.add_argument("--bootstrap-reps", type=int, default=1000)
    ap.add_argument("--bootstrap-seed", type=int, default=2026)
    ap.add_argument("--reference-index-dir", type=Path, default=RESULTS / "v3_1_fixed_w9_s1")
    ap.add_argument("--index-dir", type=Path, default=RESULTS / "v3_2_combatfix_w9_s1")
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()
    if args.bootstrap_reps < 0 or len(args.seeds) != len(set(args.seeds)):
        ap.error("Invalid bootstrap reps or repeated seeds")
    output = None
    if args.output_dir is not None:
        output = args.output_dir.resolve()
        base = RESULTS.resolve()
        if output == base or not output.is_relative_to(base) or output.exists():
            ap.error("Output must be fresh and within results/sequential/")
    reference = load_index(args.reference_index_dir)
    candidate = load_index(args.index_dir)
    parity = verify_combatfix_parity(reference, candidate)
    per_seed = [
        compare(seed, reference, candidate, args.bootstrap_reps, args.bootstrap_seed)
        for seed in sorted(args.seeds)
    ]
    means = {
        metric: float(np.mean([record["delta_v3_2_minus_v3_1"][metric] for record in per_seed]))
        for metric in METRICS
    }
    report = {
        "status": "S6_VARIANT_COMPARISON_PASS", "schema": "sc2.sequential.combatfix_gru_v1",
        "partition": "val", "seeds": sorted(args.seeds), "parity": parity,
        "bootstrap": {"unit": "whole_replay", "reps": args.bootstrap_reps,
                      "base_seed": args.bootstrap_seed},
        "per_seed": per_seed,
        "unweighted_mean_delta_v3_2_minus_v3_1": means,
        "interpretation": (
            "Retrospective validation-only exploratory experiment; early stopping reused validation. "
            "No new untouched holdout; seeds are dependent splits of the same corpus. "
            "Both corrected feature values and nine extra features changed. "
            "Paired bootstrap CI is conditional on the chosen models/splits; no test was read."),
    }
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
        (output / "summary.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        pd.DataFrame([{
            "seed": x["seed"], "endpoints": x["endpoints"], "replays": x["replays"],
            **{f"v31_{metric}": x["v3_1"][metric] for metric in METRICS},
            **{f"v32_{metric}": x["v3_2_combatfix"][metric] for metric in METRICS},
            **{f"delta_{metric}": x["delta_v3_2_minus_v3_1"][metric] for metric in METRICS},
            "accuracy_ci_low": x["paired_bootstrap"]["accuracy_delta_ci95"][0]
                if x["paired_bootstrap"]["accuracy_delta_ci95"] else None,
            "accuracy_ci_high": x["paired_bootstrap"]["accuracy_delta_ci95"][1]
                if x["paired_bootstrap"]["accuracy_delta_ci95"] else None,
            "log_loss_ci_low": x["paired_bootstrap"]["log_loss_delta_ci95"][0]
                if x["paired_bootstrap"]["log_loss_delta_ci95"] else None,
            "log_loss_ci_high": x["paired_bootstrap"]["log_loss_delta_ci95"][1]
                if x["paired_bootstrap"]["log_loss_delta_ci95"] else None,
        } for x in per_seed]).to_csv(output / "per_seed_comparison.csv", index=False)
        print(f"Saved S6 audit report to {output}", flush=True)
    for row in per_seed:
        print(f"[S6] seed={row['seed']} endpoints={row['endpoints']:,} "
              f"delta_accuracy={row['delta_v3_2_minus_v3_1']['accuracy']:+.5f} "
              f"delta_log_loss={row['delta_v3_2_minus_v3_1']['log_loss']:+.5f}", flush=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
