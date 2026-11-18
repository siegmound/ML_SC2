#!/usr/bin/env python3
"""S5: strict read-only multi-seed validation comparison, GRU vs S3 baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sc2proj.sequential.comparison import (
    BASELINES, EXPECTED_SEEDS, METRICS, MODELS,
    compare_one_seed, summarize_multiseed,
)
from sc2proj.sequential.data import load_index, load_replay_partitions

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index-dir", type=Path, default=RESULTS / "sequential/v3_1_fixed_w9_s1")
    ap.add_argument("--results-root", type=Path, default=RESULTS)
    ap.add_argument("--bootstrap-reps", type=int, default=1000,
                    help="Per-seed replay-level paired bootstrap draws (0 for point estimates only)")
    ap.add_argument("--bootstrap-seed", type=int, default=2026)
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Optional fresh directory under results/sequential/ for audit JSON and CSV")
    args = ap.parse_args()
    if args.bootstrap_reps < 0:
        ap.error("--bootstrap-reps cannot be negative")
    output = None
    if args.output_dir is not None:
        output = args.output_dir.resolve()
        base = (RESULTS / "sequential").resolve()
        if output == base or not output.is_relative_to(base) or output.exists():
            ap.error("--output-dir must be a new path under results/sequential/ (no overwrites)")
    index = load_index(args.index_dir)
    if index.manifest.get("window_length") != 9 or len(index.features) != 64:
        ap.error("S5 expects the frozen v3_1_fixed 9-window, 64-feature S1 index")
    per_seed = []
    rows = []
    delta_rows = []
    for seed in EXPECTED_SEEDS:
        split_path = ROOT / f"data/processed/splits/real_v3_1_fixed_split_seed_{seed}.json"
        gru_run_dir = RESULTS / f"sequential/s4_gru_w9_seed{seed}_v1"
        split = load_replay_partitions(split_path, index)
        seed_report, _ = compare_one_seed(
            index, split, args.results_root, gru_run_dir,
            reps=args.bootstrap_reps, bootstrap_seed=args.bootstrap_seed,
        )
        per_seed.append(seed_report)
        for model in MODELS:
            rows.append({"seed": seed, "model": model,
                         "endpoints": seed_report["endpoints"], **seed_report["models"][model]})
        for baseline in BASELINES:
            pair = seed_report["gru_minus_baseline"][baseline]
            boot = pair["paired_bootstrap"]
            delta_rows.append({
                "seed": seed, "baseline": baseline,
                **{f"delta_{metric}": pair["deltas_gru_minus_baseline"][metric] for metric in METRICS},
                "accuracy_ci_low": boot["accuracy_delta_ci95"][0] if boot["accuracy_delta_ci95"] else None,
                "accuracy_ci_high": boot["accuracy_delta_ci95"][1] if boot["accuracy_delta_ci95"] else None,
                "log_loss_ci_low": boot["log_loss_delta_ci95"][0] if boot["log_loss_delta_ci95"] else None,
                "log_loss_ci_high": boot["log_loss_delta_ci95"][1] if boot["log_loss_delta_ci95"] else None,
            })
        print(f"[S5] seed={seed} matched_val_endpoints={seed_report['endpoints']:,} "
              f"replays={seed_report['replays']:,}: PASS", flush=True)
    summary = summarize_multiseed(per_seed, reps=args.bootstrap_reps,
                                  bootstrap_seed=args.bootstrap_seed)
    # Only write after all three seeds have been validated.
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        pd.DataFrame(rows).to_csv(output / "per_seed_metrics.csv", index=False)
        pd.DataFrame(delta_rows).to_csv(output / "paired_deltas.csv", index=False)
        print(f"Saved read-only comparative report to {output}")
    print(json.dumps({
        "status": summary["status"], "seeds": summary["seeds"],
        "unweighted_seed_means": summary["unweighted_seed_means"],
        "mean_paired_differences": summary["mean_paired_differences"],
        "per_seed": [
            {"seed": entry["seed"], "endpoints": entry["endpoints"],
             "models": entry["models"], "gru_minus_baseline": entry["gru_minus_baseline"]}
            for entry in per_seed
        ],
        "interpretation": summary["interpretation"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
