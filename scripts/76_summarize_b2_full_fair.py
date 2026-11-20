#!/usr/bin/env python3
"""Read-only B2 scientific aggregation; writes only NEW analysis artifacts.

Run the strict nine-run verifier before reading metrics. This is a retrospective
three-seed descriptive analysis on historically inspected test replay groups,
not a new blind benchmark, hypothesis test or model-selection procedure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

SEEDS = (42, 43, 44)
MODELS = ("rf", "xgb", "deep")
CALIBRATIONS = ("uncalibrated", "platt", "isotonic")
METRICS = ("accuracy", "balanced_accuracy", "roc_auc", "log_loss", "brier", "ece_15", "mce_15")
METRIC_TITLES = ("Accuracy", "Balanced accuracy", "ROC-AUC", "Log loss", "Brier", "ECE (15)", "MCE (15)")
ROW_COLUMNS = ("model", "variant", "seed", "protocol", "n_rows_test", "n_features", "best_candidate", *METRICS)


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"Missing verified B2 report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def finite_metric(value, key: str, where: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{where}: missing/invalid {key}") from error
    if not math.isfinite(result):
        raise ValueError(f"{where}: nonfinite {key}")
    if key in ("accuracy", "balanced_accuracy", "roc_auc", "brier", "ece_15", "mce_15") and not 0 <= result <= 1:
        raise ValueError(f"{where}: out-of-range {key}: {result}")
    if key == "log_loss" and result < 0:
        raise ValueError(f"{where}: negative log_loss")
    return result


def collect(output_base: Path) -> tuple[list[dict], list[dict]]:
    """Collect only immutable experiment reports, never inspect test to select a winner."""
    rows = []
    inputs = []
    for seed in SEEDS:
        for model in MODELS:
            if model == "deep":
                path = output_base / "deep_finalist/real_v3_1_fixed" / f"seed_{seed}" / "metrics_summary.json"
            else:
                path = output_base / "strict_calibration/real_v3_1_fixed" / model / f"seed_{seed}" / "metrics_summary.json"
            report = read_json(path)
            if int(report.get("seed", -1)) != seed:
                raise ValueError(f"Seed mismatch: {path}")
            if report.get("model") != ("deep_final" if model == "deep" else model):
                raise ValueError(f"Model mismatch: {path}")
            inputs.append({
                "path": str(path.relative_to(output_base)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
            if model == "deep":
                variants = (("selected_candidate_only", report["test_metrics"]),)
                protocol = "train_validation_test"
                n_test = int(report["n_rows_test"])
                best = str(report["best_candidate"])
            else:
                variants = tuple((method, report["scores"][method]) for method in CALIBRATIONS)
                protocol = "four_way_replay_disjoint"
                n_test = int(report["partition_rows"]["test"])
                best = ""
            if n_test <= 0:
                raise ValueError(f"Invalid test count: {path}")
            for variant, values in variants:
                row = {
                    "model": model, "variant": variant, "seed": seed,
                    "protocol": protocol, "n_rows_test": n_test,
                    "n_features": int(report["n_features"]), "best_candidate": best,
                }
                for metric in METRICS:
                    row[metric] = finite_metric(values[metric], metric, str(path)) if metric in values else None
                    if model != "deep" and row[metric] is None:
                        raise ValueError(f"Missing calibration metric {metric}: {path}")
                    if model == "deep" and row[metric] is None and metric not in ("brier", "ece_15", "mce_15"):
                        raise ValueError(f"Missing Deep test metric {metric}: {path}")
                rows.append(row)
    if len(inputs) != 9 or len(rows) != 21:
        raise ValueError("Expected exactly nine reports and 21 seed/variant rows")
    return rows, inputs


def aggregate(rows: list[dict]) -> list[dict]:
    groups = [(model, variant) for model in MODELS
              for variant in (("selected_candidate_only",) if model == "deep" else CALIBRATIONS)]
    results = []
    for model, variant in groups:
        subset = sorted((r for r in rows if r["model"] == model and r["variant"] == variant),
                        key=lambda item: item["seed"])
        if [r["seed"] for r in subset] != list(SEEDS):
            raise ValueError(f"Missing/duplicated seeds for {model}/{variant}")
        result = {"model": model, "variant": variant, "n_seeds": len(SEEDS)}
        for metric in METRICS:
            values = [r[metric] for r in subset]
            if all(value is None for value in values):
                result[f"{metric}_mean"] = None
                result[f"{metric}_sd"] = None
            elif any(value is None for value in values):
                raise ValueError(f"Incomplete {metric}: {model}/{variant}")
            else:
                result[f"{metric}_mean"] = statistics.mean(values)
                result[f"{metric}_sd"] = statistics.stdev(values)  # sample SD (n-1), not SEM
        results.append(result)
    return results


def fmt(mean, sd):
    return "—" if mean is None else f"{mean:.6f} ± {sd:.6f}"


def render_markdown(summary: list[dict]) -> str:
    header = ["Model", "Variant", *METRIC_TITLES]
    lines = [
        "# B2 FAIR full-scale — three-seed descriptive results",
        "",
        "**Evidence policy:** all nine runs must pass the independent B2 verifier before this report is produced.",
        "**Test status:** original replay-level holdouts had been inspected historically; these are retrospective tests, not pristine blind evaluation.",
        "**Protocols:** RF/XGB use disjoint train/tune/calibration/test groups; Deep uses train/validation/test, selecting one candidate by validation before test inference.",
        "**Uncertainty:** arithmetic mean ± sample standard deviation across seeds 42, 43 and 44 (n=3), not a confidence interval or significance test.",
        "**Calibration:** RF/XGB variants are all reported; no calibration method is selected by its test score. Deep has no saved Brier/ECE/MCE in its run summary, hence those cells remain unavailable.",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in summary:
        cells = [row["model"], row["variant"]] + [
            fmt(row[f"{metric}_mean"], row[f"{metric}_sd"]) for metric in METRICS
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "Accuracy, balanced accuracy and ROC-AUC: higher is better. Log loss, Brier, ECE and MCE: lower is better.",
        "The RF no_counter and XGB full feature profiles differ. Comparisons are descriptive and cannot establish statistically significant superiority from only three correlated seed splits.",
        "",
    ]
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict], columns) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row[key] for key in columns})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-base", type=Path, default=Path("results/b2_fullscale_audit_20261119"))
    parser.add_argument("--analysis-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output_base = (root / args.output_base).resolve()
    destination = (args.analysis_dir or (output_base / "analysis")).resolve()
    if destination != output_base / "analysis":
        raise ValueError("Analysis may only be written to the isolated output-base/analysis directory")
    verifier = root / "scripts/75_verify_b2_full_fair.py"
    if not verifier.is_file():
        raise FileNotFoundError(f"Missing B2 verifier: {verifier}")
    subprocess.run([
        sys.executable, str(verifier), "--repo-root", str(root),
        "--output-base", str(output_base),
    ], check=True)
    rows, inputs = collect(output_base)
    summaries = aggregate(rows)
    markdown = render_markdown(summaries)
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / "per_seed_metrics.csv", rows, ROW_COLUMNS)
    columns = ("model", "variant", "n_seeds", *(
        f"{metric}_{suffix}" for metric in METRICS for suffix in ("mean", "sd")
    ))
    write_csv(destination / "three_seed_summary.csv", summaries, columns)
    (destination / "three_seed_summary.md").write_text(markdown, encoding="utf-8")
    manifest = {
        "scope": "B2 full-scale real_v3_1_fixed; historically inspected test replays",
        "verifier": "PASS: all nine experiments, original replay identities and metrics",
        "seeds": list(SEEDS),
        "n_experiments": 9,
        "n_seed_variant_rows": len(rows),
        "sample_sd_ddof": 1,
        "model_selection_from_test": False,
        "input_summary_sha256": inputs,
        "files": ["per_seed_metrics.csv", "three_seed_summary.csv", "three_seed_summary.md"],
    }
    (destination / "evidence_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"PASS: B2 summary for 9 verified experiments, 7 model/variant groups -> {destination}")
    print(markdown)


if __name__ == "__main__":
    main()
