"""Derive and verify the published Block 16 summary from frozen per-seed metrics.

Reads saved aggregate/detail CSVs only; never trains models or rewrites experiments.
Population standard deviation (ddof=0) is the convention in Block 16.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import io
import math
import statistics
from pathlib import Path

SEEDS = {42, 43, 44}
GROUPS = (
    ("v3_1_fixed", "rf", "no_counter"),
    ("v3_1_fixed", "xgb", "full"),
    ("v3_2_combatfix", "rf", "no_counter"),
    ("v3_2_combatfix", "xgb", "full"),
)
METRICS = (
    ("accuracy", "acc"),
    ("balanced_accuracy", "balacc"),
    ("roc_auc", "auc"),
    ("log_loss", "logloss"),
)
CSV_COLUMNS = (
    "dataset", "model", "profile", "acc_mean", "acc_std",
    "balacc_mean", "balacc_std", "auc_mean", "auc_std",
    "logloss_mean", "logloss_std",
)


def derive_table(repo_root: Path) -> str:
    detailed_path = repo_root / "results/block16_final/block16_full_detailed.csv"
    summary_path = repo_root / "results/block16_final/block16_full_summary.csv"
    with detailed_path.open(newline="", encoding="utf-8") as handle:
        detailed = list(csv.DictReader(handle))
    with summary_path.open(newline="", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))

    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for run in detailed:
        key = (run["source_dataset"], run["model_family"], run["target_profile"])
        groups.setdefault(key, []).append(run)

    if set(groups) != set(GROUPS):
        raise ValueError(f"Unexpected model groups in per-seed results: {set(groups) ^ set(GROUPS)}")

    summary_map = {
        (row["source_dataset"], row["model_family"], row["profile"]): row
        for row in summary
    }
    if len(summary_map) != len(summary) or set(summary_map) != set(GROUPS):
        raise ValueError("Full-precision Block 16 summary has missing or duplicate groups.")

    table_rows: list[dict[str, str]] = []
    for dataset, model, profile in GROUPS:
        key = (dataset, model, profile)
        runs = groups[key]
        seeds = [int(run["seed"]) for run in runs]
        if len(runs) != 3 or set(seeds) != SEEDS or len(set(seeds)) != len(seeds):
            raise ValueError(f"Expected exactly one run for each seed 42/43/44: {key}")

        reference = summary_map[key]
        if int(reference["n_runs"]) != len(runs):
            raise ValueError(f"Invalid run count in Block 16 summary: {key}")

        out = {"dataset": dataset, "model": model, "profile": profile}
        for source_metric, export_metric in METRICS:
            values = [float(run[source_metric]) for run in runs]
            mean = statistics.fmean(values)
            std = statistics.pstdev(values)  # ddof=0; same convention as collector 55
            for suffix, calculated in (("mean", mean), ("std", std)):
                stored = float(reference[f"{source_metric}_{suffix}"])
                if not math.isclose(calculated, stored, rel_tol=0, abs_tol=1e-12):
                    raise ValueError(
                        f"Per-seed data disagree with full-precision summary: "
                        f"{key} {source_metric}_{suffix}: {calculated} vs {stored}"
                    )
                out[f"{export_metric}_{suffix}"] = f"{calculated:.4f}"
        table_rows.append(out)

    result = io.StringIO(newline="")
    writer = csv.DictWriter(result, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(table_rows)
    return result.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Verify without writing (default).")
    mode.add_argument("--write", action="store_true", help="Replace the published table.")
    args = parser.parse_args()

    expected = derive_table(args.repo_root)
    output = args.repo_root / "tables/final_full_multiseed_summary.csv"
    current = output.read_text(encoding="utf-8") if output.is_file() else ""

    if args.write:
        if current != expected:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(expected, encoding="utf-8")
        print(f"SYNCED {output} from 12 frozen per-seed rows (ddof=0)")
        return 0
    if current != expected:
        diff = "".join(
            difflib.unified_diff(
                current.splitlines(keepends=True),
                expected.splitlines(keepends=True),
                fromfile=str(output),
                tofile="derived-from-block16",
            )
        )
        print(diff or "Published Block 16 summary is missing or inconsistent.")
        return 1

    print("PASS: publication table matches 12 per-seed rows and source summary (ddof=0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
