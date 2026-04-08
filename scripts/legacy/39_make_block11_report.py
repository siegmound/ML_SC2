
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def collect_model_rows(results_dir: Path, model_name: str, dataset_name: str) -> list[dict]:
    rows = []
    model_root = results_dir / model_name / dataset_name
    if not model_root.exists():
        return rows
    for metrics_path in sorted(model_root.glob("seed_*/metrics_summary.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(payload)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--output-dir", default="results/block11_report")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_name in ["rf", "xgb"]:
        rows.extend(collect_model_rows(Path(args.results_dir), model_name, args.dataset_name))

    if not rows:
        raise SystemExit("No metrics_summary.json files found.")

    df = pd.DataFrame(rows)
    detail_path = out_dir / f"{args.dataset_name}_block11_detailed.csv"
    df.to_csv(detail_path, index=False)

    metric_cols = [c for c in ["accuracy", "balanced_accuracy", "roc_auc", "log_loss"] if c in df.columns]
    grouped = df.groupby(["model_name", "profile"], dropna=False)[metric_cols].agg(["mean", "std", "count"])
    grouped.columns = ["_".join([a, b]).strip("_") for a, b in grouped.columns]
    grouped = grouped.reset_index()
    grouped.to_csv(out_dir / f"{args.dataset_name}_block11_summary.csv", index=False)

    leaderboard = grouped.sort_values(["accuracy_mean", "roc_auc_mean"], ascending=[False, False]).reset_index(drop=True)
    leaderboard.to_csv(out_dir / f"{args.dataset_name}_block11_leaderboard.csv", index=False)

    report = {
        "dataset_name": args.dataset_name,
        "n_runs": int(len(df)),
        "models": sorted(df["model_name"].unique().tolist()),
        "profiles": sorted([p for p in df["profile"].dropna().unique().tolist()]),
        "detail_csv": str(detail_path),
        "summary_csv": str(out_dir / f"{args.dataset_name}_block11_summary.csv"),
        "leaderboard_csv": str(out_dir / f"{args.dataset_name}_block11_leaderboard.csv"),
    }
    (out_dir / f"{args.dataset_name}_block11_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
