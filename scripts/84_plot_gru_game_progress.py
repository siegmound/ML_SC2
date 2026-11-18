#!/usr/bin/env python3
"""S8: render accuracy, ROC-AUC and log-loss progress curves from validated CSV.

These are descriptive means over replay-disjoint seeds reusing the same corpus.
Figures do not constitute prospective test metrics or statistical significance.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXPECTED = (25, 50, 75, 90)
METRICS = (
    ("accuracy", "Accuracy (%)", "accuracy_by_progress.png", 100.0),
    ("roc_auc", "ROC-AUC", "roc_auc_by_progress.png", 1.0),
    ("log_loss", "Log loss (lower is better)", "log_loss_by_progress.png", 1.0),
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics-csv", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    source = pd.read_csv(args.metrics_csv)
    required = {"seed", "progress_percent", "arm", "replays", "accuracy", "roc_auc", "log_loss"}
    if not required.issubset(source.columns):
        ap.error("S8 CSV lacks required metrics columns")
    if source.empty or source.duplicated(["seed", "progress_percent", "arm"]).any():
        ap.error("Empty or repeated seed/progress/arm rows")
    expected_progress = set(EXPECTED)
    for (seed, arm), group in source.groupby(["seed", "arm"]):
        if set(group.progress_percent) != expected_progress:
            ap.error(f"Missing progress checkpoint for seed={seed}, arm={arm}")
        if group.replays.nunique() != 1:
            ap.error(f"Unbalanced replay cohort for seed={seed}, arm={arm}")
    if source.isna().any().any():
        ap.error("Null metrics; cannot draw an accurate comparison")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    for metric, ylabel, filename, scale in METRICS:
        target = output / filename
        if target.exists():
            ap.error(f"Refusing to replace existing figure: {target}")
        fig, ax = plt.subplots(figsize=(8, 5))
        for arm, group in source.groupby("arm", sort=True):
            mean = group.groupby("progress_percent")[metric].mean().reindex(EXPECTED)
            if mean.isna().any() or not np.isfinite(mean.to_numpy()).all():
                ap.error(f"Invalid {metric} series for {arm}")
            ax.plot(list(EXPECTED), mean.to_numpy() * scale, marker="o", label=arm)
        ax.set_xlabel("Observed fraction of complete replay (%)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{metric.replace('_', ' ').title()} by game progress — validation")
        ax.set_xticks(EXPECTED)
        ax.grid(alpha=0.2)
        ax.legend(loc="best", fontsize=9)
        ax.text(0.5, -0.17, "Retrospective checkpoints; fixed-9 GRU; unweighted seed means",
                ha="center", transform=ax.transAxes, fontsize=9)
        fig.tight_layout()
        fig.savefig(target, dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {target}")


if __name__ == "__main__":
    main()
