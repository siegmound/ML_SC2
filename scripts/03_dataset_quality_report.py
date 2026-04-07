from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sc2proj.dataset_checks import build_dataset_quality_report
from sc2proj.utils import load_dataframe_from_zip, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-zip", type=Path, required=True)
    parser.add_argument("--dataset-name", type=str, default="dataset")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "experiments" / "dataset_quality")
    args = parser.parse_args()

    df = load_dataframe_from_zip(args.dataset_zip)
    report = build_dataset_quality_report(df)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.dataset_name
    report.feature_profile.to_csv(args.output_dir / f"{prefix}_feature_profile.csv", index=False)
    report.missingness_report.to_csv(args.output_dir / f"{prefix}_missingness_report.csv", index=False)
    report.constant_features.to_csv(args.output_dir / f"{prefix}_constant_features.csv", index=False)
    report.correlation_matrix.to_csv(args.output_dir / f"{prefix}_correlation_matrix.csv")
    write_json(report.summary, args.output_dir / f"{prefix}_dataset_quality_summary.json")
    print(report.summary)


if __name__ == "__main__":
    main()
