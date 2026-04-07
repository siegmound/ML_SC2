from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sc2proj.split_utils import SplitConfig, make_group_split
from sc2proj.utils import load_dataframe_from_zip, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-zip", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--dataset-name", type=str, default="dataset")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed" / "splits")
    args = parser.parse_args()

    df = load_dataframe_from_zip(args.dataset_zip)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        split = make_group_split(df, group_col="replay_id", target_col="p1_wins", config=SplitConfig(seed=seed))
        base_name = f"{args.dataset_name}_split_seed_{seed}.json"
        write_json(split, args.output_dir / base_name)
        write_json(split, args.output_dir / f"split_seed_{seed}.json")
        print({"seed": seed, "dataset_name": args.dataset_name, "n_train_groups": split["n_train_groups"], "n_val_groups": split["n_val_groups"], "n_test_groups": split["n_test_groups"]})


if __name__ == "__main__":
    main()
