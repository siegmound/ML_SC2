#!/usr/bin/env python3
"""Re-evaluate v4 clean raw-derived data with RF/XGB/MLP on replay splits.

Reuses pre-existing validated four-way RF/XGB calibration and MLP
validation-selection runners. Training never consumes frozen historical
v3.1/v3.2 data; all outputs are in a NEW results/v4_* directory.
--plan is the default and does not execute models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from sc2proj.split_utils import SplitConfig, make_group_split
from sc2proj.utils import load_dataframe_from_zip

MODELS = ("rf", "xgb", "deep")
SEEDS = (42, 43, 44)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest(dataset_dir: Path):
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "ml_sc2.v4.raw_replay_to_dataset.v1":
        raise ValueError("Not an independently rebuilt v4 dataset manifest")
    if manifest.get("historical_frozen_data_used_for_build") is not False:
        raise ValueError("Historical frozen data could not be verified as excluded")
    data_zip = dataset_dir / "sc2_v4_clean.zip"
    if digest(data_zip) != manifest["dataset_zip_sha256"]:
        raise ValueError("v4 dataset ZIP SHA-256 changed")
    if digest(dataset_dir / "per_replay_source_audit.csv") != manifest["per_replay_audit_sha256"]:
        raise ValueError("v4 raw replay audit SHA-256 changed")
    return manifest, data_zip


def validate_and_plan_splits(df, seeds):
    groups = df["replay_id"].astype(str)
    if not groups.str.fullmatch(r"[a-f0-9]{64}\.SC2Replay").all():
        raise ValueError("v4 replay ID is not content-SHA-256-derived")
    if df.duplicated(["replay_id", "time_sec"]).any():
        raise ValueError("Duplicate game/time snapshots")
    if (df.groupby("replay_id")["p1_wins"].nunique() != 1).any():
        raise ValueError("Game-level labels are not constant")
    if df["p1_wins"].nunique() != 2:
        raise ValueError("Both winner classes required")
    if groups.nunique() < 40:
        raise ValueError("At least 40 independent replay identities needed")
    plans = {}
    for seed in seeds:
        split = make_group_split(
            df, "replay_id", "p1_wins", SplitConfig(seed=int(seed))
        )
        sets = [set(split[x]) for x in ("train_groups", "val_groups", "test_groups")]
        if len(set().union(*sets)) != groups.nunique():
            raise ValueError("Split does not cover every replay identity")
        if any(sets[i] & sets[j] for i in range(3) for j in range(i+1, 3)):
            raise ValueError("Data leakage: one replay in multiple partitions")
        if len(split["val_groups"]) < 4:
            raise ValueError("Insufficient val groups for four-way calibration")
        # We do not inspect test labels to choose models, features or hyperparams.
        # The newly fixed split membership itself is determined by replay ID only.
        plans[int(seed)] = split
    return plans


def commands(dataset_zip: Path, splits: Path, output_root: Path, *,
             models, seeds, stage, device, n_jobs):
    commands_list = []
    for seed in seeds:
        split_path = splits / f"v4_clean_split_seed_{seed}.json"
        for model in models:
            if model == "deep":
                cmd = [
                    sys.executable, str(ROOT / "scripts/64_run_deep_finalist.py"),
                    "--dataset-zip", str(dataset_zip),
                    "--split-json", str(split_path),
                    "--dataset-name", "v4_clean_raw",
                    "--device", device, "--selection-metric", "neg_log_loss",
                    "--output-root", str(output_root / "deep_finalist"),
                ]
                run_dir = output_root / "deep_finalist/v4_clean_raw" / f"seed_{seed}"
            else:
                cmd = [
                    sys.executable, str(ROOT / "scripts/67_run_strict_calibration.py"),
                    "--dataset-zip", str(dataset_zip),
                    "--split-json", str(split_path),
                    "--dataset-name", "v4_clean_raw",
                    "--model", model, "--device", device if model == "xgb" else "cpu",
                    "--n-jobs", str(n_jobs),
                    "--n-estimators", str(200 if stage == "smoke" else 700),
                    "--num-boost-round", str(350 if stage == "smoke" else 2000),
                    "--early-stopping-rounds", str(25 if stage == "smoke" else 50),
                    "--output-root", str(output_root / "four_way"),
                ]
                run_dir = output_root / "four_way/v4_clean_raw" / model / f"seed_{seed}"
            commands_list.append((seed, model, run_dir, cmd))
    return commands_list


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--stage", choices=("smoke", "full"), default="smoke")
    p.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    p.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--n-jobs", type=int, default=4)
    x = p.add_mutually_exclusive_group()
    x.add_argument("--plan", action="store_true", help="Only show commands (default)")
    x.add_argument("--execute", action="store_true", help="Run training, new outputs only")
    a = p.parse_args()
    if len(set(a.seeds)) != len(a.seeds) or len(set(a.models)) != len(a.models):
        raise ValueError("Duplicated seed/model options are not allowed")
    if a.n_jobs < 1:
        raise ValueError("n_jobs must be positive")
    data_dir = a.dataset_dir.resolve()
    out_root = a.output_root.resolve()
    if out_root.exists():
        raise FileExistsError(f"Refusing to overwrite/continue existing results: {out_root}")
    if data_dir == out_root or data_dir in out_root.parents or out_root in data_dir.parents:
        raise ValueError("Training output must be separate from dataset source")
    manifest, zip_path = load_manifest(data_dir)
    df = load_dataframe_from_zip(zip_path)
    split_plans = validate_and_plan_splits(df, a.seeds)
    split_dir = out_root / "splits"
    plan = commands(
        zip_path, split_dir, out_root,
        models=a.models, seeds=a.seeds,
        stage=a.stage, device=a.device, n_jobs=a.n_jobs
    )
    print(f"v4 dataset: {zip_path}")
    print(f"v4 source SHA: {manifest['dataset_zip_sha256']}")
    print(f"Independent replays: {manifest['included_replays']}, snapshots: {manifest['n_snapshots']}")
    for seed, model, run_dir, cmd in plan:
        print(f"{seed} | {model}: {shlex.join(cmd)}")
    if not a.execute:
        print("PLAN ONLY: nothing written; use --execute for the new experiment suite.")
        return
    out_root.mkdir(parents=True, exist_ok=False)
    split_dir.mkdir(parents=True, exist_ok=False)
    for seed, split in split_plans.items():
        (split_dir / f"v4_clean_split_seed_{seed}.json").write_text(
            json.dumps(split, indent=2) + "\n", encoding="utf-8"
        )
    (out_root / "protocol.json").write_text(json.dumps({
        "schema": "ml_sc2.v4.rebuilt_training_suite.v1",
        "original_frozen_training_data": False,
        "data_manifest_sha256": digest(data_dir / "manifest.json"),
        "data_zip_sha256": digest(zip_path),
        "seeds": a.seeds, "models": a.models, "stage": a.stage,
        "test_selection_policy": "validation_selected_then_single_model_inference",
        "calibration_policy": "replay_disjoint_tune_calibration_test_for_RF_XGB",
        "interpretation": "Independent raw-derived exploratory reruns; not historic v3 parity",
    }, indent=2) + "\n", encoding="utf-8")
    logs = out_root / "logs"
    logs.mkdir()
    for seed, model, run_dir, cmd in plan:
        if run_dir.exists():
            raise FileExistsError(f"Existing run output: {run_dir}")
        logfile = logs / f"{model}_seed_{seed}.log"
        print("EXECUTING:", model, seed, "->", logfile, flush=True)
        env = dict(os.environ, PYTHONPATH=str(SRC))
        with logfile.open("w", encoding="utf-8") as log:
            subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                           check=True, env=env)
    print("V4 suite completed:", out_root)


if __name__ == "__main__":
    main()
