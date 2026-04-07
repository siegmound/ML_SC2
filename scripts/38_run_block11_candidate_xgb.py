
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

from sc2proj.checkpointing import append_jsonl, completed_keys, load_jsonl
from sc2proj.feature_registry import select_features_by_family
from sc2proj.metrics import classification_summary
from sc2proj.training_io import (
    RunLogger,
    downsample_loaded_split,
    load_split_manifest,
    make_run_dir,
    select_split_frames,
    write_artifacts_manifest,
    write_predictions,
)
from sc2proj.utils import load_dataframe_from_zip, write_json


PROFILE_TO_EXCLUDE = {
    "full": [],
    "no_counter": ["counter"],
    "no_counter_no_losses": ["counter", "losses"],
}


def candidate_key(params: dict) -> str:
    payload = json.dumps(params, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]


def build_candidates():
    return [
        {
            "max_depth": 5,
            "learning_rate": 0.03,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
        },
        {
            "max_depth": 6,
            "learning_rate": 0.05,
            "min_child_weight": 5,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 2.0,
        },
        {
            "max_depth": 4,
            "learning_rate": 0.05,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
        },
        {
            "max_depth": 6,
            "learning_rate": 0.03,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
        },
    ]


def make_model(seed: int, device: str, n_jobs: int, early_stopping_rounds: int, **params):
    extra = {"tree_method": "hist"}
    if device == "cuda":
        extra["device"] = "cuda"
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_estimators=2000,
        early_stopping_rounds=early_stopping_rounds,
        n_jobs=n_jobs,
        **extra,
        **params,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-zip", required=True)
    ap.add_argument("--split-json", required=True)
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--profile", choices=sorted(PROFILE_TO_EXCLUDE), default="full")
    ap.add_argument("--cv-scoring", choices=["accuracy", "roc_auc", "neg_log_loss"], default="neg_log_loss")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--max-train-rows", type=int, default=35000)
    ap.add_argument("--max-val-rows", type=int, default=12000)
    ap.add_argument("--max-test-rows", type=int, default=19452)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    dataset_zip = Path(args.dataset_zip)
    split_json = Path(args.split_json)
    split_manifest = load_split_manifest(split_json)
    seed = int(split_manifest.get("seed", 42))
    run_dir = make_run_dir(Path(args.output_dir), "xgb", args.dataset_name, seed)
    logger = RunLogger()

    df = load_dataframe_from_zip(dataset_zip)
    loaded = select_split_frames(df, split_manifest)
    loaded = downsample_loaded_split(
        loaded,
        seed=seed,
        max_train_rows=args.max_train_rows,
        max_val_rows=args.max_val_rows,
        max_test_rows=args.max_test_rows,
    )

    selected_features = select_features_by_family(
        loaded.feature_columns,
        exclude_families=PROFILE_TO_EXCLUDE[args.profile],
    )

    logger.log(
        f"XGB block11 candidate run: seed={seed} profile={args.profile} rows train={len(loaded.X_train)} val={len(loaded.X_val)} test={len(loaded.X_test)} device={args.device}"
    )

    search_log = run_dir / f"block11_validation_search_{args.profile}.jsonl"
    done = completed_keys(search_log) if args.resume else set()
    if done:
        logger.log(f"Resuming from existing search log with {len(done)} completed candidates")

    best_score = None
    best_params = None

    for params in build_candidates():
        ckey = candidate_key({"profile": args.profile, **params})
        if ckey in done:
            continue
        model = make_model(seed=seed, device=args.device, n_jobs=args.n_jobs, early_stopping_rounds=50, **params)
        model.fit(
            loaded.X_train[selected_features],
            loaded.y_train,
            eval_set=[(loaded.X_val[selected_features], loaded.y_val)],
            verbose=False,
        )
        y_prob = model.predict_proba(loaded.X_val[selected_features])[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        if args.cv_scoring == "accuracy":
            score = float((y_pred == loaded.y_val.to_numpy()).mean())
        elif args.cv_scoring == "roc_auc":
            from sklearn.metrics import roc_auc_score
            score = float(roc_auc_score(loaded.y_val, y_prob))
        else:
            from sklearn.metrics import log_loss
            score = -float(log_loss(loaded.y_val, y_prob))
        append_jsonl(
            search_log,
            {
                "candidate_key": ckey,
                "profile": args.profile,
                "params": params,
                "score": score,
                "best_iteration": int(getattr(model, "best_iteration", -1)),
            },
        )
        if best_score is None or score > best_score:
            best_score = score
            best_params = params

    if best_params is None:
        rows = load_jsonl(search_log)
        if not rows:
            raise RuntimeError("No completed candidates found in checkpoint log.")
        rows = [r for r in rows if r.get("profile") == args.profile]
        rows.sort(key=lambda r: float(r["score"]), reverse=True)
        best_score = float(rows[0]["score"])
        best_params = rows[0]["params"]

    logger.log(f"Best params after checkpointed validation search: {best_params}")
    logger.log(f"Best validation score: {best_score:.6f}")

    final_model = make_model(seed=seed, device=args.device, n_jobs=args.n_jobs, early_stopping_rounds=50, **best_params)
    final_model.fit(
        loaded.X_train[selected_features],
        loaded.y_train,
        eval_set=[(loaded.X_val[selected_features], loaded.y_val)],
        verbose=False,
    )

    y_prob = final_model.predict_proba(loaded.X_test[selected_features])[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    summary = classification_summary(loaded.y_test, y_pred, y_prob)
    summary.update(
        {
            "model_name": "xgb",
            "dataset_name": args.dataset_name,
            "seed": seed,
            "profile": args.profile,
            "n_features": len(selected_features),
            "feature_columns": selected_features,
            "best_params": best_params,
            "best_validation_score": best_score,
            "best_iteration": int(getattr(final_model, "best_iteration", -1)),
        }
    )

    fi = pd.DataFrame({"feature": selected_features, "importance": final_model.feature_importances_}).sort_values(
        "importance", ascending=False
    )
    fi.to_csv(run_dir / "feature_importance.csv", index=False)
    write_json(summary, run_dir / "metrics_summary.json")
    write_json(
        {
            "dataset_zip": args.dataset_zip,
            "split_json": args.split_json,
            "dataset_name": args.dataset_name,
            "profile": args.profile,
            "cv_scoring": args.cv_scoring,
            "device": args.device,
            "max_train_rows": args.max_train_rows,
            "max_val_rows": args.max_val_rows,
            "max_test_rows": args.max_test_rows,
            "n_jobs": args.n_jobs,
            "selected_features": selected_features,
        },
        run_dir / "config_used.json",
    )
    write_predictions(run_dir / "predictions.csv", loaded.test_df, y_pred, y_prob)
    logger.dump(run_dir / "raw_log.txt")
    write_artifacts_manifest(
        run_dir,
        [
            run_dir / "metrics_summary.json",
            run_dir / "config_used.json",
            run_dir / "predictions.csv",
            run_dir / "feature_importance.csv",
            run_dir / "raw_log.txt",
            search_log,
        ],
    )


if __name__ == "__main__":
    main()
