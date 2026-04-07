from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sc2proj.metrics import classification_summary
from sc2proj.modeling import run_group_cv_search, threshold_predictions
from sc2proj.training_io import RunLogger, downsample_loaded_split, load_split_manifest, make_run_dir, select_split_frames, write_artifacts_manifest, write_predictions
from sc2proj.utils import load_dataframe_from_zip, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-zip", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--dataset-name", type=str, default="canonical")
    parser.add_argument("--cv-scoring", choices=["accuracy", "roc_auc", "neg_log_loss"], default="neg_log_loss")
    parser.add_argument("--fast-smoke", action="store_true")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    args = parser.parse_args()

    split = load_split_manifest(args.split_json)
    seed = int(split["seed"])
    run_dir = make_run_dir(args.output_dir, "rf", args.dataset_name, seed)
    logger = RunLogger()
    df = load_dataframe_from_zip(args.dataset_zip)
    loaded = select_split_frames(df, split)
    loaded = downsample_loaded_split(loaded, seed=seed, max_train_rows=args.max_train_rows, max_val_rows=args.max_val_rows, max_test_rows=args.max_test_rows)

    base = RandomForestClassifier(random_state=seed, n_jobs=-1)
    if args.fast_smoke:
        param_grid = [
            {"n_estimators": 150, "max_depth": 12, "min_samples_split": 5, "min_samples_leaf": 2, "max_features": "sqrt", "class_weight": None},
            {"n_estimators": 200, "max_depth": 16, "min_samples_split": 5, "min_samples_leaf": 2, "max_features": "sqrt", "class_weight": "balanced"},
        ]
        n_splits = 2
        perm_repeats = 1
    else:
        param_grid = [
            {"n_estimators": 300, "max_depth": 12, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt", "class_weight": None},
            {"n_estimators": 500, "max_depth": 16, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt", "class_weight": None},
            {"n_estimators": 500, "max_depth": None, "min_samples_split": 5, "min_samples_leaf": 2, "max_features": "sqrt", "class_weight": None},
            {"n_estimators": 500, "max_depth": 16, "min_samples_split": 5, "min_samples_leaf": 2, "max_features": "log2", "class_weight": "balanced"},
        ]
        n_splits = 3
        perm_repeats = 5

    logger.log(f"Running RF group-aware CV search with scoring={args.cv_scoring}; fast_smoke={args.fast_smoke}")
    logger.log(f"Materialized rows train={len(loaded.X_train)} val={len(loaded.X_val)} test={len(loaded.X_test)}")
    search = run_group_cv_search(base, param_grid, loaded.X_train, loaded.y_train, loaded.groups_train, scoring=args.cv_scoring, n_splits=n_splits)
    logger.log(f"Best params: {search.best_params}")
    logger.log(f"Best CV score: {search.best_score:.6f}")

    best_model = search.best_estimator
    best_model.fit(pd.concat([loaded.X_train, loaded.X_val]), pd.concat([loaded.y_train, loaded.y_val]))
    y_prob = best_model.predict_proba(loaded.X_test)[:, 1]
    y_pred = threshold_predictions(y_prob)
    metrics = classification_summary(loaded.y_test, y_pred, y_prob)
    metrics.update({
        "seed": seed,
        "model_name": "rf",
        "dataset_name": args.dataset_name,
        "cv_scoring": args.cv_scoring,
        "best_params": search.best_params,
        "best_cv_score": search.best_score,
        "n_train_rows": int(len(loaded.X_train) + len(loaded.X_val)),
        "n_test_rows": int(len(loaded.X_test)),
        "n_train_replays": int(pd.concat([loaded.groups_train, loaded.groups_val]).nunique()),
        "n_test_replays": int(loaded.groups_test.nunique()),
        "fast_smoke": bool(args.fast_smoke),
    })

    feature_importance = pd.DataFrame({
        "feature": loaded.feature_columns,
        "importance": best_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    if args.fast_smoke:
        permutation_df = pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
    else:
        perm = permutation_importance(best_model, loaded.X_test, loaded.y_test, n_repeats=perm_repeats, random_state=seed, n_jobs=1)
        permutation_df = pd.DataFrame({
            "feature": loaded.feature_columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }).sort_values("importance_mean", ascending=False)

    config_path = run_dir / "config_used.json"
    metrics_path = run_dir / "metrics_summary.json"
    preds_path = run_dir / "predictions.csv"
    cv_path = run_dir / "cv_results.csv"
    importance_path = run_dir / "feature_importance.csv"
    perm_path = run_dir / "permutation_importance.csv"
    log_path = run_dir / "raw_log.txt"

    write_json({
        "dataset_zip": str(args.dataset_zip),
        "split_json": str(args.split_json),
        "dataset_name": args.dataset_name,
        "cv_scoring": args.cv_scoring,
        "candidate_params": param_grid,
        "fast_smoke": bool(args.fast_smoke),
        "max_train_rows": args.max_train_rows,
        "max_val_rows": args.max_val_rows,
        "max_test_rows": args.max_test_rows,
    }, config_path)
    write_json(metrics, metrics_path)
    write_predictions(preds_path, loaded.test_df, y_pred, y_prob)
    pd.DataFrame(search.cv_results).to_csv(cv_path, index=False)
    feature_importance.to_csv(importance_path, index=False)
    permutation_df.to_csv(perm_path, index=False)
    logger.dump(log_path)
    write_artifacts_manifest(run_dir, [config_path, metrics_path, preds_path, cv_path, importance_path, perm_path, log_path])


if __name__ == "__main__":
    main()
