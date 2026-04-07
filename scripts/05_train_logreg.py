from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sc2proj.metrics import classification_summary
from sc2proj.modeling import make_numeric_preprocessor, run_group_cv_search, threshold_predictions
from sc2proj.training_io import RunLogger, load_split_manifest, make_run_dir, select_split_frames, write_artifacts_manifest, write_predictions
from sc2proj.utils import load_dataframe_from_zip, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-zip", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--dataset-name", type=str, default="canonical")
    parser.add_argument("--cv-scoring", choices=["accuracy", "roc_auc", "neg_log_loss"], default="neg_log_loss")
    parser.add_argument("--fast-smoke", action="store_true")
    args = parser.parse_args()

    split = load_split_manifest(args.split_json)
    seed = int(split["seed"])
    run_dir = make_run_dir(args.output_dir, "logreg", args.dataset_name, seed)
    logger = RunLogger()
    logger.log(f"Loading dataset from {args.dataset_zip}")
    df = load_dataframe_from_zip(args.dataset_zip)
    loaded = select_split_frames(df, split)

    base = Pipeline([
        ("prep", make_numeric_preprocessor("standard")),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=seed)),
    ])
    param_grid = [
        {"clf__C": 0.1, "clf__class_weight": None},
        {"clf__C": 1.0, "clf__class_weight": None},
        {"clf__C": 10.0, "clf__class_weight": None},
        {"clf__C": 1.0, "clf__class_weight": "balanced"},
    ]
    n_splits = 2 if args.fast_smoke else 3
    if args.fast_smoke:
        param_grid = [
            {"clf__C": 1.0, "clf__class_weight": None},
            {"clf__C": 1.0, "clf__class_weight": "balanced"},
        ]

    logger.log(f"Running group-aware CV search with scoring={args.cv_scoring}; fast_smoke={args.fast_smoke}")
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
        "model_name": "logreg",
        "dataset_name": args.dataset_name,
        "cv_scoring": args.cv_scoring,
        "best_params": search.best_params,
        "best_cv_score": search.best_score,
        "n_train_rows": int(len(loaded.X_train) + len(loaded.X_val)),
        "n_test_rows": int(len(loaded.X_test)),
        "n_train_replays": int(pd.concat([loaded.groups_train, loaded.groups_val]).nunique()),
        "n_test_replays": int(loaded.groups_test.nunique()),
    })

    config_path = run_dir / "config_used.json"
    metrics_path = run_dir / "metrics_summary.json"
    preds_path = run_dir / "predictions.csv"
    cv_path = run_dir / "cv_results.csv"
    log_path = run_dir / "raw_log.txt"

    write_json({
        "dataset_zip": str(args.dataset_zip),
        "split_json": str(args.split_json),
        "dataset_name": args.dataset_name,
        "cv_scoring": args.cv_scoring,
        "candidate_params": param_grid,
    }, config_path)
    write_json(metrics, metrics_path)
    write_predictions(preds_path, loaded.test_df, y_pred, y_prob)
    pd.DataFrame(search.cv_results).to_csv(cv_path, index=False)
    logger.dump(log_path)
    write_artifacts_manifest(run_dir, [config_path, metrics_path, preds_path, cv_path, log_path])


if __name__ == "__main__":
    main()
