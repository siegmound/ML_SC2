from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sc2proj.metrics import classification_summary
from sc2proj.modeling import threshold_predictions
from sc2proj.training_io import RunLogger, downsample_loaded_split, load_split_manifest, make_run_dir, select_split_frames, write_artifacts_manifest, write_predictions
from sc2proj.utils import load_dataframe_from_zip, write_json


def build_search_space(stage: str) -> list[dict]:
    if stage == "smoke":
        return [
            {"max_depth": 4, "learning_rate": 0.05, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0},
            {"max_depth": 5, "learning_rate": 0.05, "min_child_weight": 3, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0},
        ]
    if stage == "standard":
        return [
            {"max_depth": 4, "learning_rate": 0.03, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0},
            {"max_depth": 5, "learning_rate": 0.03, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0},
            {"max_depth": 6, "learning_rate": 0.05, "min_child_weight": 3, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0},
            {"max_depth": 5, "learning_rate": 0.05, "min_child_weight": 5, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 2.0},
        ]
    return [
        {"max_depth": 4, "learning_rate": 0.01, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0},
        {"max_depth": 4, "learning_rate": 0.03, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0},
        {"max_depth": 5, "learning_rate": 0.03, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0},
        {"max_depth": 6, "learning_rate": 0.03, "min_child_weight": 3, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0},
        {"max_depth": 5, "learning_rate": 0.05, "min_child_weight": 3, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 2.0},
        {"max_depth": 6, "learning_rate": 0.05, "min_child_weight": 5, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 2.0},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-zip", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--dataset-name", type=str, default="canonical")
    parser.add_argument("--cv-scoring", choices=["accuracy", "roc_auc", "neg_log_loss"], default="neg_log_loss")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--fast-smoke", action="store_true")
    parser.add_argument("--search-stage", choices=["smoke", "standard", "extended"], default="standard")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    args = parser.parse_args()

    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        raise SystemExit(f"xgboost is required for 07_train_xgb.py: {exc}")

    split = load_split_manifest(args.split_json)
    seed = int(split["seed"])
    run_dir = make_run_dir(args.output_dir, "xgb", args.dataset_name, seed)
    logger = RunLogger()
    df = load_dataframe_from_zip(args.dataset_zip)
    loaded = select_split_frames(df, split)
    loaded = downsample_loaded_split(loaded, seed=seed, max_train_rows=args.max_train_rows, max_val_rows=args.max_val_rows, max_test_rows=args.max_test_rows)

    stage = "smoke" if args.fast_smoke else args.search_stage
    search_space = build_search_space(stage)
    n_estimators = 300 if stage == "smoke" else 1200 if stage == "standard" else 2000
    early_stopping_rounds = 20 if stage == "smoke" else 40 if stage == "standard" else 50

    logger.log(f"Running XGB validation search stage={stage} scoring={args.cv_scoring} device={args.device}")
    logger.log(f"Materialized rows train={len(loaded.X_train)} val={len(loaded.X_val)} test={len(loaded.X_test)}")
    cv_rows = []
    best_score = None
    best_params = None
    for params in search_space:
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device=args.device,
            random_state=seed,
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping_rounds,
            n_jobs=4,
            **params,
        )
        model.fit(loaded.X_train, loaded.y_train, eval_set=[(loaded.X_val, loaded.y_val)], verbose=False)
        y_prob_val = model.predict_proba(loaded.X_val)[:, 1]
        y_pred_val = threshold_predictions(y_prob_val)
        if args.cv_scoring == "accuracy":
            score = float((loaded.y_val.to_numpy() == y_pred_val).mean())
        elif args.cv_scoring == "roc_auc":
            from sklearn.metrics import roc_auc_score
            score = float(roc_auc_score(loaded.y_val, y_prob_val))
        else:
            from sklearn.metrics import log_loss
            score = float(-log_loss(loaded.y_val, y_prob_val, labels=[0, 1]))
        cv_rows.append({"params": params, "score": score, "best_iteration": int(getattr(model, "best_iteration", -1))})
        if best_score is None or score > best_score:
            best_score = score
            best_params = dict(params)

    assert best_params is not None and best_score is not None
    logger.log(f"Best params: {best_params}")
    logger.log(f"Best validation score: {best_score:.6f}")

    final_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device=args.device,
        random_state=seed,
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        n_jobs=4,
        **best_params,
    )
    final_model.fit(loaded.X_train, loaded.y_train, eval_set=[(loaded.X_val, loaded.y_val)], verbose=False)
    y_prob = final_model.predict_proba(loaded.X_test)[:, 1]
    y_pred = threshold_predictions(y_prob)
    metrics = classification_summary(loaded.y_test, y_pred, y_prob)
    metrics.update({
        "seed": seed,
        "model_name": "xgb",
        "dataset_name": args.dataset_name,
        "cv_scoring": args.cv_scoring,
        "best_params": best_params,
        "best_validation_score": best_score,
        "best_iteration": int(getattr(final_model, "best_iteration", -1)),
        "device": args.device,
        "n_train_rows": int(len(loaded.X_train) + len(loaded.X_val)),
        "n_test_rows": int(len(loaded.X_test)),
        "n_train_replays": int(pd.concat([loaded.groups_train, loaded.groups_val]).nunique()),
        "n_test_replays": int(loaded.groups_test.nunique()),
        "fast_smoke": bool(args.fast_smoke),
        "search_stage": stage,
    })

    feature_importance = pd.DataFrame({
        "feature": loaded.feature_columns,
        "importance": final_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    config_path = run_dir / "config_used.json"
    metrics_path = run_dir / "metrics_summary.json"
    preds_path = run_dir / "predictions.csv"
    cv_path = run_dir / "validation_search_results.csv"
    importance_path = run_dir / "feature_importance.csv"
    log_path = run_dir / "raw_log.txt"

    write_json({
        "dataset_zip": str(args.dataset_zip),
        "split_json": str(args.split_json),
        "dataset_name": args.dataset_name,
        "cv_scoring": args.cv_scoring,
        "search_space": search_space,
        "device": args.device,
        "fast_smoke": bool(args.fast_smoke),
        "search_stage": stage,
        "max_train_rows": args.max_train_rows,
        "max_val_rows": args.max_val_rows,
        "max_test_rows": args.max_test_rows,
    }, config_path)
    write_json(metrics, metrics_path)
    write_predictions(preds_path, loaded.test_df, y_pred, y_prob)
    pd.DataFrame(cv_rows).to_csv(cv_path, index=False)
    feature_importance.to_csv(importance_path, index=False)
    logger.dump(log_path)
    write_artifacts_manifest(run_dir, [config_path, metrics_path, preds_path, cv_path, importance_path, log_path])


if __name__ == "__main__":
    main()
