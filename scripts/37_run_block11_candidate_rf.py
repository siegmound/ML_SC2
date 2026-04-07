
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

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
            "n_estimators": 300,
            "max_depth": 16,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "class_weight": None,
        },
        {
            "n_estimators": 500,
            "max_depth": 20,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "class_weight": None,
        },
        {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
        },
        {
            "n_estimators": 700,
            "max_depth": 24,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
        },
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-zip", required=True)
    ap.add_argument("--split-json", required=True)
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--profile", choices=sorted(PROFILE_TO_EXCLUDE), default="full")
    ap.add_argument("--cv-scoring", choices=["accuracy", "roc_auc", "neg_log_loss"], default="neg_log_loss")
    ap.add_argument("--max-train-rows", type=int, default=30000)
    ap.add_argument("--max-val-rows", type=int, default=12000)
    ap.add_argument("--max-test-rows", type=int, default=19452)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    dataset_zip = Path(args.dataset_zip)
    split_json = Path(args.split_json)
    split_manifest = load_split_manifest(split_json)
    seed = int(split_manifest.get("seed", 42))
    run_dir = make_run_dir(Path(args.output_dir), "rf", args.dataset_name, seed)
    logger = RunLogger()

    logger.log(
        f"RF block11 candidate run: seed={seed} profile={args.profile}"
    )

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
        f"rows train={len(loaded.X_train)} val={len(loaded.X_val)} test={len(loaded.X_test)} features={len(selected_features)}"
    )

    search_log = run_dir / f"block11_search_results_{args.profile}.jsonl"
    done = completed_keys(search_log) if args.resume else set()
    if done:
        logger.log(f"Resuming from existing search log with {len(done)} completed candidates")

    cv = GroupKFold(n_splits=3)
    candidates = build_candidates()

    best_score = None
    best_params = None

    for params in candidates:
        ckey = candidate_key({"profile": args.profile, **params})
        if ckey in done:
            continue

        fold_scores = []
        for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(loaded.X_train[selected_features], loaded.y_train, groups=loaded.groups_train), start=1):
            model = RandomForestClassifier(
                random_state=seed,
                n_jobs=args.n_jobs,
                **params,
            )
            Xtr = loaded.X_train.iloc[tr_idx][selected_features]
            ytr = loaded.y_train.iloc[tr_idx]
            Xva = loaded.X_train.iloc[va_idx][selected_features]
            yva = loaded.y_train.iloc[va_idx]

            model.fit(Xtr, ytr)
            yprob = model.predict_proba(Xva)[:, 1]
            ypred = (yprob >= 0.5).astype(int)

            if args.cv_scoring == "accuracy":
                score = float((ypred == yva.to_numpy()).mean())
            elif args.cv_scoring == "roc_auc":
                from sklearn.metrics import roc_auc_score
                score = float(roc_auc_score(yva, yprob))
            else:
                from sklearn.metrics import log_loss
                score = -float(log_loss(yva, yprob))

            logger.log(f"Candidate {params} fold={fold_idx} score={score:.6f}")
            append_jsonl(
                search_log,
                {
                    "candidate_key": ckey,
                    "profile": args.profile,
                    "params": params,
                    "fold": fold_idx,
                    "score": score,
                },
            )
            fold_scores.append(score)

        mean_score = float(np.mean(fold_scores))
        logger.log(f"Candidate {params} mean_score={mean_score:.6f}")
        append_jsonl(
            search_log,
            {
                "candidate_key": ckey,
                "profile": args.profile,
                "params": params,
                "fold": "mean",
                "score": mean_score,
            },
        )
        if best_score is None or mean_score > best_score:
            best_score = mean_score
            best_params = params

    # If all candidates were already completed, recover the best from the log.
    if best_params is None:
        rows = load_jsonl(search_log)
        means = [r for r in rows if str(r.get("fold")) == "mean" and r.get("profile") == args.profile]
        if not means:
            raise RuntimeError("No completed candidates found in checkpoint log.")
        means.sort(key=lambda r: float(r["score"]), reverse=True)
        best_params = means[0]["params"]
        best_score = float(means[0]["score"])

    logger.log(f"Best params after checkpointed search: {best_params}")
    logger.log(f"Best CV score: {best_score:.6f}")

    final_model = RandomForestClassifier(
        random_state=seed,
        n_jobs=args.n_jobs,
        **best_params,
    )
    trainval_X = pd.concat([loaded.X_train[selected_features], loaded.X_val[selected_features]], axis=0)
    trainval_y = pd.concat([loaded.y_train, loaded.y_val], axis=0)
    final_model.fit(trainval_X, trainval_y)

    y_prob = final_model.predict_proba(loaded.X_test[selected_features])[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    summary = classification_summary(loaded.y_test, y_pred, y_prob)
    summary.update(
        {
            "model_name": "rf",
            "dataset_name": args.dataset_name,
            "seed": seed,
            "profile": args.profile,
            "n_features": len(selected_features),
            "feature_columns": selected_features,
            "best_params": best_params,
            "best_cv_score": best_score,
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
