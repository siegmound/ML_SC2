from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.neural_network import MLPClassifier
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
    args = parser.parse_args()

    split = load_split_manifest(args.split_json)
    seed = int(split["seed"])
    run_dir = make_run_dir(args.output_dir, "mlp", args.dataset_name, seed)
    logger = RunLogger()
    df = load_dataframe_from_zip(args.dataset_zip)
    loaded = select_split_frames(df, split)

    candidates = []
    for scaler in ["standard", "quantile"]:
        for hidden_layer_sizes in [(128,), (256,), (256, 128)]:
            for alpha in [1e-4, 1e-3]:
                candidates.append({
                    "prep": make_numeric_preprocessor(scaler),
                    "hidden_layer_sizes": hidden_layer_sizes,
                    "alpha": alpha,
                    "learning_rate_init": 1e-3,
                    "batch_size": 256,
                    "max_iter": 200,
                })

    logger.log(f"Running MLP group-aware CV search with scoring={args.cv_scoring}")
    cv_rows = []
    best_score = None
    best_bundle = None
    for candidate in candidates:
        base = Pipeline([
            ("prep", candidate["prep"]),
            ("clf", MLPClassifier(
                hidden_layer_sizes=candidate["hidden_layer_sizes"],
                alpha=candidate["alpha"],
                learning_rate_init=candidate["learning_rate_init"],
                batch_size=candidate["batch_size"],
                max_iter=candidate["max_iter"],
                early_stopping=False,
                random_state=seed,
            )),
        ])
        params = [{}]
        search = run_group_cv_search(base, params, loaded.X_train, loaded.y_train, loaded.groups_train, scoring=args.cv_scoring)
        score = search.best_score
        row = {
            "scaler": "quantile" if candidate["prep"].named_steps["scaler"].__class__.__name__ == "QuantileTransformer" else "standard",
            "hidden_layer_sizes": candidate["hidden_layer_sizes"],
            "alpha": candidate["alpha"],
            "learning_rate_init": candidate["learning_rate_init"],
            "batch_size": candidate["batch_size"],
            "score": score,
            "scoring": args.cv_scoring,
        }
        cv_rows.append(row)
        if best_score is None or score > best_score:
            best_score = score
            best_bundle = candidate

    assert best_bundle is not None and best_score is not None
    logger.log(f"Best MLP candidate: {best_bundle}")
    logger.log(f"Best CV score: {best_score:.6f}")

    final_model = Pipeline([
        ("prep", best_bundle["prep"]),
        ("clf", MLPClassifier(
            hidden_layer_sizes=best_bundle["hidden_layer_sizes"],
            alpha=best_bundle["alpha"],
            learning_rate_init=best_bundle["learning_rate_init"],
            batch_size=best_bundle["batch_size"],
            max_iter=best_bundle["max_iter"],
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=seed,
        )),
    ])
    final_model.fit(pd.concat([loaded.X_train, loaded.X_val]), pd.concat([loaded.y_train, loaded.y_val]))
    y_prob = final_model.predict_proba(loaded.X_test)[:, 1]
    y_pred = threshold_predictions(y_prob)
    metrics = classification_summary(loaded.y_test, y_pred, y_prob)
    metrics.update({
        "seed": seed,
        "model_name": "mlp",
        "dataset_name": args.dataset_name,
        "cv_scoring": args.cv_scoring,
        "best_candidate": {
            "scaler": "quantile" if best_bundle["prep"].named_steps["scaler"].__class__.__name__ == "QuantileTransformer" else "standard",
            "hidden_layer_sizes": list(best_bundle["hidden_layer_sizes"]),
            "alpha": best_bundle["alpha"],
            "learning_rate_init": best_bundle["learning_rate_init"],
            "batch_size": best_bundle["batch_size"],
            "max_iter": best_bundle["max_iter"],
        },
        "best_cv_score": best_score,
    })

    config_path = run_dir / "config_used.json"
    metrics_path = run_dir / "metrics_summary.json"
    preds_path = run_dir / "predictions.csv"
    cv_path = run_dir / "cv_results.csv"
    log_path = run_dir / "raw_log.txt"

    serializable_candidates = [
        {
            "scaler": "quantile" if c["prep"].named_steps["scaler"].__class__.__name__ == "QuantileTransformer" else "standard",
            "hidden_layer_sizes": list(c["hidden_layer_sizes"]),
            "alpha": c["alpha"],
            "learning_rate_init": c["learning_rate_init"],
            "batch_size": c["batch_size"],
            "max_iter": c["max_iter"],
        }
        for c in candidates
    ]
    write_json({
        "dataset_zip": str(args.dataset_zip),
        "split_json": str(args.split_json),
        "dataset_name": args.dataset_name,
        "cv_scoring": args.cv_scoring,
        "candidate_params": serializable_candidates,
    }, config_path)
    write_json(metrics, metrics_path)
    write_predictions(preds_path, loaded.test_df, y_pred, y_prob)
    pd.DataFrame(cv_rows).to_csv(cv_path, index=False)
    logger.dump(log_path)
    write_artifacts_manifest(run_dir, [config_path, metrics_path, preds_path, cv_path, log_path])


if __name__ == "__main__":
    main()
