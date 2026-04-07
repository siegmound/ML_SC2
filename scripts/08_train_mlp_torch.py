from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sc2proj.deep_models import default_torch_candidates, fit_torch_candidate, predict_torch_model
from sc2proj.metrics import classification_summary
from sc2proj.modeling import threshold_predictions
from sc2proj.training_io import RunLogger, load_split_manifest, make_run_dir, select_split_frames, write_artifacts_manifest, write_predictions
from sc2proj.utils import load_dataframe_from_zip, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-zip", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--dataset-name", type=str, default="canonical")
    parser.add_argument("--selection-metric", choices=["neg_log_loss", "roc_auc", "accuracy"], default="neg_log_loss")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    split = load_split_manifest(args.split_json)
    seed = int(split["seed"])
    run_dir = make_run_dir(args.output_dir, "mlp_torch", args.dataset_name, seed)
    logger = RunLogger()
    df = load_dataframe_from_zip(args.dataset_zip)
    loaded = select_split_frames(df, split)

    candidates = default_torch_candidates()
    rows = []
    best_bundle = None
    best_candidate = None
    best_score = None

    logger.log(f"Running strong deep branch selection on validation split with metric={args.selection_metric} device={args.device}")
    for idx, candidate in enumerate(candidates, start=1):
        logger.log(f"Candidate {idx}/{len(candidates)}: {candidate.to_dict()}")
        bundle = fit_torch_candidate(loaded.X_train, loaded.y_train, loaded.X_val, loaded.y_val, candidate, seed=seed, device=args.device)
        y_prob_val = bundle["val_prob"]
        y_pred_val = threshold_predictions(y_prob_val)
        if args.selection_metric == "accuracy":
            score = float((loaded.y_val.to_numpy() == y_pred_val).mean())
        elif args.selection_metric == "roc_auc":
            score = float(roc_auc_score(loaded.y_val, y_prob_val))
        else:
            score = float(-log_loss(loaded.y_val, y_prob_val, labels=[0, 1]))
        rows.append({
            **candidate.to_dict(),
            "selection_metric": args.selection_metric,
            "score": score,
            "best_epoch": bundle["best_epoch"],
            "best_val_loss": bundle["best_val_loss"],
        })
        if best_score is None or score > best_score:
            best_score = score
            best_bundle = bundle
            best_candidate = candidate

    assert best_bundle is not None and best_candidate is not None and best_score is not None
    logger.log(f"Best deep candidate: {best_candidate.to_dict()}")
    logger.log(f"Best validation score: {best_score:.6f}")

    y_prob = predict_torch_model(best_bundle, loaded.X_test)
    y_pred = threshold_predictions(y_prob)
    metrics = classification_summary(loaded.y_test, y_pred, y_prob)
    metrics.update({
        "seed": seed,
        "model_name": "mlp_torch",
        "dataset_name": args.dataset_name,
        "selection_metric": args.selection_metric,
        "best_candidate": best_candidate.to_dict(),
        "best_validation_score": best_score,
        "best_epoch": int(best_bundle["best_epoch"]),
        "best_val_loss": float(best_bundle["best_val_loss"]),
        "device": best_bundle["device"],
    })

    config_path = run_dir / "config_used.json"
    metrics_path = run_dir / "metrics_summary.json"
    preds_path = run_dir / "predictions.csv"
    search_path = run_dir / "validation_search_results.csv"
    history_path = run_dir / "best_training_history.csv"
    log_path = run_dir / "raw_log.txt"

    write_json({
        "dataset_zip": str(args.dataset_zip),
        "split_json": str(args.split_json),
        "dataset_name": args.dataset_name,
        "selection_metric": args.selection_metric,
        "device": args.device,
        "candidates": [c.to_dict() for c in candidates],
    }, config_path)
    write_json(metrics, metrics_path)
    write_predictions(preds_path, loaded.test_df, y_pred, y_prob)
    pd.DataFrame(rows).to_csv(search_path, index=False)
    pd.DataFrame(best_bundle["history"]).to_csv(history_path, index=False)
    logger.dump(log_path)
    write_artifacts_manifest(run_dir, [config_path, metrics_path, preds_path, search_path, history_path, log_path])


if __name__ == "__main__":
    main()
