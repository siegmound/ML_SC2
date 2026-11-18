#!/usr/bin/env python3
"""S7 ablation: unchanged S4 GRU on the corrected 64 shared combatfix features.

This differs from S4 only in the VALUES of the shared 64 feature columns;
same ordered 9-window endpoints, split manifests, architecture and optimizer.
S6 full-73 GRU artifacts are never modified. No test metrics are computed.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

from sc2proj.sequential.data import load_index, load_replay_partitions, prepare_sequential_data
from sc2proj.sequential.gru import GRUConfig, fit_gru, validation_predictions
from sc2proj.sequential.variants import verify_combatfix_parity

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/sequential"


def make_common64_view(reference, combatfix):
    """In-memory feature view; keep combatfix row offsets and its source hash.

    The full 73-feature S1 index is validated first. Do not write a synthetic
    index or misrepresent it as the original 64-feature v3.1 ZIP.
    """
    names = tuple(reference.features)
    if len(names) != 64 or len(combatfix.features) != 73:
        raise ValueError("S7 expects 64 reference and 73 combatfix features")
    if not set(names).issubset(combatfix.features) or len(set(combatfix.features) - set(names)) != 9:
        raise ValueError("S7 requires exactly nine extra combatfix predictors")
    metadata = dict(combatfix.manifest)
    metadata["features"] = list(names)
    metadata["feature_count"] = len(names)
    metadata["feature_view"] = "v3_2_common64"
    return replace(combatfix, features=names, manifest=metadata)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-index-dir", type=Path, default=RESULTS / "v3_1_fixed_w9_s1")
    ap.add_argument("--index-dir", type=Path, default=RESULTS / "v3_2_combatfix_w9_s1")
    ap.add_argument("--dataset-zip", type=Path,
                    default=ROOT / "data/processed/starcraft_full_dataset_v3_2_combatfix.zip")
    ap.add_argument("--split-json", type=Path,
                    default=ROOT / "data/processed/splits/real_v3_1_fixed_split_seed_42.json")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--smoke", action="store_true", help="Two batches and at most two epochs, no output")
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    if args.smoke and args.output_dir is not None:
        ap.error("--smoke does not write artifacts")
    if not args.smoke and args.output_dir is None:
        ap.error("Full training needs a fresh --output-dir")
    output = None
    if args.output_dir is not None:
        output = args.output_dir.resolve()
        base = RESULTS.resolve()
        if output == base or not output.is_relative_to(base) or output.exists():
            ap.error("Output directory must be fresh and under results/sequential/")
    if args.device == "cuda" and not torch.cuda.is_available():
        ap.error("Requested CUDA but torch.cuda.is_available() is false")
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )

    reference = load_index(args.reference_index_dir)
    combatfix = load_index(args.index_dir)
    parity = verify_combatfix_parity(reference, combatfix, require_full=True)
    view = make_common64_view(reference, combatfix)
    split_reference = load_replay_partitions(args.split_json, reference)
    split_combatfix = load_replay_partitions(args.split_json, combatfix)
    if split_reference.seed != split_combatfix.seed:
        raise ValueError("S7 seed mismatch")
    for part in ("train", "val", "test"):
        if not (split_reference.window_positions[part] == split_combatfix.window_positions[part]).all():
            raise ValueError(f"S7 {part} split positions differ")
        if not (split_reference.row_masks[part] == split_combatfix.row_masks[part]).all():
            raise ValueError(f"S7 {part} split rows differ")
    print(json.dumps(parity, sort_keys=True), flush=True)

    config = GRUConfig(batch_size=32) if args.smoke else GRUConfig()
    # The v3.2 ZIP is read (with frozen hash), selecting only 64 common columns.
    prepared = prepare_sequential_data(view, split_combatfix, args.dataset_zip, verify_sha=True)
    model, report = fit_gru(prepared, config, device, smoke=args.smoke, verbose=True)
    report.update({
        "status": "S7_SMOKE_PASS" if args.smoke else "S7_FIT_PASS",
        "dataset": "v3_2_combatfix_common64",
        "feature_count": len(view.features),
        "feature_names": list(view.features),
        "feature_view": "v3_2_common64",
        "window_length": 9,
        "source_zip_sha256": view.manifest["source_zip_sha256"],
        "reference_zip_sha256": reference.manifest["source_zip_sha256"],
        "parity": parity,
        "index_dir": str(args.index_dir.resolve()),
        "reference_index_dir": str(args.reference_index_dir.resolve()),
        "split_json": str(args.split_json.resolve()),
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "caveat": "Exploratory validation-only ablation; cannot establish unbiased test performance.",
    })
    if args.smoke:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    pred, metrics = validation_predictions(model, prepared, view, split_combatfix,
                                           device, config.batch_size)
    report["validation_metrics"] = metrics
    report["status"] = "S7_VALIDATION_PASS"
    assert output is not None
    output.mkdir(parents=True, exist_ok=False)
    torch.save({
        "state_dict": model.state_dict(),
        "feature_count": len(view.features),
        "feature_names": list(view.features),
        "feature_view": "v3_2_common64",
        "seed": split_combatfix.seed,
        "window_length": 9,
        "selected_epoch": report["selected_epoch"],
        "model_config": report["config"],
        "source_zip_sha256": report["source_zip_sha256"],
    }, output / "best_model.pt")
    pred.to_csv(output / "validation_predictions.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Saved {len(pred):,} matched common64 validation endpoints to {output}")
    print("No test predictions or metrics were generated.")


if __name__ == "__main__":
    main()
