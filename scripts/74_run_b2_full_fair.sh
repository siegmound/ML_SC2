#!/usr/bin/env bash
# B2 full-size three-seed rerun; never touches historical result locations.
# Usage: bash scripts/74_run_b2_full_fair.sh --plan|--execute
set -euo pipefail

mode="${1:---plan}"
if [[ "${mode}" != "--plan" && "${mode}" != "--execute" ]]; then
  echo "Usage: bash scripts/74_run_b2_full_fair.sh --plan|--execute" >&2
  exit 2
fi

root="$(git rev-parse --show-toplevel)"
cd "$root"
branch="$(git branch --show-current)"
if [[ "$branch" != "audit/sc2-corrections-2026-11-17" ]]; then
  echo "Refusing to run B2 from branch $branch (expected audit correction branch)" >&2
  exit 2
fi

py="${SC2_PYTHON:-$HOME/.venvs/ml-sc2/bin/python}"
dataset="data/processed/starcraft_full_dataset_v3_1_fixed.zip"
dataset_name="real_v3_1_fixed"
out_base="${B2_OUT_BASE:-results/b2_fullscale_audit_20261119}"
threads="${B2_THREADS:-8}"
gpu="${B2_GPU:-0}"

[[ -x "$py" ]] || { echo "Python executable not found: $py" >&2; exit 2; }
[[ -f "$dataset" ]] || { echo "Missing LFS payload: $dataset" >&2; exit 2; }
[[ "$(head -c 7 "$dataset")" != "version" ]] || {
  echo "Dataset still appears to be a Git LFS pointer, not ZIP payload" >&2; exit 2;
}
for seed in 42 43 44; do
  [[ -f "data/processed/splits/real_v3_1_fixed_split_seed_${seed}.json" ]] || {
    echo "Missing split for seed $seed" >&2; exit 2;
  }
done
[[ -d results/rf_calibration/real_v3_1_fixed ]] || {
  echo "Missing original RF reference runs; refusing to continue" >&2; exit 2;
}
[[ -d results/xgb_calibration/real_v3_1_fixed ]] || {
  echo "Missing original XGB reference runs; refusing to continue" >&2; exit 2;
}

export CUDA_VISIBLE_DEVICES="$gpu"
echo "B2 mode=$mode | branch=$branch | Python=$py"
echo "Output namespace: $out_base | CPU threads=$threads | CUDA_VISIBLE_DEVICES=$gpu"
if [[ "$mode" == "--execute" ]]; then
  "$py" -c 'import torch, sklearn, xgboost; assert torch.cuda.is_available(), "CUDA required"; print("CUDA:", torch.cuda.get_device_name(0))'
  "$py" scripts/70_verify_freeze.py --mode payload
  "$py" scripts/66_check_matched_train_budget.py --check
  mkdir -p "$out_base/logs"
fi

for seed in 42 43 44; do
  split="data/processed/splits/real_v3_1_fixed_split_seed_${seed}.json"
  for model in rf xgb deep; do
    if [[ "$model" == "deep" ]]; then
      run_dir="$out_base/deep_finalist/$dataset_name/seed_$seed"
      command=("$py" scripts/64_run_deep_finalist.py
        --dataset-zip "$dataset" --split-json "$split"
        --dataset-name "$dataset_name" --device cuda
        --output-root "$out_base/deep_finalist")
    else
      run_dir="$out_base/strict_calibration/$dataset_name/$model/seed_$seed"
      command=("$py" scripts/67_run_strict_calibration.py
        --dataset-zip "$dataset" --split-json "$split"
        --dataset-name "$dataset_name" --model "$model"
        --n-jobs "$threads"
        --output-root "$out_base/strict_calibration")
      if [[ "$model" == "xgb" ]]; then command+=(--device cuda); fi
    fi
    echo "=== B2 $model seed_$seed ==="
    if [[ "$mode" == "--plan" ]]; then
      printf ' '; printf '%q ' "${command[@]}"; echo
      continue
    fi
    if [[ -e "$run_dir" ]]; then
      if [[ "$model" == "deep" ]]; then
        complete=0
        [[ -s "$run_dir/metrics_summary.json" &&
           -s "$run_dir/validation_search_results.csv" &&
           -s "$run_dir/artifacts_manifest.json" &&
           -s "$run_dir/test_predictions.csv" &&
           -s "$run_dir/selected_model_state.pt" ]] && complete=1
      else
        complete=0
        [[ -s "$run_dir/metrics_summary.json" &&
           -s "$run_dir/split_manifest.json" &&
           -s "$run_dir/test_predictions_uncalibrated.csv" &&
           -s "$run_dir/test_predictions_platt.csv" &&
           -s "$run_dir/test_predictions_isotonic.csv" ]] && complete=1
      fi
      if [[ "$complete" == 1 ]]; then
        echo "Existing summary present; SKIP (do not replace historical/new outputs): $run_dir"
        continue
      fi
      echo "INCOMPLETE output directory; refusing overwrite: $run_dir" >&2
      exit 2
    fi
    log="$out_base/logs/${model}_seed_${seed}.log"
    echo "Run log: $log"
    if "${command[@]}" >"$log" 2>&1; then
      [[ -s "$run_dir/metrics_summary.json" ]] || {
        echo "Missing model summary after successful process: $run_dir" >&2; exit 2;
      }
      echo "PASS: $model seed_$seed"
      tail -n 5 "$log"
    else
      rc="$?"
      echo "FAIL: $model seed_$seed (exit $rc); inspect $log" >&2
      tail -n 30 "$log" >&2
      exit "$rc"
    fi
  done
done
echo "B2 batch complete; review all nine summaries before interpreting test metrics."
