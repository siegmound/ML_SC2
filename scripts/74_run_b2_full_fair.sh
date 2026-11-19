#!/usr/bin/env bash
# B2 FAIR full-scale replay-aware re-evaluation, isolated from historical runs.
# Usage:
#   bash scripts/74_run_b2_full_fair.sh --plan [--seed 42] [--model rf]
#   bash scripts/74_run_b2_full_fair.sh --preflight
#   bash scripts/74_run_b2_full_fair.sh --execute [--seed 42] [--model rf]
#   bash scripts/74_run_b2_full_fair.sh --verify [--seed 42] [--model rf]
# --execute is opt-in and saves to NEW output roots; --verify is read-only.
set -euo pipefail

usage() {
  echo "Usage: bash scripts/74_run_b2_full_fair.sh --plan|--preflight|--execute|--verify [--seed 42|43|44] [--model rf|xgb|deep]" >&2
}
mode="--plan"
seed_filter=""
model_filter=""
while (($#)); do
  case "$1" in
    --plan|--preflight|--execute|--verify)
      mode="$1"; shift ;;
    --seed)
      (($# >= 2)) || { usage; exit 2; }
      seed_filter="$2"; shift 2
      [[ "$seed_filter" =~ ^(42|43|44)$ ]] || { usage; exit 2; } ;;
    --model)
      (($# >= 2)) || { usage; exit 2; }
      model_filter="$2"; shift 2
      [[ "$model_filter" =~ ^(rf|xgb|deep)$ ]] || { usage; exit 2; } ;;
    *)
      usage; exit 2 ;;
  esac
done

root="$(git rev-parse --show-toplevel)"
cd "$root"
expected_branch="audit/sc2-corrections-2026-11-17"
branch="$(git branch --show-current)"
[[ "$branch" == "$expected_branch" ]] || {
  echo "Refusing B2 from branch $branch (expected $expected_branch)" >&2; exit 2;
}

py="${SC2_PYTHON:-$HOME/.venvs/ml-sc2-b4-offline/bin/python}"
dataset="data/processed/starcraft_full_dataset_v3_1_fixed.zip"
dataset_name="real_v3_1_fixed"
out_base="${B2_OUT_BASE:-results/b2_fullscale_audit_20261119}"
threads="${B2_THREADS:-8}"
gpu="${B2_GPU:-0}"
[[ "$threads" =~ ^[1-9][0-9]*$ ]] || { echo "B2_THREADS must be a positive integer" >&2; exit 2; }
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo "B2_GPU must be a non-negative integer" >&2; exit 2; }
[[ -x "$py" ]] || { echo "Python not found: $py; set SC2_PYTHON to the locked FAIR interpreter" >&2; exit 2; }
[[ -f "$dataset" ]] || { echo "Missing frozen LFS ZIP: $dataset" >&2; exit 2; }
[[ "$(head -c 7 "$dataset")" != "version" ]] || {
  echo "Dataset is an LFS pointer; pull full payload first" >&2; exit 2;
}
for seed in 42 43 44; do
  [[ -f "data/processed/splits/real_v3_1_fixed_split_seed_${seed}.json" ]] || {
    echo "Missing frozen split: $seed" >&2; exit 2;
  }
done

# Prevent accidental redirection into any committed historical result path.
case "$out_base/" in
  results/b2_fullscale_audit_*/*) ;;
  *) echo "B2_OUT_BASE must be a new results/b2_fullscale_audit_* directory" >&2; exit 2 ;;
esac
if [[ "$mode" == "--execute" || "$mode" == "--preflight" ]]; then
  [[ -d results/rf_calibration/real_v3_1_fixed &&
     -d results/xgb_calibration/real_v3_1_fixed ]] || {
    echo "Missing frozen retrospective RF/XGB evidence" >&2; exit 2;
  }
fi

export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1
echo "B2 mode=$mode branch=$branch python=$py"
echo "Output root: $out_base | threads=$threads | visible GPU=$gpu"

if [[ "$mode" == "--preflight" || "$mode" == "--execute" ]]; then
  "$py" -c 'import sys, torch, sklearn, xgboost, pandas; assert sys.version_info[:2] == (3, 12), sys.version; assert torch.cuda.is_available(), "CUDA required"; print("CUDA:", torch.cuda.get_device_name(0), "torch", torch.__version__, "xgboost", xgboost.__version__)'
  "$py" scripts/70_verify_freeze.py --mode payload
  "$py" scripts/66_check_matched_train_budget.py --check
  echo "PASS: B2 FAIR preflight (CUDA, frozen data, retrospective evidence)"
fi
if [[ "$mode" == "--preflight" ]]; then
  exit 0
fi

seeds=(42 43 44)
models=(rf xgb deep)
[[ -z "$seed_filter" ]] || seeds=("$seed_filter")
[[ -z "$model_filter" ]] || models=("$model_filter")

verify_run() {
  local seed="$1" model="$2"
  "$py" scripts/75_verify_b2_full_fair.py \
    --repo-root "$root" --output-base "$out_base" --seed "$seed" --model "$model"
}
if [[ "$mode" == "--verify" ]]; then
  "$py" scripts/75_verify_b2_full_fair.py --repo-root "$root" --output-base "$out_base" \
    ${seed_filter:+--seed "$seed_filter"} ${model_filter:+--model "$model_filter"}
  exit 0
fi

if [[ "$mode" == "--execute" ]]; then
  mkdir -p "$out_base/logs"
fi
for seed in "${seeds[@]}"; do
  split="data/processed/splits/real_v3_1_fixed_split_seed_${seed}.json"
  for model in "${models[@]}"; do
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
        --n-jobs "$threads" --output-root "$out_base/strict_calibration")
      [[ "$model" != "xgb" ]] || command+=(--device cuda)
    fi
    echo "=== B2 $model seed_$seed ==="
    if [[ "$mode" == "--plan" ]]; then
      printf ' '; printf '%q ' "${command[@]}"; echo
      continue
    fi
    if [[ -e "$run_dir" ]]; then
      # A prior output can be skipped ONLY if all its predictions, original
      # replay groups, metrics and fitted-run metadata pass fresh validation.
      verify_run "$seed" "$model" || {
        echo "Existing but incomplete/invalid output: $run_dir" >&2
        echo "Do not overwrite it; inspect logs and use a NEW B2_OUT_BASE." >&2
        exit 2
      }
      echo "SKIP existing VERIFIED $model seed_$seed"
      continue
    fi
    log="$out_base/logs/${model}_seed_${seed}.log"
    [[ ! -e "$log" ]] || {
      echo "Log already exists without valid run: $log; choose a new B2_OUT_BASE" >&2
      exit 2
    }
    echo "Running; log: $log"
    if "${command[@]}" >"$log" 2>&1; then
      verify_run "$seed" "$model" || {
        echo "B2 output failed post-run verification: $run_dir" >&2
        exit 1
      }
      echo "PASS: $model seed_$seed"; tail -n 4 "$log"
    else
      status=$?
      echo "FAIL: $model seed_$seed (exit $status), log $log" >&2
      tail -n 30 "$log" >&2
      exit "$status"
    fi
  done
done
echo "B2 selected runs complete. After all 9, run: bash scripts/74_run_b2_full_fair.sh --verify"
