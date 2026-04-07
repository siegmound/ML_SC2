import argparse
import json
from pathlib import Path

REQUIRED = [
    'scripts/53_run_xgb_full_gpu_fixed_v3.py',
    'scripts/54_run_rf_full.py',
    'scripts/55_collect_block16_full_comparison.py',
    'scripts/56_make_block16_freeze_report.py',
    'results/xgb_full/real_v3_1_fixed_fullgpu_clean/real_v3_1_fixed_fullgpu_clean.zip',
    'results/xgb_full/real_v3_2_combatfix_fullgpu_clean/xgbreal_v3_2_combatfix_fullgpu_clean.zip',
    'results/rf_full/real_v3_1_fixed_fullrf_clean/real_v3_1_fixed_fullrf_clean.zip',
    'results/rf_full/real_v3_2_combatfix_fullrf_clean/rfreal_v3_2_combatfix_fullrf_clean.zip',
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo-root', default='.')
    args = ap.parse_args()
    repo = Path(args.repo_root)
    missing = [p for p in REQUIRED if not (repo / p).exists()]
    status = {
        'ok': len(missing) == 0,
        'missing': missing,
        'required_count': len(REQUIRED),
        'present_count': len(REQUIRED) - len(missing),
    }
    out = repo / 'results/block16_final/block16_verification.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(status, f, indent=2)
    print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
