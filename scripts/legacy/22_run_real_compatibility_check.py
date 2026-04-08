from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze-zip', type=Path, required=True)
    parser.add_argument('--dataset-zip', type=Path, default=None)
    args = parser.parse_args()

    py = sys.executable
    _run([py, 'scripts/18_probe_legacy_freeze.py', '--freeze-zip', str(args.freeze_zip)])
    _run([py, 'scripts/19_import_legacy_freeze.py', '--freeze-zip', str(args.freeze_zip)])
    _run([py, 'scripts/20_build_real_artifact_manifests.py'])
    cmd = [py, 'scripts/21_validate_real_schema.py']
    if args.dataset_zip is not None:
        cmd += ['--dataset-zip', str(args.dataset_zip)]
    _run(cmd)
    print({'status': 'ok', 'freeze_zip': str(args.freeze_zip), 'dataset_zip': str(args.dataset_zip) if args.dataset_zip else None})


if __name__ == '__main__':
    main()
