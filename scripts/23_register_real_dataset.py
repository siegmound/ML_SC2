from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sc2proj.real_data import build_real_dataset_manifest, write_manifest_pair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-zip', type=Path, required=True)
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--dataset-version', type=str, default='v1')
    parser.add_argument('--notes', type=str, default='')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'data' / 'processed' / 'manifests')
    args = parser.parse_args()

    manifest = build_real_dataset_manifest(args.dataset_zip, args.dataset_name, args.dataset_version, notes=args.notes)
    dataset_path, experiment_path = write_manifest_pair(manifest, args.output_dir)
    print({'dataset_manifest': str(dataset_path), 'experiment_manifest': str(experiment_path), 'n_rows': manifest['n_rows'], 'n_replays': manifest['n_replays']})


if __name__ == '__main__':
    main()
