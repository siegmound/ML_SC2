from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sc2proj.utils import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle-zip', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'results' / 'real_calibration_import')
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    imported = []
    with zipfile.ZipFile(args.bundle_zip, 'r') as zf:
        for name in zf.namelist():
            target = args.output_dir / Path(name).name
            ensure_dir(target.parent)
            with zf.open(name) as src, target.open('wb') as dst:
                shutil.copyfileobj(src, dst)
            imported.append(target.name)
    write_json({'source_bundle': str(args.bundle_zip), 'imported_files': imported, 'n_imported': len(imported)}, args.output_dir / 'import_manifest.json')
    print({'n_imported': len(imported), 'output_dir': str(args.output_dir)})


if __name__ == '__main__':
    main()
