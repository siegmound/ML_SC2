from __future__ import annotations

import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sc2proj.bridge import ReplayBridge
from sc2proj.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replays-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "experiments" / "parser_audit")
    args = parser.parse_args()

    bridge = ReplayBridge()
    rows = []
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(args.replays_zip, "r") as zf:
            zf.extractall(tmpdir)
        replay_paths = sorted(Path(tmpdir).rglob("*.SC2Replay"))
        for replay_path in replay_paths:
            result = bridge.process_replay(replay_path)
            rows.append(result.metadata)

    ok_count = sum(1 for row in rows if row["parse_status"] == "ok")
    report = {
        "n_replays": len(rows),
        "n_ok": ok_count,
        "success_rate": (ok_count / len(rows)) if rows else 0.0,
        "acceptance_passed": (ok_count / len(rows)) >= 0.95 if rows else False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(args.output_dir / "smoke_test_rows.csv", index=False)
    write_json(report, args.output_dir / "smoke_test_report.json")
    print(report)


if __name__ == "__main__":
    main()
