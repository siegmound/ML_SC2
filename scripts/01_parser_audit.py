from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd

from sc2proj.bridge import ReplayBridge
from sc2proj.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replays-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "experiments" / "parser_audit")
    args = parser.parse_args()

    bridge = ReplayBridge()
    rows = []
    reason_counter = Counter()

    replay_paths = sorted(args.replays_dir.rglob("*.SC2Replay"))
    for replay_path in replay_paths:
        result = bridge.process_replay(replay_path)
        rows.append(result.metadata)
        reason = result.metadata.get("failure_reason") or "ok"
        reason_counter[reason] += 1

    df = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_dir / "replay_audit.csv", index=False)
    pd.DataFrame(sorted(reason_counter.items()), columns=["reason", "count"]).to_csv(args.output_dir / "failure_reasons.csv", index=False)
    summary = {
        "n_replays": int(len(df)),
        "n_ok": int((df["parse_status"] == "ok").sum()) if not df.empty else 0,
        "total_unknown_unit_count": int(df["unknown_unit_count"].fillna(0).sum()) if not df.empty else 0,
        "total_alias_unit_count": int(df["alias_unit_count"].fillna(0).sum()) if not df.empty else 0,
        "reason_counts": dict(reason_counter),
    }
    write_json(summary, args.output_dir / "parser_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
