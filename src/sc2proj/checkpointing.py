
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .utils import ensure_dir


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def completed_keys(path: Path, key_field: str = "candidate_key") -> set[str]:
    return {str(row[key_field]) for row in load_jsonl(path) if key_field in row}
