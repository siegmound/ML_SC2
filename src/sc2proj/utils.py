from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(data: Any, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def load_dataframe_from_zip(zip_path: Path):
    import pandas as pd

    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_members = [name for name in zf.namelist() if name.endswith(".csv")]
        if len(csv_members) != 1:
            raise ValueError(f"Expected exactly one CSV in {zip_path}, found {csv_members}")
        with zf.open(csv_members[0]) as fh:
            return pd.read_csv(fh)


def dump_dataframe_to_zip(df, zip_path: Path, inner_name: str) -> None:
    import io

    ensure_dir(zip_path.parent)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(inner_name, csv_buffer.getvalue().encode("utf-8"))
