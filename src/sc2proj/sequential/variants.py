"""S6: fail-closed identity gate for v3.1 vs v3.2 temporal experiments."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import IndexedSequences

V31_SHA256 = "65e402884f43c03bf5450d6975e91db3d61ee7e28edf1a56483a37f15d944517"
V32_SHA256 = "87ce36435e4a0fc87b8d9421291d3b1fb04d67b5187ede6a2a3b25d0dce1942e"

ROW_COLS = ("replay_id", "time_sec", "p1_wins", "source_row")
WINDOW_COLS = ("start_row", "stop_row", "replay_id", "time_sec", "p1_wins")


def _matching_metadata(a: pd.DataFrame, b: pd.DataFrame, columns: tuple[str, ...],
                       label: str) -> None:
    if not set(columns).issubset(a.columns) or not set(columns).issubset(b.columns):
        raise ValueError(f"S6 {label}: missing metadata fields")
    if len(a) != len(b):
        raise ValueError(f"S6 {label}: unequal number of rows")
    # Exact (not fuzzy or inner-joined) identity in physical S1 order.
    for col in columns:
        if not np.array_equal(a[col].to_numpy(), b[col].to_numpy()):
            raise ValueError(f"S6 {label}: {col} mismatch; refusing unpaired comparison")


def verify_combatfix_parity(reference: IndexedSequences, candidate: IndexedSequences,
                            *, require_full: bool = True) -> dict:
    """Prove both variants have identical replay/time/label/windows for comparison."""
    ref = reference.manifest
    cand = candidate.manifest
    if ref.get("source_zip_sha256") != V31_SHA256 or cand.get("source_zip_sha256") != V32_SHA256:
        raise ValueError("S6 frozen dataset SHA-256 mismatch")
    if int(ref.get("window_length", -1)) != 9 or int(cand.get("window_length", -1)) != 9:
        raise ValueError("S6 requires matched 9-snapshot windows")
    if int(ref.get("step_sec", -1)) != 15 or int(cand.get("step_sec", -1)) != 15:
        raise ValueError("S6 requires identical 15-second grid")
    if int(ref.get("stride", -1)) != 1 or int(cand.get("stride", -1)) != 1:
        raise ValueError("S6 requires identical stride 1")
    if len(reference.features) != 64 or len(candidate.features) != 73:
        raise ValueError("S6 feature counts must be 64 and 73")
    if not set(reference.features).issubset(candidate.features) or len(
            set(candidate.features) - set(reference.features)) != 9:
        raise ValueError("S6 combatfix must preserve all 64 predictor names plus 9 added features")
    if cand.get("dataset_variant") != "v3_2_combatfix":
        raise ValueError("S6 expected a combatfix S1 index manifest")
    if require_full:
        if any(x.manifest.get("subset_max_replays") is not None for x in (reference, candidate)):
            raise ValueError("S6 full comparison refuses smoke/subset indexes")
        if (len(reference.ordered_rows), len(reference.windows),
            len(candidate.ordered_rows), len(candidate.windows)) != (
                892047, 706952, 892047, 706952):
            raise ValueError("S6 full index must contain 892047 rows and 706952 windows")
    _matching_metadata(reference.ordered_rows, candidate.ordered_rows, ROW_COLS, "ordered rows")
    _matching_metadata(reference.windows, candidate.windows, WINDOW_COLS, "sequence endpoints")
    return {
        "status": "S6_DATASET_PARITY_PASS",
        "reference_dataset": "v3_1_fixed",
        "candidate_dataset": "v3_2_combatfix",
        "rows": len(candidate.ordered_rows),
        "windows": len(candidate.windows),
        "reference_features": len(reference.features),
        "candidate_features": len(candidate.features),
        "v3_1_sha256": V31_SHA256,
        "v3_2_sha256": V32_SHA256,
    }
