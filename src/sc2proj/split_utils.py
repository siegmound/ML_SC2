from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

@dataclass
class SplitConfig:
    seed: int
    test_size: float = 0.2
    val_size_within_train: float = 0.2


def make_group_split(df: pd.DataFrame, group_col: str, target_col: str, config: SplitConfig) -> Dict[str, List[str]]:
    groups = df[group_col].astype(str)
    y = df[target_col]
    outer = GroupShuffleSplit(n_splits=1, test_size=config.test_size, random_state=config.seed)
    train_val_idx, test_idx = next(outer.split(df, y, groups=groups))

    train_val = df.iloc[train_val_idx]
    groups_train_val = train_val[group_col].astype(str)
    y_train_val = train_val[target_col]
    inner = GroupShuffleSplit(n_splits=1, test_size=config.val_size_within_train, random_state=config.seed)
    train_rel, val_rel = next(inner.split(train_val, y_train_val, groups=groups_train_val))

    train_groups = sorted(train_val.iloc[train_rel][group_col].astype(str).unique().tolist())
    val_groups = sorted(train_val.iloc[val_rel][group_col].astype(str).unique().tolist())
    test_groups = sorted(df.iloc[test_idx][group_col].astype(str).unique().tolist())

    overlap = set(train_groups) & set(val_groups) | set(train_groups) & set(test_groups) | set(val_groups) & set(test_groups)
    if overlap:
        raise ValueError(f"Replay overlap detected across splits: {sorted(overlap)[:10]}")

    return {
        "seed": config.seed,
        "train_groups": train_groups,
        "val_groups": val_groups,
        "test_groups": test_groups,
        "n_train_groups": len(train_groups),
        "n_val_groups": len(val_groups),
        "n_test_groups": len(test_groups),
    }
