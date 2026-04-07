from __future__ import annotations

from collections import defaultdict

DEFAULT_FAMILIES = [
    'economy',
    'combat',
    'scouting',
    'composition',
    'losses',
    'delta_15s',
    'trend_60_120',
    'counter',
    'metadata',
    'other',
]


def infer_feature_family(column: str) -> str:
    col = column.lower()
    if col in {'matchup', 'race_matchup', 'race_pair', 'league', 'map_name'}:
        return 'metadata'
    if 'counter' in col or 'synergy' in col:
        return 'counter'
    if 'delta_' in col:
        return 'delta_15s'
    if 'trend' in col or 'rolling' in col or 'std_' in col:
        return 'trend_60_120'
    if 'scout' in col or 'camera' in col:
        return 'scouting'
    if 'entropy' in col or 'unit_types' in col or 'supply_ratio' in col or 'tech' in col:
        return 'composition'
    if 'loss' in col or 'recent_losses' in col:
        return 'losses'
    if 'combat' in col or 'army' in col:
        return 'combat'
    if any(tok in col for tok in ['worker', 'sq', 'epm', 'income', 'mineral', 'vespene', 'resource']):
        return 'economy'
    return 'other'


def build_feature_family_map(feature_columns: list[str]) -> dict[str, list[str]]:
    families = defaultdict(list)
    for col in feature_columns:
        families[infer_feature_family(col)].append(col)
    return {k: sorted(v) for k, v in sorted(families.items())}


def select_features_by_family(feature_columns: list[str], include_families: list[str] | None = None, exclude_families: list[str] | None = None) -> list[str]:
    include_families = include_families or []
    exclude_families = exclude_families or []
    fam_map = build_feature_family_map(feature_columns)
    if include_families:
        selected = []
        for fam in include_families:
            selected.extend(fam_map.get(fam, []))
    else:
        selected = list(feature_columns)
    if exclude_families:
        excluded = set()
        for fam in exclude_families:
            excluded.update(fam_map.get(fam, []))
        selected = [c for c in selected if c not in excluded]
    return selected
