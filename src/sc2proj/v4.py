"""SC2 v4 clean-room parser semantics; not a claim of frozen-v3 lineage.

The matchup dictionary records unit -> unit types that counter THAT unit.
Combat penalty uses it directly; own counter advantage must use its inverse.
Unit lifecycle de-duplicates UnitInit/UnitBorn for the same SC2 unit_id,
handles type morphs and unit transfers, and records *current* inventories.
Formula coefficients are prespecified project assumptions, never fitted
against the previously inspected frozen test values.
"""
from __future__ import annotations

from typing import Dict, Iterable

from .bridge import ReplayBridge
from .constants import UNIT_ALIASES, UNIT_SPECS
from .counters import UNIT_COUNTERS
from .parser import SnapshotParser


def counters_effective_against(unit_name: str) -> frozenset[str]:
    """Countered target -> listed effective opposing unit types."""
    return frozenset(UNIT_COUNTERS.get(unit_name, ()))


def own_units_counter_enemy(own_type: str, enemy_type: str) -> bool:
    """True iff own_type occurs as an effective counter for enemy_type."""
    return own_type in counters_effective_against(enemy_type)


class V4SnapshotParser(SnapshotParser):
    def _counter_advantage(self, own_army: Dict[str, int],
                           enemy_army: Dict[str, int]) -> float:
        """Directional fraction of enemy army effectively countered by own.

        Keeps the historical symbolic weight equation but fixes which
        *side* of the qualitative matchup relation is consulted.
        Not game-engine-calibrated; see v4 documentation for caveats.
        """
        total_own = sum(n for n in own_army.values() if n > 0)
        total_enemy = sum(n for n in enemy_army.values() if n > 0)
        if not total_own or not total_enemy:
            return 0.0
        result = 0.0
        for own_type, own_count in own_army.items():
            if own_count <= 0:
                continue
            countered = sum(
                count for enemy_type, count in enemy_army.items()
                if count > 0 and own_units_counter_enemy(own_type, enemy_type)
            )
            if countered == 0:
                continue
            spec = UNIT_SPECS.get(own_type)
            power = float(spec.power) if spec else 1.0
            weight = 0.75 + 0.25 * min(4.0, power / 20.0)
            result += own_count * (countered / total_enemy) * weight
        return float(result / total_own)


class V4ReplayBridge(ReplayBridge):
    """Fix tracker duplicate births and morphs; preserve other bridge checks."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parser = V4SnapshotParser()

    def _inventories_until(self, events: Iterable[object],
                           p1_pid: int, p2_pid: int, second: int):
        inventories: dict[int, dict[str, int]] = {p1_pid: {}, p2_pid: {}}
        active: dict[object, tuple[int, str]] = {}
        recent_losses = {p1_pid: 0, p2_pid: 0}
        window_start = second - self.snapshot_step_sec
        aliased = 0
        unknown = 0

        def adjust(pid: int, name: str, delta: int):
            new_count = inventories[pid].get(name, 0) + delta
            if new_count < 0:
                raise ValueError("Tracker lifecycle would create negative inventory")
            if new_count == 0:
                inventories[pid].pop(name, None)
            else:
                inventories[pid][name] = new_count

        # Preserve event stream order; sc2reader tracker_events is chronological.
        for event in events:
            if event.second > second:
                break
            kind = event.name
            unit_id = getattr(event, "unit_id", None)
            if kind in ("UnitBornEvent", "UnitInitEvent"):
                owner = getattr(event, "control_pid", None)
                if owner not in inventories:
                    continue
                raw_name = getattr(event, "unit_type_name", None)
                if not raw_name:
                    continue
                unit_type = UNIT_ALIASES.get(raw_name, raw_name)
                if unit_type != raw_name:
                    aliased += 1
                if unit_type not in UNIT_SPECS:
                    unknown += 1
                # An absent unit_id cannot be de-duplicated reliably;
                # we omit it rather than manufacture an extra live unit.
                if unit_id is None:
                    continue
                previous = active.get(unit_id)
                if previous == (owner, unit_type):
                    continue  # same unit has UnitInit + UnitBorn
                if previous is not None:
                    adjust(previous[0], previous[1], -1)
                active[unit_id] = (owner, unit_type)
                adjust(owner, unit_type, 1)
            elif kind in ("UnitTypeChangeEvent", "UnitOwnerChangeEvent"):
                if unit_id is None or unit_id not in active:
                    continue
                old_owner, old_type = active[unit_id]
                new_owner = getattr(event, "control_pid", old_owner)
                if new_owner not in inventories:
                    new_owner = old_owner
                raw_type = getattr(event, "unit_type_name", None)
                new_type = UNIT_ALIASES.get(raw_type, raw_type) if raw_type else old_type
                if new_type != old_type or new_owner != old_owner:
                    adjust(old_owner, old_type, -1)
                    adjust(new_owner, new_type, 1)
                    active[unit_id] = (new_owner, new_type)
            elif kind == "UnitDiedEvent":
                previous = active.pop(unit_id, None)
                if previous is not None:
                    owner, unit_type = previous
                    adjust(owner, unit_type, -1)
                    if window_start < event.second <= second:
                        recent_losses[owner] += 1

        return inventories, {
            "alias_unit_count": aliased,
            "unknown_unit_count": unknown,
            "tracked_unique_unit_ids": len(active),
        }, recent_losses
