from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, Tuple

import numpy as np

from .constants import UNIT_SPECS, UPGRADE_MULTIPLIERS
from .counters import UNIT_COUNTERS


@dataclass
class SnapshotState:
    time_sec: float
    p1_workers: float
    p2_workers: float
    p1_combat: float
    p2_combat: float
    p1_sq: float
    p2_sq: float
    p1_income: float
    p2_income: float


@dataclass
class SnapshotParser:
    loops_per_second: float = 22.4
    history: Deque[SnapshotState] = field(default_factory=lambda: deque(maxlen=16))

    def reset(self) -> None:
        self.history.clear()

    def _spending_quotient(self, minerals: float, vespene: float, minerals_income: float, vespene_income: float) -> float:
        unspent = max(10.0, minerals + vespene)
        income = max(10.0, minerals_income + vespene_income)
        return 35.0 * (0.0013 * income - math.log(unspent)) + 41.3

    def _upgrade_multiplier(self, unit_name: str, upgrades_completed: Iterable[str]) -> float:
        spec = UNIT_SPECS.get(unit_name)
        if spec is None:
            return 1.0
        bonus = 0.0
        for upgrade in upgrades_completed:
            if spec.upgrade_category != "None" and spec.upgrade_category in upgrade and any(level in upgrade for level in ("Level1", "Level2", "Level3")):
                bonus += 0.10
        for tech in UPGRADE_MULTIPLIERS.get(unit_name, ()):  # specific techs
            if any(tech in upgrade for upgrade in upgrades_completed):
                bonus += 0.15
        return 1.0 + bonus

    def _army_breakdown(self, inventory: Dict[str, int]) -> Tuple[Dict[str, int], int, float, float, int]:
        known = {u: c for u, c in inventory.items() if u in UNIT_SPECS and c > 0}
        army = {u: c for u, c in known.items() if UNIT_SPECS[u].role == "army"}
        workers = sum(c for u, c in known.items() if UNIT_SPECS[u].role == "worker")
        army_supply = sum(c * UNIT_SPECS[u].supply for u, c in army.items())
        total_supply = army_supply + workers
        tech_level = max([UNIT_SPECS[u].tech_level for u in known] + [0])
        return army, workers, float(army_supply), float(total_supply), tech_level

    def _combat_score(self, own_inventory: Dict[str, int], opp_inventory: Dict[str, int], upgrades_completed: Iterable[str]) -> float:
        total = 0.0
        for unit_name, count in own_inventory.items():
            spec = UNIT_SPECS.get(unit_name)
            if spec is None or spec.role != "army" or count <= 0:
                continue
            base_power = spec.power * count
            penalty = 0.0
            for opp_unit, opp_count in opp_inventory.items():
                if opp_count > 0 and opp_unit in UNIT_COUNTERS.get(unit_name, []):
                    penalty += 0.15
            counter_multiplier = max(0.4, 1.0 - penalty)
            total += base_power * counter_multiplier * self._upgrade_multiplier(unit_name, upgrades_completed)
        return total / 100.0

    def _counter_advantage(self, own_army: Dict[str, int], enemy_army: Dict[str, int]) -> float:
        own_total = sum(own_army.values())
        enemy_total = sum(enemy_army.values())
        if own_total <= 0 or enemy_total <= 0:
            return 0.0
        score = 0.0
        for unit_name, count in own_army.items():
            if count <= 0:
                continue
            countered = sum(enemy_army.get(enemy_name, 0) for enemy_name in UNIT_COUNTERS.get(unit_name, []))
            if countered <= 0:
                continue
            fraction = countered / enemy_total
            spec = UNIT_SPECS.get(unit_name)
            power = spec.power if spec else 1.0
            weight = 0.75 + 0.25 * min(4.0, power / 20.0)
            score += count * fraction * weight
        return score / own_total

    def _army_entropy(self, army: Dict[str, int]) -> float:
        total_units = sum(army.values())
        if total_units <= 1:
            return 0.0
        entropy = 0.0
        for count in army.values():
            if count > 0:
                p = count / total_units
                entropy -= p * math.log2(p)
        return entropy

    def _scouting_score(self, camera_positions: list[tuple[float, float]], enemy_base: tuple[float, float]) -> float:
        if len(camera_positions) < 5:
            return 0.0
        cells = {(int(x / 10), int(y / 10)) for x, y in camera_positions}
        coverage = len(cells)
        ex, ey = enemy_base
        rival_focus = sum(1 for x, y in camera_positions if math.hypot(x - ex, y - ey) < 30) / len(camera_positions) * 100.0
        jumps = 0
        for i in range(1, len(camera_positions)):
            x0, y0 = camera_positions[i - 1]
            x1, y1 = camera_positions[i]
            if math.hypot(x1 - x0, y1 - y0) > 40:
                jumps += 1
        jump_score = jumps / len(camera_positions) * 100.0
        return coverage * 0.3 + rival_focus * 0.5 + jump_score * 0.2

    def _history_ago(self, seconds_ago: float) -> SnapshotState | None:
        if not self.history:
            return None
        current = self.history[-1].time_sec
        target = current - seconds_ago
        for snapshot in reversed(self.history):
            if snapshot.time_sec <= target:
                return snapshot
        return None

    def _rolling_std(self, attr_name: str, window_sec: float) -> float:
        if len(self.history) < 2:
            return 0.0
        current = self.history[-1].time_sec
        values = [getattr(snapshot, attr_name) for snapshot in self.history if snapshot.time_sec >= current - window_sec]
        if len(values) < 2:
            return 0.0
        return float(np.std(values))

    def build_snapshot(
        self,
        *,
        replay_id: str,
        loop: int,
        p1_pid: int,
        p2_pid: int,
        inventories: Dict[int, Dict[str, int]],
        camera_history: Dict[int, list[tuple[float, float]]],
        upgrades_completed: Dict[int, set[str]],
        winner_pid: int | None,
        base_locations: Dict[int, tuple[float, float]],
        resource_stats: Dict[int, Dict[str, float]],
        epms: Dict[int, int],
        recent_losses: Dict[int, int],
    ) -> Dict[str, float | int | str]:
        time_sec = round(loop / self.loops_per_second, 1)
        p1_army, p1_workers, p1_army_supply, p1_total_supply, p1_tech = self._army_breakdown(inventories[p1_pid])
        p2_army, p2_workers, p2_army_supply, p2_total_supply, p2_tech = self._army_breakdown(inventories[p2_pid])

        p1_resources = resource_stats[p1_pid]
        p2_resources = resource_stats[p2_pid]
        p1_income = p1_resources["mi"] + p1_resources["vi"]
        p2_income = p2_resources["mi"] + p2_resources["vi"]
        p1_sq = self._spending_quotient(p1_resources["m"], p1_resources["v"], p1_resources["mi"], p1_resources["vi"])
        p2_sq = self._spending_quotient(p2_resources["m"], p2_resources["v"], p2_resources["mi"], p2_resources["vi"])

        p1_combat = self._combat_score(inventories[p1_pid], inventories[p2_pid], upgrades_completed[p1_pid])
        p2_combat = self._combat_score(inventories[p2_pid], inventories[p1_pid], upgrades_completed[p2_pid])
        p1_scout = self._scouting_score(camera_history[p1_pid], base_locations.get(p2_pid, (0.0, 0.0)))
        p2_scout = self._scouting_score(camera_history[p2_pid], base_locations.get(p1_pid, (0.0, 0.0)))
        counter_advantage_diff = self._counter_advantage(p1_army, p2_army) - self._counter_advantage(p2_army, p1_army)

        prev = self.history[-1] if self.history else None
        state_now = SnapshotState(time_sec, p1_workers, p2_workers, p1_combat, p2_combat, p1_sq, p2_sq, p1_income, p2_income)

        def delta_from(previous: SnapshotState | None, current: float, attr_name: str) -> float:
            return 0.0 if previous is None else float(current - getattr(previous, attr_name))

        snapshot = {
            "replay_id": replay_id,
            "time_sec": time_sec,
            "diff_workers": float(p1_workers - p2_workers),
            "diff_combat_score": float(p1_combat - p2_combat),
            "diff_sq": float(p1_sq - p2_sq),
            "diff_epm": float(epms[p1_pid] - epms[p2_pid]),
            "diff_income": float(p1_income - p2_income),
            "tech_diff": float(p1_tech - p2_tech),
            "counter_advantage_diff": float(counter_advantage_diff),
            "p1_combat": float(p1_combat),
            "p1_workers": float(p1_workers),
            "p1_sq": float(p1_sq),
            "p1_epm": float(epms[p1_pid]),
            "p1_income": float(p1_income),
            "p1_scout": float(p1_scout),
            "p1_army_entropy": float(self._army_entropy(p1_army)),
            "p1_num_unit_types": float(len(p1_army)),
            "p1_army_supply_ratio": float(p1_army_supply / max(1.0, p1_total_supply)),
            "p1_recent_losses": float(recent_losses[p1_pid]),
            "p2_combat": float(p2_combat),
            "p2_workers": float(p2_workers),
            "p2_sq": float(p2_sq),
            "p2_epm": float(epms[p2_pid]),
            "p2_income": float(p2_income),
            "p2_scout": float(p2_scout),
            "p2_army_entropy": float(self._army_entropy(p2_army)),
            "p2_num_unit_types": float(len(p2_army)),
            "p2_army_supply_ratio": float(p2_army_supply / max(1.0, p2_total_supply)),
            "p2_recent_losses": float(recent_losses[p2_pid]),
            "delta_combat_p1": delta_from(prev, p1_combat, "p1_combat"),
            "delta_workers_p1": delta_from(prev, p1_workers, "p1_workers"),
            "delta_sq_p1": delta_from(prev, p1_sq, "p1_sq"),
            "delta_income_p1": delta_from(prev, p1_income, "p1_income"),
            "delta_workers_p2": delta_from(prev, p2_workers, "p2_workers"),
            "trend60_workers_p1": delta_from(self._history_ago(60), p1_workers, "p1_workers"),
            "trend60_combat_p1": delta_from(self._history_ago(60), p1_combat, "p1_combat"),
            "trend120_workers_p1": delta_from(self._history_ago(120), p1_workers, "p1_workers"),
            "trend120_combat_p1": delta_from(self._history_ago(120), p1_combat, "p1_combat"),
            "rolling_std_combat_p1_120": self._rolling_std("p1_combat", 120),
            "rolling_std_workers_p1_120": self._rolling_std("p1_workers", 120),
            "p1_wins": int(winner_pid == p1_pid) if winner_pid is not None else 0,
        }

        self.history.append(state_now)
        return snapshot
