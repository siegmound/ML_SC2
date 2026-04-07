from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import sc2reader

from .constants import DEFAULT_MAX_MATCH_SEC, DEFAULT_MIN_MATCH_SEC, DEFAULT_SNAPSHOT_STEP_SEC, DEFAULT_EARLY_CUT_SEC, LOOPS_PER_SECOND, UNIT_ALIASES
from .parser import SnapshotParser


@dataclass
class ParseResult:
    replay_id: str
    status: str
    reason: str
    dataframe: pd.DataFrame | None
    metadata: dict


class ReplayBridge:
    def __init__(self, snapshot_step_sec: int = DEFAULT_SNAPSHOT_STEP_SEC, early_cut_sec: int = DEFAULT_EARLY_CUT_SEC, min_match_sec: int = DEFAULT_MIN_MATCH_SEC, max_match_sec: int = DEFAULT_MAX_MATCH_SEC) -> None:
        self.snapshot_step_sec = snapshot_step_sec
        self.early_cut_sec = early_cut_sec
        self.min_match_sec = min_match_sec
        self.max_match_sec = max_match_sec
        self.parser = SnapshotParser(loops_per_second=LOOPS_PER_SECOND)

    def process_replay(self, replay_path: str | Path) -> ParseResult:
        replay_path = Path(replay_path)
        replay_id = replay_path.name
        self.parser.reset()
        metadata = {
            "replay_id": replay_id,
            "parse_status": "failed",
            "failure_reason": "unknown_error",
            "match_duration_sec": None,
            "num_snapshots": 0,
            "winner_known": False,
            "player_count": 0,
            "unknown_unit_count": 0,
            "alias_unit_count": 0,
            "min_time_sec": None,
            "max_time_sec": None,
        }
        try:
            replay = sc2reader.load_replay(str(replay_path), load_level=4)
            replay.load_game_events()
        except Exception as exc:
            metadata["failure_reason"] = f"load_error:{type(exc).__name__}"
            return ParseResult(replay_id, "failed", metadata["failure_reason"], None, metadata)

        duration_sec = int(getattr(replay.length, "seconds", 0))
        metadata["match_duration_sec"] = duration_sec
        metadata["player_count"] = len(getattr(replay, "players", []))
        if duration_sec < self.min_match_sec:
            metadata["failure_reason"] = "duration_too_short"
            return ParseResult(replay_id, "filtered", metadata["failure_reason"], None, metadata)
        if duration_sec > self.max_match_sec:
            metadata["failure_reason"] = "duration_too_long"
            return ParseResult(replay_id, "filtered", metadata["failure_reason"], None, metadata)
        if metadata["player_count"] < 2:
            metadata["failure_reason"] = "not_enough_players"
            return ParseResult(replay_id, "failed", metadata["failure_reason"], None, metadata)

        p1, p2 = replay.players[0], replay.players[1]
        winner_pid = replay.winner.players[0].pid if getattr(replay, "winner", None) else None
        metadata["winner_known"] = winner_pid is not None

        unit_events = [e for e in replay.tracker_events if "Unit" in e.name]
        stats_events = [e for e in replay.tracker_events if e.name == "PlayerStatsEvent"]
        upgrade_events = [e for e in replay.tracker_events if "Upgrade" in e.name]
        command_events = [e for e in replay.game_events if "Command" in e.name]
        camera_events = [e for e in replay.game_events if "Camera" in e.name]

        base_locations: Dict[int, Tuple[float, float]] = {}
        tag_to_owner: Dict[int, int] = {}
        for event in unit_events:
            if event.name in {"UnitBornEvent", "UnitInitEvent"}:
                if hasattr(event, "unit_id") and hasattr(event, "control_pid"):
                    tag_to_owner[event.unit_id] = event.control_pid
                if getattr(event, "unit_type_name", None) in {"SCV", "Probe", "Drone"} and hasattr(event, "location"):
                    if event.control_pid not in base_locations:
                        loc = event.location
                        if hasattr(loc, "x"):
                            base_locations[event.control_pid] = (float(loc.x), float(loc.y))
                        else:
                            base_locations[event.control_pid] = (float(loc[0]), float(loc[1]))
                if len(base_locations) == 2:
                    break
        if len(base_locations) < 2:
            metadata["failure_reason"] = "missing_start_locations"
            return ParseResult(replay_id, "failed", metadata["failure_reason"], None, metadata)

        snapshots: List[dict] = []
        alias_unit_count = 0
        unknown_unit_count = 0
        for second in range(self.snapshot_step_sec, duration_sec, self.snapshot_step_sec):
            loop = int(second * LOOPS_PER_SECOND)
            inventories, unit_stats = self._inventories_until(unit_events, p1.pid, p2.pid, second)
            alias_unit_count += unit_stats["alias_unit_count"]
            unknown_unit_count += unit_stats["unknown_unit_count"]
            recent_losses = {
                p1.pid: sum(1 for e in unit_events if e.name == "UnitDiedEvent" and tag_to_owner.get(getattr(e, "unit_id", -1)) == p1.pid and (second - self.snapshot_step_sec) < e.second <= second),
                p2.pid: sum(1 for e in unit_events if e.name == "UnitDiedEvent" and tag_to_owner.get(getattr(e, "unit_id", -1)) == p2.pid and (second - self.snapshot_step_sec) < e.second <= second),
            }
            resource_stats = {
                p1.pid: self._resources_at(stats_events, p1.pid, second),
                p2.pid: self._resources_at(stats_events, p2.pid, second),
            }
            epms = {
                p1.pid: self._epm_window(command_events, p1.pid, second),
                p2.pid: self._epm_window(command_events, p2.pid, second),
            }
            upgrades_completed = {
                p1.pid: {e.upgrade_type_name for e in upgrade_events if getattr(e, "pid", None) == p1.pid and e.second <= second},
                p2.pid: {e.upgrade_type_name for e in upgrade_events if getattr(e, "pid", None) == p2.pid and e.second <= second},
            }
            camera_history = {
                p1.pid: self._camera_positions(camera_events, p1.pid, second),
                p2.pid: self._camera_positions(camera_events, p2.pid, second),
            }
            snapshot = self.parser.build_snapshot(
                replay_id=replay_id,
                loop=loop,
                p1_pid=p1.pid,
                p2_pid=p2.pid,
                inventories=inventories,
                camera_history=camera_history,
                upgrades_completed=upgrades_completed,
                winner_pid=winner_pid,
                base_locations=base_locations,
                resource_stats=resource_stats,
                epms=epms,
                recent_losses=recent_losses,
            )
            snapshots.append(snapshot)

        if not snapshots:
            metadata["failure_reason"] = "no_snapshots_generated"
            return ParseResult(replay_id, "failed", metadata["failure_reason"], None, metadata)

        df = pd.DataFrame(snapshots)
        df = df[df["time_sec"] > self.early_cut_sec].copy()
        if df.empty:
            metadata["failure_reason"] = "empty_after_early_cut"
            return ParseResult(replay_id, "filtered", metadata["failure_reason"], None, metadata)

        metadata.update({
            "parse_status": "ok",
            "failure_reason": "",
            "num_snapshots": int(len(df)),
            "unknown_unit_count": int(unknown_unit_count),
            "alias_unit_count": int(alias_unit_count),
            "min_time_sec": float(df["time_sec"].min()),
            "max_time_sec": float(df["time_sec"].max()),
        })
        return ParseResult(replay_id, "ok", "", df, metadata)

    def _inventories_until(self, events, p1_pid: int, p2_pid: int, second: int):
        inventories = {p1_pid: {}, p2_pid: {}}
        live_tags: Dict[int, tuple[int, str]] = {}
        alias_unit_count = 0
        unknown_unit_count = 0
        for event in events:
            if event.second > second:
                break
            if event.name in {"UnitBornEvent", "UnitInitEvent"}:
                owner = getattr(event, "control_pid", None)
                if owner not in inventories:
                    continue
                raw_name = getattr(event, "unit_type_name", "UnknownUnit")
                canonical_name = UNIT_ALIASES.get(raw_name, raw_name)
                if canonical_name != raw_name:
                    alias_unit_count += 1
                from .constants import UNIT_SPECS
                if canonical_name not in UNIT_SPECS:
                    unknown_unit_count += 1
                inventories[owner][canonical_name] = inventories[owner].get(canonical_name, 0) + 1
                live_tags[getattr(event, "unit_id", -1)] = (owner, canonical_name)
            elif event.name == "UnitDiedEvent":
                info = live_tags.pop(getattr(event, "unit_id", -1), None)
                if info is not None:
                    owner, canonical_name = info
                    inventories[owner][canonical_name] = max(0, inventories[owner].get(canonical_name, 0) - 1)
        return inventories, {"alias_unit_count": alias_unit_count, "unknown_unit_count": unknown_unit_count}

    def _resources_at(self, events, pid: int, second: int):
        last = None
        for event in events:
            if event.second > second:
                break
            if getattr(event, "pid", None) == pid:
                last = event
        if last is None:
            return {"m": 0.0, "v": 0.0, "mi": 0.0, "vi": 0.0}
        return {
            "m": float(getattr(last, "minerals_current", 0.0)),
            "v": float(getattr(last, "vespene_current", 0.0)),
            "mi": float(getattr(last, "minerals_collection_rate", 0.0)),
            "vi": float(getattr(last, "vespene_collection_rate", 0.0)),
        }

    def _epm_window(self, events, pid: int, second: int, window: int = 60) -> int:
        start = max(0, second - window)
        return sum(1 for event in events if hasattr(event, "player") and event.player.pid == pid and start <= event.second <= second)

    def _camera_positions(self, events, pid: int, second: int):
        positions = []
        for event in events:
            if event.second > second:
                break
            if getattr(getattr(event, "player", None), "pid", None) != pid:
                continue
            loc = getattr(event, "location", None)
            if loc is None:
                continue
            if hasattr(loc, "x"):
                positions.append((float(loc.x), float(loc.y)))
            else:
                positions.append((float(loc[0]), float(loc[1])))
        return positions
