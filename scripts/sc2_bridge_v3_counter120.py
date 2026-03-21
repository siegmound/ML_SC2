import os
import pandas as pd
import sc2reader

from sc2_parser_v3_counter120 import SC2ReplayParser
from sc2_constants import UNIT_MAPPING


SECONDS_STEP = 15
LOOPS_PER_SEC = 22.4


class SC2Bridge:
    def __init__(self):
        self.parser = SC2ReplayParser()

    def process_single_replay(self, file_path):
        try:
            replay_id = os.path.basename(file_path)
            self.parser.reset_state()

            replay = sc2reader.load_replay(file_path, load_level=4)
            replay.load_game_events()
        except Exception:
            return None

        if len(replay.players) < 2:
            return None

        p1, p2 = replay.players[0], replay.players[1]

        # Tracker/game events usati dal parser, sempre troncati al tempo corrente.
        unit_events = [e for e in replay.tracker_events if 'Unit' in e.name]
        stats_events = [e for e in replay.tracker_events if e.name == 'PlayerStatsEvent']
        upgrade_events = [e for e in replay.tracker_events if 'Upgrade' in e.name]
        cam_events = [e for e in replay.game_events if 'Camera' in e.name]
        cmd_events = [e for e in replay.game_events if 'Command' in e.name]

        # Mapping unit tag -> owner, necessario per attribuire correttamente le perdite.
        tag_to_owner = {}
        for e in unit_events:
            if 'UnitBorn' in e.name or 'UnitInit' in e.name:
                tag_to_owner[e.unit_id] = e.control_pid

        # Start locations ricavate dagli worker iniziali.
        base_locs = {}
        for e in unit_events:
            if 'UnitBorn' in e.name and e.unit_type_name in ['SCV', 'Probe', 'Drone']:
                if e.control_pid not in base_locs:
                    base_locs[e.control_pid] = e.location
            if len(base_locs) == 2:
                break

        if len(base_locs) < 2:
            return None

        snapshots = []

        for second in range(SECONDS_STEP, replay.length.seconds, SECONDS_STEP):
            loop = int(second * LOOPS_PER_SEC)

            inv = {
                p1.pid: self._inventory_until(unit_events, p1.pid, second),
                p2.pid: self._inventory_until(unit_events, p2.pid, second),
            }

            losses = {
                p1.pid: sum(
                    1
                    for e in unit_events
                    if e.name == 'UnitDiedEvent'
                    and tag_to_owner.get(e.unit_id) == p1.pid
                    and (second - SECONDS_STEP) < e.second <= second
                ),
                p2.pid: sum(
                    1
                    for e in unit_events
                    if e.name == 'UnitDiedEvent'
                    and tag_to_owner.get(e.unit_id) == p2.pid
                    and (second - SECONDS_STEP) < e.second <= second
                ),
            }

            res_stats = {
                p1.pid: self._resources_at(stats_events, p1.pid, second),
                p2.pid: self._resources_at(stats_events, p2.pid, second),
            }

            epms = {
                p1.pid: self._epm_window(cmd_events, p1.pid, second, SECONDS_STEP),
                p2.pid: self._epm_window(cmd_events, p2.pid, second, SECONDS_STEP),
            }

            upgrades = {
                p1.pid: {
                    e.upgrade_type_name
                    for e in upgrade_events
                    if e.pid == p1.pid and e.second <= second
                },
                p2.pid: {
                    e.upgrade_type_name
                    for e in upgrade_events
                    if e.pid == p2.pid and e.second <= second
                },
            }

            def get_cam_tuple(event):
                loc = getattr(event, 'location', (0, 0))
                if hasattr(loc, 'x'):
                    return (getattr(loc, 'x', 0), getattr(loc, 'y', 0))
                return loc

            cameras = {
                p1.pid: [
                    get_cam_tuple(e)
                    for e in cam_events
                    if hasattr(e, 'player') and e.player.pid == p1.pid and e.second <= second
                ],
                p2.pid: [
                    get_cam_tuple(e)
                    for e in cam_events
                    if hasattr(e, 'player') and e.player.pid == p2.pid and e.second <= second
                ],
            }

            winner_pid = replay.winner.players[0].pid if replay.winner else None
            metadata = {
                'winner': winner_pid,
                'start_locs': base_locs,
            }

            snap = self.parser._create_snapshot(
                loop=loop,
                p1=p1.pid,
                p2=p2.pid,
                inventories=inv,
                camera_history=cameras,
                upgrades_completed=upgrades,
                metadata=metadata,
                resource_stats=res_stats,
                epms=epms,
                recent_losses=losses,
                replay_id=replay_id,
            )
            snapshots.append(snap)

        return pd.DataFrame(snapshots) if snapshots else None

    # ---------------- UTILS ----------------

    def _inventory_until(self, events, pid, second):
        alive = {}
        tags = {}

        for e in events:
            if e.second > second:
                break

            if 'UnitBorn' in e.name and e.control_pid == pid:
                clean_name = UNIT_MAPPING.get(e.unit_type_name, e.unit_type_name)
                alive[clean_name] = alive.get(clean_name, 0) + 1
                tags[e.unit_id] = clean_name
            elif 'UnitDied' in e.name and e.unit_id in tags:
                name = tags[e.unit_id]
                alive[name] = max(0, alive[name] - 1)
                del tags[e.unit_id]

        return alive

    def _resources_at(self, events, pid, second):
        last = None
        for e in events:
            if e.second > second:
                break
            if e.pid == pid:
                last = e

        if not last:
            return {'m': 0, 'v': 0, 'mi': 0, 'vi': 0}

        return {
            'm': last.minerals_current,
            'v': last.vespene_current,
            'mi': last.minerals_collection_rate,
            'vi': last.vespene_collection_rate,
        }

    def _epm_window(self, events, pid, second, step):
        count = sum(
            1
            for e in events
            if hasattr(e, 'player')
            and e.player.pid == pid
            and (second - step) < e.second <= second
        )
        return (count / step) * 60


if __name__ == '__main__':
    bridge = SC2Bridge()
    # Esempio rapido locale: sostituisci con un replay esistente.
    test_path = 'replays/example.SC2Replay'
    if os.path.exists(test_path):
        df = bridge.process_single_replay(test_path)
        if df is not None:
            print('Dataset di test generato correttamente.')
            cols = [c for c in [
                'time_sec',
                'diff_workers',
                'diff_workers_60s',
                'diff_combat_score',
                'diff_combat_60s',
                'counter_advantage_diff',
                'loss_trade_ratio',
                'p1_wins',
            ] if c in df.columns]
            print(df[cols].tail())
