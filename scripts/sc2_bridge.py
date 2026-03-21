import sc2reader
import pandas as pd
import numpy as np
import os

from sc2_parser import SC2ReplayParser
from sc2_constants import UNIT_MAPPING


SECONDS_STEP = 15
LOOPS_PER_SEC = 22.4
SCOUT_RADIUS = 15   # distanza dalla base nemica


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

        # --- FILTRAGGIO EVENTI (CORRETTO) ---
        unit_events = [e for e in replay.tracker_events if 'Unit' in e.name]
        # Fondamentale: solo PlayerStatsEvent contengono i dati economici
        stats_events = [e for e in replay.tracker_events if e.name == 'PlayerStatsEvent']
        upgrade_events = [e for e in replay.tracker_events if 'Upgrade' in e.name]
        cmd_events = [e for e in replay.game_events if 'Command' in e.name]
        cam_events = [e for e in replay.game_events if 'Camera' in e.name]
        
        
        # Tracciamento tag -> proprietario (per risolvere l'AttributeError delle perdite)
        tag_to_owner = {}
        for e in unit_events:
            if 'UnitBorn' in e.name or 'UnitInit' in e.name:
                tag_to_owner[e.unit_id] = e.control_pid


        # --- BASE LOCATIONS ---
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

            # 1. Inventari
            inv = {
                p1.pid: self._inventory_until(unit_events, p1.pid, second),
                p2.pid: self._inventory_until(unit_events, p2.pid, second)
            }
            
            # Calcolo perdite corretto usando tag_to_owner
            losses = {
                p1.pid: sum(1 for e in unit_events if e.name == 'UnitDiedEvent' 
                            and tag_to_owner.get(e.unit_id) == p1.pid 
                            and (second-SECONDS_STEP) < e.second <= second),
                p2.pid: sum(1 for e in unit_events if e.name == 'UnitDiedEvent' 
                            and tag_to_owner.get(e.unit_id) == p2.pid 
                            and (second-SECONDS_STEP) < e.second <= second)
            }


            # 2. Risorse (Sincronizzato con Parser: m, v, mi, vi)
            res_stats = {
                p1.pid: self._resources_at(stats_events, p1.pid, second),
                p2.pid: self._resources_at(stats_events, p2.pid, second)
            }

            # 3. EPM
            epms = {
                p1.pid: self._epm_window(cmd_events, p1.pid, second),
                p2.pid: self._epm_window(cmd_events, p2.pid, second)
            }

            # 4. Upgrades e Camera
            upgrades = {
                p1.pid: set(e.upgrade_type_name for e in upgrade_events if e.pid == p1.pid and e.second <= second),
                p2.pid: set(e.upgrade_type_name for e in upgrade_events if e.pid == p2.pid and e.second <= second)
            }
            
            def get_cam_tuple(e):
                loc = getattr(e, 'location', (0,0))
                return (getattr(loc, 'x', 0), getattr(loc, 'y', 0)) if hasattr(loc, 'x') else loc

            cameras = {
                p1.pid: [get_cam_tuple(e) for e in cam_events if e.player.pid == p1.pid and e.second <= second],
                p2.pid: [get_cam_tuple(e) for e in cam_events if e.player.pid == p2.pid and e.second <= second]
            }

            # Assicurati che recent_losses sia passato correttamente
            snap = self.parser._create_snapshot(
                loop, p1.pid, p2.pid, inv, cameras, upgrades, 
                {'winner': replay.winner.players[0].pid if replay.winner else None, 'start_locs': base_locs}, 
                res_stats, epms, losses, replay_id
            )

            snapshots.append(snap)

        return pd.DataFrame(snapshots)

    # ---------------- UTILS ----------------

    def _inventory_until(self, events, pid, second):
        alive = {}
        tags = {}

        for e in events:
            if e.second > second:
                break

            if 'UnitBorn' in e.name and e.control_pid == pid:
                name = UNIT_MAPPING.get(e.unit_type_name, e.unit_type_name)
                alive[name] = alive.get(name, 0) + 1
                tags[e.unit_id] = name

            elif 'UnitDied' in e.name and e.unit_id in tags:
                name = tags[e.unit_id]
                alive[name] = max(0, alive[name] - 1)
                del tags[e.unit_id]

        return alive

    def _resources_at(self, events, pid, second):
        """Recupera dati economici. Ritorna m, v, mi, vi."""
        last = None
        for e in events:
            if e.second > second:
                break
            if e.pid == pid:
                last = e

        if not last:
            return {"m": 0, "v": 0, "mi": 0, "vi": 0}

        return {
            "m": last.minerals_current,
            "v": last.vespene_current,
            "mi": last.minerals_collection_rate,
            "vi": last.vespene_collection_rate,
        }

    def _epm_window(self, events, pid, second, window=60):
        start = max(0, second - window)
        count = sum(
            1 for e in events
            if hasattr(e, "player")
            and e.player.pid == pid
            and start <= e.second <= second
        )
        return count

    def _scout_ratio(self, events, pid, second, enemy_base):
        cams = [
            e for e in events
            if 'Camera' in e.name
            and e.player.pid == pid
            and e.second <= second
        ]

        if not cams:
            return 0.0

        def dist(a, b):
            return np.linalg.norm(np.array(a) - np.array(b))

        scout_frames = sum(
            1 for e in cams
            if hasattr(e, "location")
            and dist(e.location, enemy_base) <= SCOUT_RADIUS
        )

        return scout_frames / len(cams)

if __name__ == "__main__":
    bridge = SC2Bridge()
    # Test su un file locale (cambia il nome con uno dei tuoi)
    df = bridge.process_single_replay('replays/45add59eb97b5ac01eac21c3749ce037.SC2Replay')
    if df is not None:
        print("Dataset di test generato correttamente.")
        print(df[['time_sec', 'diff_combat_score', 'diff_sq', 'p1_wins']].tail())