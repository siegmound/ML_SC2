import math
import numpy as np
from sc2_constants import *
from sc2_counters import *

class SC2ReplayParser:
    def __init__(self):
        self.LOOPS_PER_SEC = 22.4
        self.UNIT_STATS = UNIT_STATS
        self.UNIT_COUNTERS = UNIT_COUNTERS
        self.EARLY_GAME_UNITS = EARLY_GAME_UNITS
        self.MID_GAME_UNITS = MID_GAME_UNITS
        self.LATE_GAME_UNITS = LATE_GAME_UNITS
        self.UNIT_TECH_MULTIPLIERS = UNIT_TECH_MULTIPLIERS
        self.prev_snapshot = None

    # ============================================================
    # RESET STATO TRA I REPLAY
    # ============================================================
    def reset_state(self):
        self.prev_snapshot = None

    # ============================================================
    # COMBAT SCORE
    # ============================================================
    def _get_upgrade_multiplier(self, pid, unit, upgrades_completed):
        bonus = 0.0
        player_upgrades = upgrades_completed.get(pid, set())
        u_cat = self.UNIT_STATS.get(unit, {}).get('upg_cat', 'None')
        for upg in player_upgrades:
            if (u_cat != 'None' and u_cat in upg) or ("Shields" in upg):
                if any(x in upg for x in ["Level1", "Level2", "Level3"]):
                    bonus += 0.10
        if unit in self.UNIT_TECH_MULTIPLIERS:
            for tech in self.UNIT_TECH_MULTIPLIERS[unit]:
                if any(tech in u for u in player_upgrades):
                    bonus += 0.15
        return 1.0 + bonus

    def _calc_combat_score_pair(self, inventories, upgrades_completed, p1_pid, p2_pid):
        scores = {p1_pid: 0.0, p2_pid: 0.0}
        for curr, opp in [(p1_pid, p2_pid), (p2_pid, p1_pid)]:
            total_p = 0.0
            for unit, count in inventories[curr].items():
                stats = self.UNIT_STATS.get(unit)
                if not stats or stats['type'] != 'army' or count <= 0: continue
                
                # Torniamo alla logica LINEARE (più stabile per XGBoost)
                base_power = stats['power'] * count
                
                penalty = 0
                for opp_u, opp_c in inventories[opp].items():
                    if opp_c > 0 and opp_u in self.UNIT_COUNTERS.get(unit, []):
                        penalty += 0.15
                
                m_counter = max(0.4, 1.0 - penalty)
                m_upgrade = self._get_upgrade_multiplier(curr, unit, upgrades_completed)
                total_p += (base_power * m_counter * m_upgrade)
            scores[curr] = total_p / 100.0
        return scores[p1_pid], scores[p2_pid]

    # ============================================================
    # ECONOMIA
    # ============================================================
    def _calc_spending_quotient(self, unspent_m, unspent_v, inc_m, inc_v):
        unspent = max(10, unspent_m + unspent_v)
        income = max(10, inc_m + inc_v)
        return 35 * (0.0013 * income - math.log(unspent)) + 41.3

    # ============================================================
    # SCOUTING
    # ============================================================
    def _calc_scouting_metrics(self, camera_data, enemy_base):
        # Protezione: camera_data deve contenere tuple (x, y)
        if len(camera_data) < 5: 
            return {'score': 0.0}

        # Trasformiamo in tuple se non lo sono già (doppia sicurezza)
        try:
            cells = set((int(pos[0] / 10), int(pos[1] / 10)) for pos in camera_data)
            coverage = len(cells)

            ex, ey = enemy_base
            # Calcolo focus sulla base nemica
            rival_looks = 0
            for pos in camera_data:
                dist = math.sqrt((pos[0] - ex)**2 + (pos[1] - ey)**2)
                if dist < 30:
                    rival_looks += 1
            
            rival_focus = (rival_looks / len(camera_data)) * 100

            # Calcolo salti camera
            jumps = 0
            for i in range(1, len(camera_data)):
                d = math.sqrt((camera_data[i][0] - camera_data[i-1][0])**2 + 
                              (camera_data[i][1] - camera_data[i-1][1])**2)
                if d > 40:
                    jumps += 1
            jump_score = (jumps / len(camera_data)) * 100

            return {'score': (coverage * 0.3) + (rival_focus * 0.5) + (jump_score * 0.2)}
        except Exception as e:
            return {'score': 0.0}
    # ============================================================
    # ENTROPY CALCULATION
    # ============================================================
    
    def _calc_army_entropy(self, army_inventory):
        """Calcola l'entropia di Shannon della composizione dell'esercito."""
        total_units = sum(army_inventory.values())
        if total_units <= 1: return 0.0
        
        entropy = 0.0
        for count in army_inventory.values():
            if count > 0:
                p = count / total_units
                entropy -= p * math.log2(p)
        return entropy

    def _get_max_tech(self, inventory):
        """Ritorna il livello tecnologico massimo attuale."""
        levels = [self.UNIT_STATS[u]['tech_level'] for u in inventory if u in self.UNIT_STATS and inventory[u] > 0]
        return max(levels) if levels else 0
    
    
    

    # ============================================================
    # SNAPSHOT
    # ============================================================
    def _create_snapshot(
        self, loop, p1, p2, inventories, camera_history,
        upgrades_completed, metadata, resource_stats,
        epms,  recent_losses,replay_id
    ):
        time_sec = round(loop / self.LOOPS_PER_SEC, 1)
        c_p1, c_p2 = self._calc_combat_score_pair(inventories, upgrades_completed, p1, p2)
        
        # Helper per categorie
        # Helper per categorie con protezione contro unità sconosciute (es. BeaconArmy)
        def get_data(pid):
            # Filtriamo l'inventario: teniamo solo le unità presenti nel nostro UNIT_STATS
            known_inv = {u: count for u, count in inventories[pid].items() if u in self.UNIT_STATS}
            
            # Calcolo Army: solo unità di tipo 'army' e con conteggio > 0
            army = {u: count for u, count in known_inv.items() 
                    if self.UNIT_STATS[u].get('type') == 'army' and count > 0}
            
            # Calcolo Workers
            workers = sum(count for u, count in known_inv.items() 
                          if self.UNIT_STATS[u].get('type') == 'worker')
            
            # Calcolo Supply (usando solo unità conosciute)
            supply_army = sum(count * self.UNIT_STATS[u].get('supply', 0) for u, count in army.items())
            supply_total = supply_army + workers
            
            # Calcolo Tech Level (evita il KeyError cercando solo in known_inv)
            tech_levels = [self.UNIT_STATS[u].get('tech_level', 0) for u in known_inv if known_inv[u] > 0]
            tech = max(tech_levels + [0])
            
            return army, workers, supply_army, supply_total, tech

        a1, w1, sa1, st1, t1 = get_data(p1)
        a2, w2, sa2, st2, t2 = get_data(p2)

        sq1 = self._calc_spending_quotient(
            resource_stats[p1]['m'], resource_stats[p1]['v'],
            resource_stats[p1]['mi'], resource_stats[p1]['vi']
        )
        sq2 = self._calc_spending_quotient(
            resource_stats[p2]['m'], resource_stats[p2]['v'],
            resource_stats[p2]['mi'], resource_stats[p2]['vi']
        )

        scout1 = self._calc_scouting_metrics(
            camera_history[p1], metadata['start_locs'].get(p2, (0, 0))
        )
        scout2 = self._calc_scouting_metrics(
            camera_history[p2], metadata['start_locs'].get(p1, (0, 0))
        )

        # ==========================
        # DELTA TEMPORALI
        # ==========================
        d_combat = d_workers = d_sq = d_workers_p2 = worker_growth_p1 = 0.0
        if self.prev_snapshot:
            d_combat = c_p1 - self.prev_snapshot['p1_combat']
            d_workers = w1 - self.prev_snapshot['p1_workers']
            d_sq = sq1 - self.prev_snapshot['p1_sq']
            d_workers_p2 = w2 - self.prev_snapshot['p2_workers']
            worker_growth_p1 = (w1 - self.prev_snapshot['p1_workers']) / max(1, self.prev_snapshot['p1_workers'])

        # ==========================
        # FASE DI GIOCO
        # ==========================
        phase = 0 if time_sec < 300 else 1 if time_sec < 900 else 2

        snapshot = {
            
            'replay_id': replay_id,
            'time_sec': time_sec,
            #'phase': phase,
            
            # Differenziali
            
            'diff_combat_score': c_p1 - c_p2,
            'diff_workers': w1 - w2, 
            'diff_sq': sq1 - sq2, 
            'diff_epm': epms[p1] - epms[p2], 
            #'tech_diff': t1 - t2,
            
            # Player 1 Raw & New
            
            'p1_combat': c_p1, 
            'p1_workers': w1, 
            'p1_sq': sq1, 
            'p1_epm': epms[p1], 
            'p1_scout': scout1['score'],
            'p1_army_entropy': self._calc_army_entropy(a1),
            'p1_num_unit_types': len(a1),
            #'p1_tech_level': t1,
            'p1_army_supply_ratio': sa1 / max(1, st1),
            'p1_recent_losses': recent_losses[p1],
            'p1_worker_growth_ratio': worker_growth_p1,
            
            # Player 2 Raw & New
            
            'p2_combat': c_p2,
            'p2_workers': w2, 
            'p2_sq': sq2, 
            'p2_epm': epms[p2], 
            'p2_scout': scout2['score'],
            'p2_army_entropy': self._calc_army_entropy(a2),
            'p2_num_unit_types': len(a2),
            'p2_army_entropy': self._calc_army_entropy(a2), 
            #'p2_tech_level': t2, 
            'p2_army_supply_ratio': sa2 / max(1, st2),
            'p2_recent_losses': recent_losses[p2], 
            'delta_workers_p2': d_workers_p2,
            
            # Trend
            
            'delta_combat_p1': d_combat, 
            'delta_workers_p1': d_workers, 
            'delta_sq_p1': d_sq,
            
            # Target
            
            'p1_wins': 1 if metadata['winner'] == p1 else 0
        }

        self.prev_snapshot = snapshot
        return snapshot
