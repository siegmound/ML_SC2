import math
from collections import deque
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
        self.history = deque(maxlen=16)  # 15s step -> fino a 4 minuti di storico

    def reset_state(self):
        self.prev_snapshot = None
        self.history.clear()

    # ============================================================
    # COMBAT / UPGRADE
    # ============================================================
    def _get_upgrade_multiplier(self, pid, unit, upgrades_completed):
        bonus = 0.0
        player_upgrades = upgrades_completed.get(pid, set())
        stats = self.UNIT_STATS.get(unit, {})
        attack_cats = stats.get('attack_upg_cats', [])
        armor_cats = stats.get('armor_upg_cats', [])
        shield_upgrade = stats.get('shield_upgrade', False)
        legacy_cat = stats.get('upg_cat')

        for upg in player_upgrades:
            is_level_upgrade = any(x in upg for x in ['Level1', 'Level2', 'Level3'])
            if not is_level_upgrade:
                continue

            matched = False
            if legacy_cat and legacy_cat != 'None' and legacy_cat in upg:
                matched = True
            if any(cat in upg for cat in attack_cats):
                matched = True
            if any(cat in upg for cat in armor_cats):
                matched = True
            if shield_upgrade and 'Shields' in upg:
                matched = True

            if matched:
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
                if not stats or stats.get('type') != 'army' or count <= 0:
                    continue

                base_power = stats['power'] * count

                penalty = 0.0
                for opp_u, opp_c in inventories[opp].items():
                    if opp_c > 0 and opp_u in self.UNIT_COUNTERS.get(unit, []):
                        penalty += 0.15

                m_counter = max(0.4, 1.0 - penalty)
                m_upgrade = self._get_upgrade_multiplier(curr, unit, upgrades_completed)
                total_p += base_power * m_counter * m_upgrade

            scores[curr] = total_p / 100.0

        return scores[p1_pid], scores[p2_pid]

    # ============================================================
    # ECON / SCOUT / ARMY SHAPE
    # ============================================================
    def _calc_spending_quotient(self, unspent_m, unspent_v, inc_m, inc_v):
        unspent = max(10, unspent_m + unspent_v)
        income = max(10, inc_m + inc_v)
        return 35 * (0.0013 * income - math.log(unspent)) + 41.3

    def _calc_scouting_metrics(self, camera_data, enemy_base):
        if len(camera_data) < 5:
            return {'score': 0.0}

        try:
            cells = set((int(pos[0] / 10), int(pos[1] / 10)) for pos in camera_data)
            coverage = len(cells)

            ex, ey = enemy_base
            rival_looks = 0
            for pos in camera_data:
                dist = math.sqrt((pos[0] - ex) ** 2 + (pos[1] - ey) ** 2)
                if dist < 30:
                    rival_looks += 1
            rival_focus = (rival_looks / len(camera_data)) * 100

            jumps = 0
            for i in range(1, len(camera_data)):
                d = math.sqrt(
                    (camera_data[i][0] - camera_data[i - 1][0]) ** 2
                    + (camera_data[i][1] - camera_data[i - 1][1]) ** 2
                )
                if d > 40:
                    jumps += 1
            jump_score = (jumps / len(camera_data)) * 100

            return {'score': (coverage * 0.3) + (rival_focus * 0.5) + (jump_score * 0.2)}
        except Exception:
            return {'score': 0.0}

    def _calc_army_entropy(self, army_inventory):
        total_units = sum(army_inventory.values())
        if total_units <= 1:
            return 0.0

        entropy = 0.0
        for count in army_inventory.values():
            if count > 0:
                p = count / total_units
                entropy -= p * math.log2(p)
        return entropy

    def _get_known_player_data(self, inventories, pid):
        known_inv = {u: count for u, count in inventories[pid].items() if u in self.UNIT_STATS}

        army = {
            u: count
            for u, count in known_inv.items()
            if self.UNIT_STATS[u].get('type') == 'army' and count > 0
        }
        workers = sum(
            count for u, count in known_inv.items() if self.UNIT_STATS[u].get('type') == 'worker'
        )
        supply_army = sum(count * self.UNIT_STATS[u].get('supply', 0) for u, count in army.items())
        supply_total = supply_army + workers
        tech_levels = [self.UNIT_STATS[u].get('tech_level', 0) for u in known_inv if known_inv[u] > 0]
        tech = max(tech_levels + [0])

        return army, workers, supply_army, supply_total, tech

    # ============================================================
    # COUNTER ADVANTAGE
    # ============================================================
    def _calc_counter_score(self, own_army, enemy_army):
        own_total = sum(own_army.values())
        enemy_total = sum(enemy_army.values())
        if own_total <= 0 or enemy_total <= 0:
            return 0.0

        score = 0.0
        for own_unit, own_count in own_army.items():
            counters = self.UNIT_COUNTERS.get(own_unit, [])
            if not counters or own_count <= 0:
                continue

            countered_enemy = sum(enemy_army.get(enemy_unit, 0) for enemy_unit in counters)
            if countered_enemy <= 0:
                continue

            enemy_fraction_countered = countered_enemy / enemy_total
            unit_power = self.UNIT_STATS.get(own_unit, {}).get('power', 1.0)
            unit_weight = 0.75 + 0.25 * min(4.0, unit_power / 20.0)
            score += own_count * enemy_fraction_countered * unit_weight

        return score / own_total

    # ============================================================
    # HISTORY HELPERS
    # ============================================================
    def _history_ago(self, seconds_ago):
        if not self.history:
            return None
        current_time = self.history[-1]['time_sec']
        target_time = current_time - seconds_ago
        for snap in reversed(self.history):
            if snap['time_sec'] <= target_time:
                return snap
        return None

    def _rolling_std(self, key, window_sec):
        if len(self.history) < 2:
            return 0.0
        current_time = self.history[-1]['time_sec']
        vals = [snap[key] for snap in self.history if snap['time_sec'] >= current_time - window_sec]
        if len(vals) < 2:
            return 0.0
        return float(np.std(vals))

    # ============================================================
    # SNAPSHOT
    # ============================================================
    def _create_snapshot(
        self,
        loop,
        p1,
        p2,
        inventories,
        camera_history,
        upgrades_completed,
        metadata,
        resource_stats,
        epms,
        recent_losses,
        replay_id,
    ):
        time_sec = round(loop / self.LOOPS_PER_SEC, 1)
        c_p1, c_p2 = self._calc_combat_score_pair(inventories, upgrades_completed, p1, p2)

        a1, w1, sa1, st1, t1 = self._get_known_player_data(inventories, p1)
        a2, w2, sa2, st2, t2 = self._get_known_player_data(inventories, p2)

        res1 = resource_stats[p1]
        res2 = resource_stats[p2]
        inc1 = res1['mi'] + res1['vi']
        inc2 = res2['mi'] + res2['vi']

        sq1 = self._calc_spending_quotient(res1['m'], res1['v'], res1['mi'], res1['vi'])
        sq2 = self._calc_spending_quotient(res2['m'], res2['v'], res2['mi'], res2['vi'])

        scout1 = self._calc_scouting_metrics(camera_history[p1], metadata['start_locs'].get(p2, (0, 0)))
        scout2 = self._calc_scouting_metrics(camera_history[p2], metadata['start_locs'].get(p1, (0, 0)))

        counter_adv_p1 = self._calc_counter_score(a1, a2)
        counter_adv_p2 = self._calc_counter_score(a2, a1)
        counter_adv_diff = counter_adv_p1 - counter_adv_p2

        base_snapshot = {
            'time_sec': time_sec,
            'diff_workers': w1 - w2,
            'diff_combat_score': c_p1 - c_p2,
            'diff_sq': sq1 - sq2,
            'diff_epm': epms[p1] - epms[p2],
            'diff_income': inc1 - inc2,
            'tech_diff': t1 - t2,
            'counter_advantage_diff': counter_adv_diff,
            'p1_workers': w1,
            'p2_workers': w2,
            'p1_combat': c_p1,
            'p2_combat': c_p2,
            'p1_sq': sq1,
            'p2_sq': sq2,
            'p1_income': inc1,
            'p2_income': inc2,
            'p1_army_supply_ratio': sa1 / max(1, st1),
            'p2_army_supply_ratio': sa2 / max(1, st2),
        }

        prev_15 = self.history[-1] if self.history else None
        self.history.append(base_snapshot.copy())
        prev_60 = self._history_ago(60)
        prev_120 = self._history_ago(120)

        delta_workers_p1 = w1 - prev_15['p1_workers'] if prev_15 else 0.0
        delta_workers_p2 = w2 - prev_15['p2_workers'] if prev_15 else 0.0
        delta_combat_p1 = c_p1 - prev_15['p1_combat'] if prev_15 else 0.0
        delta_combat_p2 = c_p2 - prev_15['p2_combat'] if prev_15 else 0.0
        delta_sq_p1 = sq1 - prev_15['p1_sq'] if prev_15 else 0.0
        delta_sq_p2 = sq2 - prev_15['p2_sq'] if prev_15 else 0.0
        delta_diff_workers_15 = base_snapshot['diff_workers'] - prev_15['diff_workers'] if prev_15 else 0.0
        delta_diff_combat_15 = base_snapshot['diff_combat_score'] - prev_15['diff_combat_score'] if prev_15 else 0.0
        delta_diff_sq_15 = base_snapshot['diff_sq'] - prev_15['diff_sq'] if prev_15 else 0.0
        delta_diff_income_15 = base_snapshot['diff_income'] - prev_15['diff_income'] if prev_15 else 0.0

        diff_workers_60s = base_snapshot['diff_workers'] - prev_60['diff_workers'] if prev_60 else 0.0
        diff_combat_60s = base_snapshot['diff_combat_score'] - prev_60['diff_combat_score'] if prev_60 else 0.0
        diff_sq_60s = base_snapshot['diff_sq'] - prev_60['diff_sq'] if prev_60 else 0.0
        diff_income_60s = base_snapshot['diff_income'] - prev_60['diff_income'] if prev_60 else 0.0
        counter_adv_diff_60s = base_snapshot['counter_advantage_diff'] - prev_60['counter_advantage_diff'] if prev_60 else 0.0

        diff_workers_120s = base_snapshot['diff_workers'] - prev_120['diff_workers'] if prev_120 else 0.0
        diff_combat_120s = base_snapshot['diff_combat_score'] - prev_120['diff_combat_score'] if prev_120 else 0.0
        diff_sq_120s = base_snapshot['diff_sq'] - prev_120['diff_sq'] if prev_120 else 0.0
        diff_income_120s = base_snapshot['diff_income'] - prev_120['diff_income'] if prev_120 else 0.0
        counter_adv_diff_120s = base_snapshot['counter_advantage_diff'] - prev_120['counter_advantage_diff'] if prev_120 else 0.0

        worker_growth_p1 = delta_workers_p1 / max(1, prev_15['p1_workers']) if prev_15 else 0.0
        worker_growth_p2 = delta_workers_p2 / max(1, prev_15['p2_workers']) if prev_15 else 0.0

        income_per_worker_p1 = inc1 / max(1, w1)
        income_per_worker_p2 = inc2 / max(1, w2)

        loss_trade_ratio = recent_losses[p2] / max(1, recent_losses[p1])
        army_supply_ratio_diff = base_snapshot['p1_army_supply_ratio'] - base_snapshot['p2_army_supply_ratio']
        army_to_worker_ratio_p1 = sa1 / max(1, w1)
        army_to_worker_ratio_p2 = sa2 / max(1, w2)

        diff_workers_vol_120 = self._rolling_std('diff_workers', 120)
        diff_combat_vol_120 = self._rolling_std('diff_combat_score', 120)
        diff_sq_vol_120 = self._rolling_std('diff_sq', 120)
        diff_income_vol_120 = self._rolling_std('diff_income', 120)
        counter_adv_vol_120 = self._rolling_std('counter_advantage_diff', 120)

        phase = 0 if time_sec < 300 else 1 if time_sec < 900 else 2

        snapshot = {
            'replay_id': replay_id,
            'time_sec': time_sec,
            'phase': phase,

            # differenziali statici
            'diff_combat_score': base_snapshot['diff_combat_score'],
            'diff_workers': base_snapshot['diff_workers'],
            'diff_sq': base_snapshot['diff_sq'],
            'diff_epm': base_snapshot['diff_epm'],
            'diff_income': base_snapshot['diff_income'],
            'tech_diff': base_snapshot['tech_diff'],
            'army_supply_ratio_diff': army_supply_ratio_diff,
            'income_per_worker_diff': income_per_worker_p1 - income_per_worker_p2,
            'counter_advantage_p1': counter_adv_p1,
            'counter_advantage_p2': counter_adv_p2,
            'counter_advantage_diff': counter_adv_diff,

            # raw p1
            'p1_combat': c_p1,
            'p1_workers': w1,
            'p1_sq': sq1,
            'p1_epm': epms[p1],
            'p1_scout': scout1['score'],
            'p1_army_entropy': self._calc_army_entropy(a1),
            'p1_num_unit_types': len(a1),
            'p1_tech_level': t1,
            'p1_army_supply_ratio': base_snapshot['p1_army_supply_ratio'],
            'p1_recent_losses': recent_losses[p1],
            'p1_worker_growth_ratio': worker_growth_p1,
            'p1_income_per_worker': income_per_worker_p1,
            'p1_army_to_worker_ratio': army_to_worker_ratio_p1,

            # raw p2
            'p2_combat': c_p2,
            'p2_workers': w2,
            'p2_sq': sq2,
            'p2_epm': epms[p2],
            'p2_scout': scout2['score'],
            'p2_army_entropy': self._calc_army_entropy(a2),
            'p2_num_unit_types': len(a2),
            'p2_tech_level': t2,
            'p2_army_supply_ratio': base_snapshot['p2_army_supply_ratio'],
            'p2_recent_losses': recent_losses[p2],
            'p2_worker_growth_ratio': worker_growth_p2,
            'p2_income_per_worker': income_per_worker_p2,
            'p2_army_to_worker_ratio': army_to_worker_ratio_p2,

            # delta 15s simmetrici
            'delta_combat_p1': delta_combat_p1,
            'delta_combat_p2': delta_combat_p2,
            'delta_workers_p1': delta_workers_p1,
            'delta_workers_p2': delta_workers_p2,
            'delta_sq_p1': delta_sq_p1,
            'delta_sq_p2': delta_sq_p2,
            'economic_momentum_15': delta_diff_workers_15,
            'combat_momentum_15': delta_diff_combat_15,
            'sq_momentum_15': delta_diff_sq_15,
            'income_momentum_15': delta_diff_income_15,

            # trend 60s / 120s
            'diff_workers_60s': diff_workers_60s,
            'diff_combat_60s': diff_combat_60s,
            'diff_sq_60s': diff_sq_60s,
            'diff_income_60s': diff_income_60s,
            'counter_advantage_diff_60s': counter_adv_diff_60s,
            'diff_workers_120s': diff_workers_120s,
            'diff_combat_120s': diff_combat_120s,
            'diff_sq_120s': diff_sq_120s,
            'diff_income_120s': diff_income_120s,
            'counter_advantage_diff_120s': counter_adv_diff_120s,

            # stabilità / trade quality
            'diff_workers_vol_120': diff_workers_vol_120,
            'diff_combat_vol_120': diff_combat_vol_120,
            'diff_sq_vol_120': diff_sq_vol_120,
            'diff_income_vol_120': diff_income_vol_120,
            'counter_advantage_vol_120': counter_adv_vol_120,
            'loss_trade_ratio': loss_trade_ratio,

            # target
            'p1_wins': 1 if metadata['winner'] == p1 else 0,
        }

        self.prev_snapshot = snapshot
        self.history[-1] = {
            'time_sec': time_sec,
            'diff_workers': snapshot['diff_workers'],
            'diff_combat_score': snapshot['diff_combat_score'],
            'diff_sq': snapshot['diff_sq'],
            'diff_income': snapshot['diff_income'],
            'counter_advantage_diff': snapshot['counter_advantage_diff'],
            'p1_workers': snapshot['p1_workers'],
            'p2_workers': snapshot['p2_workers'],
            'p1_combat': snapshot['p1_combat'],
            'p2_combat': snapshot['p2_combat'],
            'p1_sq': snapshot['p1_sq'],
            'p2_sq': snapshot['p2_sq'],
            'p1_army_supply_ratio': snapshot['p1_army_supply_ratio'],
            'p2_army_supply_ratio': snapshot['p2_army_supply_ratio'],
        }
        return snapshot
