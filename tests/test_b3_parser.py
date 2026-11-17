"""B3 regression cases for winner correctness and replay-time unit losses."""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

try:
    import sc2reader  # type: ignore
except ImportError:
    # Allow synthetic parser tests in minimal environments without sc2reader.
    sc2reader = types.ModuleType('sc2reader')
    sys.modules['sc2reader'] = sc2reader

from sc2proj.bridge import ReplayBridge
from sc2proj.parser import SnapshotParser


def event(name: str, second: int, **extra):
    return types.SimpleNamespace(name=name, second=second, **extra)


def make_replay(winner_pid=1, n_players=2):
    players = [types.SimpleNamespace(pid=i + 1) for i in range(n_players)]
    def unit(pid, tag, name, second, x):
        return event('UnitBornEvent', second, unit_id=tag, control_pid=pid,
                     unit_type_name=name, location=(x, 0))
    events = [
        unit(1, 11, 'SCV', 0, 0),
        unit(2, 21, 'Probe', 0, 60),
        unit(1, 12, 'Marine', 136, 0),
        event('UnitDiedEvent', 142, unit_id=12),
        unit(2, 22, 'Stalker', 147, 60),
        event('UnitDiedEvent', 160, unit_id=22),
    ]
    replay = types.SimpleNamespace(
        length=types.SimpleNamespace(seconds=210), players=players,
        winner=(types.SimpleNamespace(players=[types.SimpleNamespace(pid=winner_pid)])
                if winner_pid is not None else None),
        tracker_events=events, game_events=[],
        load_game_events=lambda: None,
    )
    return replay


class B3ParserTests(unittest.TestCase):
    def test_missing_winner_is_filtered_not_labeled_zero(self):
        bridge = ReplayBridge()
        with patch.object(sc2reader, 'load_replay', create=True, return_value=make_replay(None)):
            result = bridge.process_replay('unknown.SC2Replay')
        self.assertEqual(result.status, 'filtered')
        self.assertEqual(result.reason, 'missing_winner')
        self.assertEqual(result.metadata['parse_status'], 'filtered')
        self.assertFalse(result.metadata['winner_known'])
        self.assertIsNone(result.dataframe)

    def test_invalid_winner_is_filtered(self):
        bridge = ReplayBridge()
        with patch.object(sc2reader, 'load_replay', create=True, return_value=make_replay(99)):
            result = bridge.process_replay('invalid.SC2Replay')
        self.assertEqual(result.reason, 'invalid_winner_pid')
        self.assertEqual(result.status, 'filtered')
        self.assertIsNone(result.dataframe)

    def test_non_1v1_match_is_filtered(self):
        bridge = ReplayBridge()
        with patch.object(sc2reader, 'load_replay', create=True, return_value=make_replay(1, 3)):
            result = bridge.process_replay('three.SC2Replay')
        self.assertEqual(result.reason, 'not_one_vs_one')
        self.assertEqual(result.metadata['parse_status'], 'filtered')
        self.assertIsNone(result.dataframe)

    def test_recent_losses_detected_after_initial_base_locations(self):
        bridge = ReplayBridge()
        with patch.object(sc2reader, 'load_replay', create=True, return_value=make_replay(2)):
            result = bridge.process_replay('valid.SC2Replay')
        self.assertEqual(result.status, 'ok', result.metadata)
        self.assertTrue(result.metadata['winner_known'])
        df = result.dataframe.set_index('time_sec')
        self.assertEqual(df.loc[150.0, 'p1_recent_losses'], 1)
        self.assertEqual(df.loc[165.0, 'p2_recent_losses'], 1)
        self.assertEqual(df.loc[180.0, 'p1_recent_losses'], 0)
        self.assertTrue((df['p1_wins'] == 0).all())
        self.assertEqual(df.index.min(), 135.0)

    def test_known_p1_winner_produces_label_one(self):
        bridge = ReplayBridge()
        with patch.object(sc2reader, 'load_replay', create=True, return_value=make_replay(1)):
            result = bridge.process_replay('p1.SC2Replay')
        self.assertEqual(result.status, 'ok')
        self.assertTrue((result.dataframe['p1_wins'] == 1).all())

    def test_direct_snapshot_rejects_unknown_winner(self):
        parser = SnapshotParser()
        common = dict(replay_id='unknown', loop=3024, p1_pid=1, p2_pid=2,
                      inventories={1: {}, 2: {}}, camera_history={1: [], 2: []},
                      upgrades_completed={1: set(), 2: set()},
                      base_locations={1: (0., 0.), 2: (60., 0.)},
                      resource_stats={1: {'m': 0, 'v': 0, 'mi': 0, 'vi': 0},
                                      2: {'m': 0, 'v': 0, 'mi': 0, 'vi': 0}},
                      epms={1: 0, 2: 0}, recent_losses={1: 0, 2: 0})
        with self.assertRaisesRegex(ValueError, 'known winner'):
            parser.build_snapshot(**common, winner_pid=None)
        with self.assertRaisesRegex(ValueError, 'known winner'):
            parser.build_snapshot(**common, winner_pid=3)
        self.assertEqual(parser.build_snapshot(**common, winner_pid=1)['p1_wins'], 1)


if __name__ == '__main__':
    unittest.main()
