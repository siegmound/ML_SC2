from __future__ import annotations

from typing import Dict, List

UNIT_COUNTERS: Dict[str, List[str]] = {
    "Marine": ["Baneling", "Colossus", "HighTemplar", "Disruptor", "WidowMine", "SiegeTank", "Lurker"],
    "Marauder": ["Immortal", "Archon", "VoidRay", "SiegeTank", "Lurker"],
    "Zealot": ["Hellbat", "Baneling", "Roach", "Archon", "Colossus", "Lurker"],
    "Stalker": ["Marauder", "Immortal", "Zergling", "Roach"],
    "Immortal": ["Marine", "Zergling", "Hydralisk", "Mutalisk"],
    "Colossus": ["Viking", "Corruptor", "VoidRay", "BroodLord", "Marauder"],
    "Zergling": ["Hellion", "Hellbat", "Colossus", "Archon", "Baneling", "Adept"],
    "Baneling": ["Stalker", "Marauder", "Archon", "Immortal", "Banshee", "Phoenix"],
    "Roach": ["Immortal", "Marauder", "Stalker", "VoidRay"],
    "Hydralisk": ["Colossus", "Disruptor", "SiegeTank", "HighTemplar", "Baneling"],
    "Lurker": ["Disruptor", "Colossus", "SiegeTank", "Carrier", "Battlecruiser", "Liberator", "Tempest"],
    "Mutalisk": ["Phoenix", "Archon", "Thor", "Marine", "Viking", "Liberator"],
    "Corruptor": ["VoidRay", "Phoenix", "Marine", "Stalker", "Hydralisk"],
    "Battlecruiser": ["Corruptor", "VoidRay", "Tempest", "Viking", "Viper"],
}
