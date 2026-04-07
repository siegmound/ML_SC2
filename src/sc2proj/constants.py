from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet

LOOPS_PER_SECOND: float = 22.4
DEFAULT_SNAPSHOT_STEP_SEC: int = 15
DEFAULT_EARLY_CUT_SEC: int = 120
DEFAULT_MIN_MATCH_SEC: int = 180
DEFAULT_MAX_MATCH_SEC: int = 1800

UNIT_ALIASES: Dict[str, str] = {
    "SiegeTankSieged": "SiegeTank",
    "VikingFighter": "Viking",
    "VikingAssault": "Viking",
    "LiberatorAG": "Liberator",
    "WidowMineBurrowed": "WidowMine",
    "HellionTank": "Hellbat",
    "ObserverSiegeMode": "Observer",
    "OverseerSiegeMode": "Overseer",
    "WarpPrismPhasing": "WarpPrism",
    "AdeptPhaseShift": "Adept",
    "LurkerBurrowed": "Lurker",
    "RoachBurrowed": "Roach",
    "ZerglingBurrowed": "Zergling",
    "BanelingBurrowed": "Baneling",
    "DroneBurrowed": "Drone",
}

@dataclass(frozen=True)
class UnitSpec:
    power: float
    supply: float
    role: str
    tech_level: int = 0
    upgrade_category: str = "None"

UNIT_SPECS: Dict[str, UnitSpec] = {
    "SCV": UnitSpec(1, 1, "worker"),
    "Probe": UnitSpec(1, 1, "worker"),
    "Drone": UnitSpec(1, 1, "worker"),
    "Marine": UnitSpec(2, 1, "army", 1, "Infantry"),
    "Marauder": UnitSpec(5, 2, "army", 2, "Infantry"),
    "Reaper": UnitSpec(3, 1, "army", 1, "Infantry"),
    "Ghost": UnitSpec(15, 2, "army", 3, "Infantry"),
    "Hellion": UnitSpec(3, 2, "army", 1, "Vehicle"),
    "Hellbat": UnitSpec(3, 2, "army", 1, "Vehicle"),
    "WidowMine": UnitSpec(4, 2, "army", 2, "Vehicle"),
    "SiegeTank": UnitSpec(12, 3, "army", 2, "Vehicle"),
    "Cyclone": UnitSpec(7, 3, "army", 2, "Vehicle"),
    "Thor": UnitSpec(25, 6, "army", 3, "Vehicle"),
    "Viking": UnitSpec(8, 2, "army", 2, "Ship"),
    "Medivac": UnitSpec(6, 2, "army", 2, "Ship"),
    "Raven": UnitSpec(12, 2, "army", 3, "Ship"),
    "Banshee": UnitSpec(9, 3, "army", 2, "Ship"),
    "Battlecruiser": UnitSpec(45, 6, "army", 3, "Ship"),
    "Liberator": UnitSpec(10, 3, "army", 2, "Ship"),
    "Zealot": UnitSpec(3, 2, "army", 1, "GroundWeapons"),
    "Stalker": UnitSpec(5, 2, "army", 1, "GroundWeapons"),
    "Sentry": UnitSpec(6, 2, "army", 2, "GroundWeapons"),
    "Adept": UnitSpec(4, 2, "army", 1, "GroundWeapons"),
    "HighTemplar": UnitSpec(18, 2, "army", 3, "GroundWeapons"),
    "DarkTemplar": UnitSpec(15, 2, "army", 3, "GroundWeapons"),
    "Archon": UnitSpec(22, 4, "army", 3, "GroundWeapons"),
    "Immortal": UnitSpec(18, 4, "army", 2, "GroundWeapons"),
    "Colossus": UnitSpec(25, 6, "army", 3, "GroundWeapons"),
    "Disruptor": UnitSpec(15, 3, "army", 3, "GroundWeapons"),
    "Observer": UnitSpec(2, 1, "army", 2, "AirWeapons"),
    "WarpPrism": UnitSpec(5, 2, "army", 2, "AirWeapons"),
    "Phoenix": UnitSpec(7, 2, "army", 2, "AirWeapons"),
    "VoidRay": UnitSpec(12, 4, "army", 3, "AirWeapons"),
    "Oracle": UnitSpec(10, 3, "army", 2, "AirWeapons"),
    "Carrier": UnitSpec(40, 6, "army", 3, "AirWeapons"),
    "Tempest": UnitSpec(30, 5, "army", 3, "AirWeapons"),
    "Mothership": UnitSpec(60, 8, "army", 3, "AirWeapons"),
    "Zergling": UnitSpec(0.5, 0.5, "army", 1, "MeleeAttacks"),
    "Baneling": UnitSpec(2, 0.5, "army", 1, "MeleeAttacks"),
    "Roach": UnitSpec(4, 2, "army", 1, "MissileAttacks"),
    "Ravager": UnitSpec(6, 3, "army", 2, "MissileAttacks"),
    "Hydralisk": UnitSpec(6, 2, "army", 2, "MissileAttacks"),
    "Lurker": UnitSpec(15, 3, "army", 3, "MissileAttacks"),
    "Infestor": UnitSpec(15, 2, "army", 3, "None"),
    "SwarmHost": UnitSpec(8, 3, "army", 3, "MissileAttacks"),
    "Ultralisk": UnitSpec(35, 6, "army", 3, "MeleeAttacks"),
    "Mutalisk": UnitSpec(7, 2, "army", 2, "FlyerAttacks"),
    "Corruptor": UnitSpec(9, 2, "army", 2, "FlyerAttacks"),
    "BroodLord": UnitSpec(30, 6, "army", 3, "FlyerAttacks"),
    "Viper": UnitSpec(20, 3, "army", 3, "None"),
    "Queen": UnitSpec(8, 2, "army", 1, "None"),
}

EARLY_GAME_UNITS: FrozenSet[str] = frozenset({"Marine", "Zergling", "Zealot", "Reaper", "Adept", "Roach"})
MID_GAME_UNITS: FrozenSet[str] = frozenset({"Marauder", "Stalker", "Hydralisk", "Immortal", "SiegeTank", "WidowMine", "Banshee"})
LATE_GAME_UNITS: FrozenSet[str] = frozenset({"Ghost", "Thor", "Battlecruiser", "Ultralisk", "Colossus", "Carrier", "Tempest", "BroodLord"})

UPGRADE_MULTIPLIERS: Dict[str, tuple[str, ...]] = {
    "Marine": ("Stimpack", "ShieldWall"),
    "Marauder": ("Stimpack", "PunisherGrenades"),
    "SiegeTank": ("SiegeTech", "SmartServos"),
    "Battlecruiser": ("Yamato",),
    "Zealot": ("Charge",),
    "Stalker": ("BlinkTech",),
    "HighTemplar": ("PsiStormTech",),
    "Colossus": ("ExtendedThermalLance",),
    "Zergling": ("MetabolicBoost", "AdrenalGlands"),
    "Baneling": ("CentrificalHooks",),
    "Roach": ("GlialReconstitution",),
    "Hydralisk": ("MuscularAugments", "GroovedSpines"),
    "Lurker": ("LurkerRange",),
    "Ultralisk": ("ChitinousPlating", "AnabolicSynthesis"),
}
