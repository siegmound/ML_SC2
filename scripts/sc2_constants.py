# sc2_constants_v2_fixed.py

# ============================================================================
# 1. NAME NORMALIZATION (Aliasing)
# ============================================================================
UNIT_MAPPING = {
    # Terran
    "SiegeTankSieged": "SiegeTank",
    "VikingFighter": "Viking",
    "VikingAssault": "Viking",
    "LiberatorAG": "Liberator",
    "WidowMineBurrowed": "WidowMine",
    "HellionTank": "Hellbat",
    "MULE": "SCV",
    # Protoss
    "ObserverSiegeMode": "Observer",
    "OverseerSiegeMode": "Overseer",
    "WarpPrismPhasing": "WarpPrism",
    "AdeptPhaseShift": "Adept",
    # Zerg
    "LurkerBurrowed": "Lurker",
    "RavagerBurrowed": "Ravager",
    "DroneBurrowed": "Drone",
    "RoachBurrowed": "Roach",
    "ZerglingBurrowed": "Zergling",
    "BanelingBurrowed": "Baneling",
    "QueenBurrowed": "Queen",
    "HydraliskBurrowed": "Hydralisk",
    "InfestorBurrowed": "Infestor",
    "UltraliskBurrowed": "Ultralisk",
    "SwarmHostBurrowed": "SwarmHost",
}


def _mk(power, supply, unit_type, tech_level, attack_upg_cats=None, armor_upg_cats=None, shield_upgrade=False):
    return {
        "power": power,
        "supply": supply,
        "type": unit_type,
        "tech_level": tech_level,
        "attack_upg_cats": attack_upg_cats or [],
        "armor_upg_cats": armor_upg_cats or [],
        "shield_upgrade": shield_upgrade,
    }


UNIT_STATS = {
    # TERRAN
    "SCV":           _mk(1,   1,   "worker", 0, ["TerranInfantryWeapons"], ["TerranInfantryArmor"]),
    "Marine":        _mk(2,   1,   "army",   0, ["TerranInfantryWeapons"], ["TerranInfantryArmor"]),
    "Marauder":      _mk(5,   2,   "army",   1, ["TerranInfantryWeapons"], ["TerranInfantryArmor"]),
    "Reaper":        _mk(3,   1,   "army",   0, ["TerranInfantryWeapons"], ["TerranInfantryArmor"]),
    "Ghost":         _mk(15,  2,   "army",   3, ["TerranInfantryWeapons"], ["TerranInfantryArmor"]),
    "Hellion":       _mk(3,   2,   "army",   1, ["TerranVehicleWeapons"], ["TerranVehicleAndShipPlating"]),
    "Hellbat":       _mk(3,   2,   "army",   1, ["TerranVehicleWeapons"], ["TerranVehicleAndShipPlating"]),
    "WidowMine":     _mk(4,   2,   "army",   2, ["TerranVehicleWeapons"], ["TerranVehicleAndShipPlating"]),
    "SiegeTank":     _mk(12,  3,   "army",   2, ["TerranVehicleWeapons"], ["TerranVehicleAndShipPlating"]),
    "Cyclone":       _mk(7,   3,   "army",   2, ["TerranVehicleWeapons"], ["TerranVehicleAndShipPlating"]),
    "Thor":          _mk(25,  6,   "army",   3, ["TerranVehicleWeapons"], ["TerranVehicleAndShipPlating"]),
    "Viking":        _mk(8,   2,   "army",   2, ["TerranShipWeapons"], ["TerranVehicleAndShipPlating"]),
    "Medivac":       _mk(6,   2,   "army",   2, [], ["TerranVehicleAndShipPlating"]),
    "Banshee":       _mk(9,   3,   "army",   2, ["TerranShipWeapons"], ["TerranVehicleAndShipPlating"]),
    "Raven":         _mk(12,  2,   "army",   3, [], ["TerranVehicleAndShipPlating"]),
    "Battlecruiser": _mk(45,  6,   "army",   3, ["TerranShipWeapons"], ["TerranVehicleAndShipPlating"]),
    "Liberator":     _mk(10,  3,   "army",   3, ["TerranShipWeapons"], ["TerranVehicleAndShipPlating"]),

    # PROTOSS
    "Probe":         _mk(1,   1,   "worker", 0, ["ProtossGroundWeapons"], ["ProtossGroundArmor"], True),
    "Zealot":        _mk(3,   2,   "army",   0, ["ProtossGroundWeapons"], ["ProtossGroundArmor"], True),
    "Stalker":       _mk(5,   2,   "army",   1, ["ProtossGroundWeapons"], ["ProtossGroundArmor"], True),
    "Sentry":        _mk(6,   2,   "army",   1, [], ["ProtossGroundArmor"], True),
    "Adept":         _mk(4,   2,   "army",   1, ["ProtossGroundWeapons"], ["ProtossGroundArmor"], True),
    "HighTemplar":   _mk(18,  2,   "army",   3, [], ["ProtossGroundArmor"], True),
    "DarkTemplar":   _mk(15,  2,   "army",   2, ["ProtossGroundWeapons"], ["ProtossGroundArmor"], True),
    "Archon":        _mk(22,  4,   "army",   3, ["ProtossGroundWeapons"], ["ProtossGroundArmor"], True),
    "Immortal":      _mk(18,  4,   "army",   2, ["ProtossGroundWeapons"], ["ProtossGroundArmor"], True),
    "Colossus":      _mk(25,  6,   "army",   3, ["ProtossGroundWeapons"], ["ProtossGroundArmor"], True),
    "Disruptor":     _mk(15,  3,   "army",   3, [], ["ProtossGroundArmor"], True),
    "Observer":      _mk(2,   1,   "army",   1, [], ["ProtossAirArmor"], True),
    "Overseer":      _mk(2,   1,   "army",   1, [], ["ZergFlyerCarapace"]),
    "WarpPrism":     _mk(5,   2,   "army",   2, [], ["ProtossAirArmor"], True),
    "Phoenix":       _mk(7,   2,   "army",   2, ["ProtossAirWeapons"], ["ProtossAirArmor"], True),
    "VoidRay":       _mk(12,  4,   "army",   2, ["ProtossAirWeapons"], ["ProtossAirArmor"], True),
    "Oracle":        _mk(10,  3,   "army",   2, ["ProtossAirWeapons"], ["ProtossAirArmor"], True),
    "Carrier":       _mk(40,  6,   "army",   3, ["ProtossAirWeapons"], ["ProtossAirArmor"], True),
    "Tempest":       _mk(30,  5,   "army",   3, ["ProtossAirWeapons"], ["ProtossAirArmor"], True),
    "Mothership":    _mk(60,  8,   "army",   3, ["ProtossAirWeapons"], ["ProtossAirArmor"], True),

    # ZERG
    "Drone":         _mk(1,   1,   "worker", 0, ["ZergMeleeWeapons"], ["ZergGroundCarapace"]),
    "Zergling":      _mk(0.5, 0.5, "army",   0, ["ZergMeleeWeapons"], ["ZergGroundCarapace"]),
    "Baneling":      _mk(2,   0.5, "army",   1, ["ZergMeleeWeapons"], ["ZergGroundCarapace"]),
    "Roach":         _mk(4,   2,   "army",   1, ["ZergMissileWeapons"], ["ZergGroundCarapace"]),
    "Ravager":       _mk(6,   3,   "army",   2, ["ZergMissileWeapons"], ["ZergGroundCarapace"]),
    "Hydralisk":     _mk(6,   2,   "army",   2, ["ZergMissileWeapons"], ["ZergGroundCarapace"]),
    "Lurker":        _mk(15,  3,   "army",   3, ["ZergMissileWeapons"], ["ZergGroundCarapace"]),
    "Infestor":      _mk(15,  2,   "army",   3, [], ["ZergGroundCarapace"]),
    "SwarmHost":     _mk(8,   3,   "army",   3, [], ["ZergGroundCarapace"]),
    "Ultralisk":     _mk(35,  6,   "army",   3, ["ZergMeleeWeapons"], ["ZergGroundCarapace"]),
    "Mutalisk":      _mk(7,   2,   "army",   2, ["ZergFlyerWeapons"], ["ZergFlyerCarapace"]),
    "Corruptor":     _mk(9,   2,   "army",   2, ["ZergFlyerWeapons"], ["ZergFlyerCarapace"]),
    "BroodLord":     _mk(30,  6,   "army",   3, ["ZergFlyerWeapons"], ["ZergFlyerCarapace"]),
    "Viper":         _mk(20,  3,   "army",   3, [], ["ZergFlyerCarapace"]),
    "Queen":         _mk(8,   2,   "army",   1, [], ["ZergGroundCarapace"]),
}


EARLY_GAME_UNITS = {"Marine", "Zergling", "Zealot", "Reaper", "Adept", "Roach"}
MID_GAME_UNITS = {"Marauder", "Stalker", "Hydralisk", "Immortal", "SiegeTank", "WidowMine", "Banshee"}
LATE_GAME_UNITS = {"Ghost", "Thor", "Battlecruiser", "Ultralisk", "Colossus", "Carrier", "Tempest", "BroodLord"}
SPLASH_DAMAGE_UNITS = {"SiegeTank", "Colossus", "Baneling", "HighTemplar", "Disruptor", "Lurker", "WidowMine"}
DETECTORS = {"Raven", "Observer", "Overseer", "Oracle"}
BASES = {"CommandCenter", "OrbitalCommand", "PlanetaryFortress", "Nexus", "Hatchery", "Lair", "Hive"}


UNIT_TECH_MULTIPLIERS = {
    # TERRAN
    "Marine": ["Stimpack", "ShieldWall"],
    "Marauder": ["Stimpack", "PunisherGrenades"],
    "Ghost": ["PersonalCloaking", "GhostMoebiusReactor"],
    "Reaper": ["ReaperSpeed"],
    "Hellion": ["HighCapacityBarrels"],
    "Hellbat": ["HighCapacityBarrels"],
    "WidowMine": ["DrillClaws"],
    "SiegeTank": ["SiegeTech", "SmartServos"],
    "Cyclone": ["MagFieldLaunchers"],
    "Banshee": ["BansheeCloak", "BansheeHyperflightRotors"],
    "Raven": ["RavenCorvidReactor"],
    "Medivac": ["MedivacCaduceusReactor"],
    "Liberator": ["LiberatorAGRangeUpgrade"],
    "Battlecruiser": ["BattlecruiserEnableSpecialRefits", "Yamato"],

    # PROTOSS
    "Zealot": ["Charge"],
    "Stalker": ["BlinkTech"],
    "Adept": ["AdeptPiercingAttack"],
    "HighTemplar": ["PsiStormTech"],
    "DarkTemplar": ["DarkTemplarBlinkUpgrade"],
    "Observer": ["ObserverGraviticBooster"],
    "WarpPrism": ["WarpPrismGraviticDrive"],
    "Colossus": ["ExtendedThermalLance"],
    "VoidRay": ["VoidRaySpeedUpgrade"],
    "Tempest": ["TempestRangeUpgrade"],

    # ZERG
    "Zergling": ["MetabolicBoost", "AdrenalGlands"],
    "Baneling": ["CentrificalHooks"],
    "Roach": ["GlialReconstitution", "TunnelingClaws"],
    "Hydralisk": ["MuscularAugments", "GroovedSpines"],
    "Lurker": ["LurkerRange", "DiggingClaws"],
    "Ultralisk": ["ChitinousPlating", "AnabolicSynthesis"],
    "Infestor": ["InfestorEnergyUpgrade", "NeuralParasite"],
}
