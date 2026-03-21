# sc2_counters.py

"""
Sistema di counter e relazioni tra unità per StarCraft 2
Contiene matchups, efficacia e sinergie
"""

# ============================================================================
# SISTEMA DI COUNTER (unità -> counter efficaci)
# ============================================================================

UNIT_COUNTERS = {
    # ========== TERRAN ==========
    "Marine":        ["Baneling", "Colossus", "HighTemplar", "Disruptor", "WidowMine", "Hellion", "Hellbat", "SiegeTank", "Lurker"],
    "Marauder":      ["Immortal", "Archon", "VoidRay", "SiegeTank", "Lurker"],
    "Reaper":        ["Queen", "Adept", "Stalker", "Zergling", "Marine"],
    "Ghost":         ["Marine", "Zergling", "Zealot", "Oracle", "Observer", "Overseer"],
    "Hellion":       ["Roach", "Stalker", "Marauder", "Queen"],
    "Hellbat":       ["Roach", "Stalker", "Immortal", "Archon"],
    "WidowMine":     ["Observer", "Overseer", "Raven", "Oracle"], # Necessitano di detezione
    "SiegeTank":     ["Immortal", "VoidRay", "Mutalisk", "Banshee", "Tempest", "BroodLord", "Ravager"],
    "Cyclone":       ["Zergling", "Marine", "Adept", "Stalker"],
    "Thor":          ["Immortal", "Zergling", "VoidRay", "Corruptor", "Viper"],
    "Viking":        ["Phoenix", "Stalker", "Hydralisk", "Archon"],
    "Medivac":       ["Viking", "Phoenix", "Mutalisk", "Corruptor", "Stalker", "Hydralisk"],
    "Raven":         ["Ghost", "Phoenix", "Viking", "Viper", "Corruptor"],
    "Banshee":       ["Phoenix", "Viking", "Mutalisk", "Stalker", "Hydralisk", "Marine", "Oracle"],
    "Battlecruiser": ["Corruptor", "VoidRay", "Tempest", "Viking", "Viper"],
    "Liberator":     ["Viking", "Phoenix", "Corruptor", "Stalker", "Hydralisk"],
    
    # ========== PROTOSS ==========
    "Zealot":        ["Hellbat", "Baneling", "Roach", "Archon", "Colossus", "Lurker"],
    "Stalker":       ["Marauder", "Immortal", "Zergling", "Roach"],
    "Sentry":        ["Marine", "Zergling", "Adept", "Stalker"],
    "Adept":         ["Roach", "Marauder", "Marine", "Stalker"],
    "HighTemplar":   ["Ghost", "Viper", "Ravager", "Marine", "Zergling"],
    "DarkTemplar":   ["Raven", "Observer", "Overseer", "Oracle"], # Necessitano di detezione
    "Archon":        ["Ghost", "Thor", "Ultralisk", "Immortal", "Lurker", "BroodLord"],
    "Immortal":      ["Marine", "Zergling", "Hydralisk", "Mutalisk"],
    "Colossus":      ["Viking", "Corruptor", "VoidRay", "BroodLord", "Marauder"],
    "Disruptor":     ["Mutalisk", "Phoenix", "Viking", "Marine", "Zergling", "Viper"],
    "Observer":      ["Raven", "Oracle", "Overseer"],
    "WarpPrism":     ["Viking", "Phoenix", "Mutalisk", "Corruptor", "Stalker", "Hydralisk"],
    "Phoenix":       ["Corruptor", "Viking", "Hydralisk", "Marine", "Stalker", "Thor"],
    "VoidRay":       ["Viking", "Mutalisk", "Corruptor", "Marine", "Hydralisk"],
    "Oracle":        ["Marine", "Stalker", "Hydralisk", "Queen", "Viking", "Phoenix"],
    "Carrier":       ["Corruptor", "Viking", "VoidRay", "Tempest", "Hydralisk"],
    "Tempest":       ["Viking", "Corruptor", "Phoenix", "VoidRay"],
    "Mothership":    ["Corruptor", "Viking", "VoidRay", "Ghost", "Viper"],
    
    # ========== ZERG ==========
    "Zergling":      ["Hellion", "Hellbat", "Colossus", "Archon", "Baneling", "Adept"],
    "Baneling":      ["Stalker", "Marauder", "Archon", "Immortal", "Banshee", "Phoenix"], 
    "Roach":         ["Immortal", "Marauder", "Stalker", "VoidRay"],
    "Ravager":       ["Immortal", "Stalker", "Viking", "Banshee", "Phoenix", "VoidRay"],
    "Hydralisk":     ["Colossus", "Disruptor", "SiegeTank", "HighTemplar", "Baneling"],
    "Lurker":        ["Disruptor", "Colossus", "SiegeTank", "Carrier", "Battlecruiser", "Liberator", "Tempest"],
    "Infestor":      ["Ghost", "Viper", "HighTemplar", "Marine", "Zergling"],
    "SwarmHost":     ["Hellion", "Hellbat", "Phoenix", "Adept", "Marine"],
    "Ultralisk":     ["Immortal", "Marauder", "VoidRay", "Liberator", "Ghost", "Thor"],
    "Mutalisk":      ["Phoenix", "Archon", "Thor", "Marine", "Viking", "Liberator"],
    "Corruptor":     ["VoidRay", "Phoenix", "Marine", "Stalker", "Hydralisk"],
    "BroodLord":     ["Viking", "Corruptor", "Phoenix", "VoidRay", "Banshee"],
    "Viper":         ["Ghost", "HighTemplar", "Phoenix", "Viking", "Marine"],
    "Queen":         ["Marauder", "Stalker", "Immortal", "Zergling"],
    "Overseer":      ["Viking", "Phoenix", "Mutalisk", "Corruptor"]
}

# ============================================================================
# SISTEMA DI SINERGIE (unità che lavorano bene insieme)
# ============================================================================

UNIT_SYNERGIES = {
    "Marine": ["Medivac", "Marauder", "SiegeTank", "WidowMine", "Ghost", "Raven"],
    "Marauder": ["Medivac", "Marine", "SiegeTank", "Ghost", "Raven", "Cyclone"],
    "Zealot": ["Sentry", "Immortal", "Colossus", "Archon", "Disruptor", "HighTemplar"],
    "Stalker": ["Sentry", "Immortal", "Colossus", "Observer", "Disruptor", "Oracle"],
    "Zergling": ["Baneling", "Roach", "Hydralisk", "Queen", "Infestor", "Viper"],
    "Roach": ["Ravager", "Hydralisk", "Queen", "Viper", "Infestor", "SwarmHost"],
    "Hydralisk": ["Roach", "Queen", "Viper", "Ultralisk", "Infestor", "SwarmHost"],
    "Immortal": ["Zealot", "Sentry", "Stalker", "Colossus", "Archon", "Disruptor"],
    "Colossus": ["Zealot", "Stalker", "Sentry", "Immortal", "Archon", "HighTemplar"],
    "SiegeTank": ["Marine", "Marauder", "Medivac", "Viking", "WidowMine", "Raven"],
    "Thor": ["Marine", "Marauder", "Medivac", "Viking", "Raven", "Battlecruiser"],
    "Mutalisk": ["Corruptor", "Viper", "Queen", "Hydralisk", "Infestor", "BroodLord"],
    "Carrier": ["Mothership", "Tempest", "VoidRay", "Oracle", "Phoenix", "HighTemplar"],
    "Battlecruiser": ["Raven", "Viking", "Medivac", "Ghost", "Liberator", "Thor"],
    "Phoenix": ["VoidRay", "Oracle", "Carrier", "Tempest", "Mothership", "Observer"],
    "VoidRay": ["Phoenix", "Oracle", "Carrier", "Tempest", "Mothership", "Colossus"],
    "Baneling": ["Zergling", "Roach", "Hydralisk", "Ultralisk", "Infestor", "Queen"],
    "Archon": ["Zealot", "Stalker", "Immortal", "Colossus", "HighTemplar", "DarkTemplar"],
    "Ravager": ["Roach", "Hydralisk", "Queen", "Viper", "Lurker", "Infestor"],
    "Liberator": ["Viking", "Raven", "Medivac", "Marine", "Marauder", "Thor"],
    "Disruptor": ["Immortal", "Colossus", "Stalker", "Sentry", "Archon", "HighTemplar"],
    "Viper": ["Roach", "Hydralisk", "Mutalisk", "Corruptor", "Ultralisk", "Infestor"],
    "Oracle": ["Phoenix", "VoidRay", "Tempest", "Mothership", "Observer", "WarpPrism"],
    "Tempest": ["Carrier", "Mothership", "VoidRay", "Phoenix", "Oracle", "Colossus"],
}

# ============================================================================
# EFFICACIA CONTRO TIPO (ground, air, armored, light, biological, etc.)
# ============================================================================

UNIT_EFFECTIVENESS = {
    # Unità buone contro armored
    "vs_armored": ["Immortal", "Marauder", "Cyclone", "Corruptor", "VoidRay", "Tempest", "Battlecruiser"],
    
    # Unità buone contro light
    "vs_light": ["Hellion", "Colossus", "Baneling", "PsiStorm", "WidowMine", "Disruptor", "LiberatorAG"],
    
    # Unità buone contro biological
    "vs_biological": ["Ghost", "PsiStorm", "Baneling", "Raven", "Infestor", "Viper"],
    
    # Unità buone contro mech
    "vs_mechanical": ["Immortal", "Cyclone", "VoidRay", "Corruptor", "Tempest", "Battlecruiser"],
    
    # Unità buone contro psionic
    "vs_psionic": ["Ghost", "Emp", "Feedback", "NeuralParasite"],
    
    # Unità buone contro massive
    "vs_massive": ["Tempest", "Corruptor", "VoidRay", "Battlecruiser", "Thor", "Immortal"],
    
    # Unità buone contro strutture
    "vs_structures": ["SiegeTank", "Colossus", "Ultralisk", "Battlecruiser", "Tempest", "BroodLord"],
    
    # Unità buone contro air
    "vs_air": ["Viking", "Phoenix", "Hydralisk", "Marine", "Stalker", "MissileTurret"],
    
    # Unità buone contro ground
    "vs_ground": ["SiegeTank", "Colossus", "Baneling", "PsiStorm", "Liberator", "Disruptor"],
}

# ============================================================================
# COMPOSIZIONI TIPICHE PER RAZZA
# ============================================================================

TYPICAL_COMPOSITIONS = {
    # Terran
    "Bio": ["Marine", "Marauder", "Medivac", "WidowMine", "Ghost"],
    "BioTank": ["Marine", "Marauder", "SiegeTank", "Medivac", "Viking"],
    "Mech": ["Hellion", "SiegeTank", "Thor", "Viking", "Raven"],
    "SkyTerran": ["Viking", "Raven", "Battlecruiser", "Liberator", "Banshee"],
    "MarineMedivac": ["Marine", "Medivac"],
    "MarauderMedivac": ["Marauder", "Medivac"],
    "CycloneHellion": ["Cyclone", "Hellion"],
    
    # Protoss
    "Gateway": ["Zealot", "Stalker", "Sentry", "Adept"],
    "Robo": ["Immortal", "Colossus", "Disruptor", "Observer"],
    "AirToss": ["VoidRay", "Phoenix", "Oracle", "Carrier"],
    "Skytoss": ["Carrier", "Tempest", "Mothership", "VoidRay"],
    "ChargelotArchon": ["Zealot", "Archon", "HighTemplar"],
    "BlinkStalker": ["Stalker", "Observer", "Sentry"],
    "DisruptorImmortal": ["Disruptor", "Immortal", "Stalker"],
    
    # Zerg
    "LingBane": ["Zergling", "Baneling", "Queen"],
    "RoachHydra": ["Roach", "Hydralisk", "Queen", "Viper"],
    "MutaLingBane": ["Mutalisk", "Zergling", "Baneling", "Queen"],
    "UltraLingBane": ["Ultralisk", "Zergling", "Baneling", "Queen"],
    "RoachRavager": ["Roach", "Ravager", "Queen", "Viper"],
    "HydraLurker": ["Hydralisk", "Lurker", "Queen", "Viper"],
    "BroodLordInfestor": ["BroodLord", "Infestor", "Corruptor", "Queen"],
    "SwarmHost": ["SwarmHost", "Roach", "Hydralisk", "Queen"],
}

# ============================================================================
# MATCHUP SPECIFICI (razza vs razza)
# ============================================================================

MATCHUP_COUNTERS = {
    # TvZ (Terran vs Zerg)
    "TvZ": {
        "Bio": ["LingBane", "RoachHydra", "UltraLingBane"],
        "BioTank": ["RoachHydra", "MutaLingBane", "RoachRavager"],
        "Mech": ["RoachHydra", "SwarmHost", "HydraLurker"],
        "MarineMedivac": ["LingBane", "Roach", "Hydralisk"],
        "Battlecruiser": ["Corruptor", "Hydralisk", "Queen", "Viper"],
    },
    
    # TvP (Terran vs Protoss)
    "TvP": {
        "Bio": ["Colossus", "Storm", "Disruptor", "ChargelotArchon"],
        "BioTank": ["Colossus", "Immortal", "Disruptor", "BlinkStalker"],
        "Mech": ["Immortal", "VoidRay", "Tempest", "Carrier"],
        "MarineMedivac": ["Colossus", "Storm", "Archon", "Zealot"],
        "Battlecruiser": ["VoidRay", "Tempest", "Viking", "Stalker"],
    },
    
    # PvZ (Protoss vs Zerg)
    "PvZ": {
        "Gateway": ["Roach", "Hydralisk", "LingBane"],
        "Robo": ["Hydralisk", "Corruptor", "RoachHydra"],
        "AirToss": ["Hydralisk", "Corruptor", "Queen", "Viper"],
        "Skytoss": ["Corruptor", "Hydralisk", "Queen", "Viper"],
        "ChargelotArchon": ["Roach", "Hydralisk", "Ultralisk"],
        "BlinkStalker": ["Roach", "Hydralisk", "LingBane"],
    },
    
    # PvT (Protoss vs Terran)
    "PvT": {
        "Colossus": ["Viking", "Ghost", "Marine"],
        "Storm": ["Ghost", "Emp", "Marauder"],
        "Disruptor": ["Marauder", "Ghost", "Viking"],
        "Immortal": ["Marine", "Marauder", "Viking", "Ghost"],
        "Carrier": ["Viking", "Marine", "Ghost", "Raven"],
        "BlinkStalker": ["Marine", "Marauder", "SiegeTank"],
    },
    
    # ZvT (Zerg vs Terran)
    "ZvT": {
        "LingBane": ["Hellion", "WidowMine", "Hellbat"],
        "Roach": ["Marauder", "SiegeTank", "Cyclone"],
        "Mutalisk": ["Thor", "Viking", "WidowMine", "MissileTurret"],
        "Ultralisk": ["Ghost", "Marauder", "Liberator", "SiegeTank"],
        "RoachHydra": ["SiegeTank", "Hellbat", "Thor"],
        "HydraLurker": ["SiegeTank", "Liberator", "Battlecruiser"],
    },
    
    # ZvP (Zerg vs Protoss)
    "ZvP": {
        "Roach": ["Immortal", "Colossus", "Stalker"],
        "Hydralisk": ["Colossus", "Storm", "Disruptor", "Archon"],
        "Mutalisk": ["Phoenix", "Storm", "Archon", "PhotonCannon"],
        "SwarmHost": ["Colossus", "Disruptor", "Phoenix", "Tempest"],
        "UltraLingBane": ["Immortal", "Colossus", "Archon"],
        "BroodLordInfestor": ["Tempest", "VoidRay", "Phoenix", "Carrier"],
    },
}

# ============================================================================
# TIERING E TIMING
# ============================================================================

UNIT_TIER = {
    # Tier 1 (Early Game)
    "T1": ["Marine", "Zergling", "Zealot", "Reaper", "Adept", "Roach", "Stalker"],
    
    # Tier 2 (Mid Game)
    "T2": ["Marauder", "Hydralisk", "Immortal", "SiegeTank", "WidowMine", "Oracle", "Banshee"],
    
    # Tier 3 (Late Game)
    "T3": ["Ghost", "Thor", "Battlecruiser", "Ultralisk", "Colossus", "Carrier", "Tempest", "BroodLord"],
    
    # Support Units
    "Support": ["Medivac", "Raven", "Observer", "Overseer", "Queen", "Viper", "Infestor"],
    
    # Spellcasters
    "Spellcaster": ["Ghost", "HighTemplar", "Infestor", "Raven", "Viper", "Oracle"],
}

# ============================================================================
# TIMING ATTACKS
# ============================================================================

TIMING_ATTACKS = {
    # Terran Timings
    "Terran": {
        "2-1-1": {"time": "5:00", "units": ["Marine", "Medivac"], "upgrades": ["Stimpack"]},
        "3Rax": {"time": "4:30", "units": ["Marine", "Marauder"], "upgrades": ["Stimpack", "CombatShield"]},
        "1-1-1": {"time": "6:00", "units": ["Marine", "SiegeTank", "Medivac"], "upgrades": ["SiegeTech"]},
        "BattlecruiserRush": {"time": "7:30", "units": ["Battlecruiser"], "upgrades": ["Yamato"]},
        "HellbatTiming": {"time": "5:30", "units": ["Hellion", "Hellbat"], "upgrades": ["BlueFlame"]},
    },
    
    # Protoss Timings
    "Protoss": {
        "4Gate": {"time": "5:30", "units": ["Zealot", "Stalker", "Sentry"], "upgrades": ["WarpGate"]},
        "BlinkTiming": {"time": "6:30", "units": ["Stalker"], "upgrades": ["Blink"]},
        "ImmortalSent": {"time": "6:00", "units": ["Immortal", "Sentry"], "upgrades": []},
        "OracleHarass": {"time": "4:30", "units": ["Oracle"], "upgrades": []},
        "ChargelotArchon": {"time": "8:00", "units": ["Zealot", "Archon"], "upgrades": ["Charge"]},
    },
    
    # Zerg Timings
    "Zerg": {
        "Speedling": {"time": "3:30", "units": ["Zergling"], "upgrades": ["MetabolicBoost"]},
        "RoachRavager": {"time": "5:30", "units": ["Roach", "Ravager"], "upgrades": ["GlialReconstitution"]},
        "BanelingBust": {"time": "4:30", "units": ["Zergling", "Baneling"], "upgrades": ["MetabolicBoost"]},
        "MutaHarass": {"time": "7:00", "units": ["Mutalisk"], "upgrades": []},
        "UltraTiming": {"time": "9:00", "units": ["Ultralisk", "Zergling"], "upgrades": ["ChitinousPlating"]},
    },
}

# ============================================================================
# BUILDS COUNTERS (build order -> counter build)
# ============================================================================

BUILD_COUNTERS = {
    # Terran Builds
    "2-1-1": ["RoachRavager", "BlinkStalker", "CycloneHellion"],
    "3Rax": ["Colossus", "BanelingBust", "ImmortalSent"],
    "1-1-1": ["RoachHydra", "DisruptorImmortal", "MutaLingBane"],
    "BattlecruiserRush": ["Corruptor", "Hydralisk", "Tempest"],
    "Mech": ["RoachHydra", "HydraLurker", "SwarmHost"],
    
    # Protoss Builds
    "4Gate": ["Roach", "Bunker", "SiegeTank"],
    "BlinkStalker": ["Roach", "WidowMine", "Hellion"],
    "Skytoss": ["Corruptor", "Hydralisk", "Viking"],
    "ChargelotArchon": ["Hellbat", "WidowMine", "Roach"],
    "DisruptorImmortal": ["MarineMedivac", "MutaLingBane", "Cyclone"],
    
    # Zerg Builds
    "LingBane": ["Hellion", "WidowMine", "Colossus"],
    "RoachRavager": ["SiegeTank", "Immortal", "VoidRay"],
    "MutaLingBane": ["Thor", "Phoenix", "MissileTurret"],
    "UltraLingBane": ["Ghost", "Liberator", "Immortal"],
    "BroodLordInfestor": ["Viking", "Tempest", "Battlecruiser"],
}

# ============================================================================
# UPGRADE COUNTERS
# ============================================================================

UPGRADE_COUNTERS = {
    # Terran Upgrades
    "Stimpack": ["Baneling", "PsiStorm", "Disruptor"],
    "CombatShield": ["Colossus", "Baneling", "PsiStorm"],
    "SiegeTech": ["Immortal", "VoidRay", "Mutalisk"],
    "InfernalPreIgniter": ["Roach", "Hydralisk", "Marine"],
    
    # Protoss Upgrades
    "Blink": ["Roach", "Hellion", "WidowMine"],
    "Charge": ["Hellbat", "WidowMine", "Baneling"],
    "PsionicStorm": ["Ghost", "Emp", "Raven"],
    "GraviticDrive": ["Viking", "Hydralisk", "Marine"],
    
    # Zerg Upgrades
    "MetabolicBoost": ["Hellion", "Colossus", "WidowMine"],
    "GlialReconstitution": ["Marauder", "Stalker", "Cyclone"],
    "Burrow": ["ScannerSweep", "Observer", "Overseer"],
    "FlyerAttacks": ["PhotonCannon", "MissileTurret", "Thor"],
}