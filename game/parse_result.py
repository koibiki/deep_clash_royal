import numpy as np

CARD_TYPE = 93

card_dict = {0: "empty", 1: "Lightning", 2: "Furnace", 3: "GoblinBarrel", 4: "DarkPrince",
             5: "Prince", 6: "RoyalHogs", 7: "Freeze", 8: "Giant", 9: "Bowler",
             10: "Mortar", 11: "Arrows", 12: "FireSpirit", 13: "Bomber", 14: "Bandit",
             15: "GoblinGiant", 16: "SpearGoblin", 17: "IceGolem", 18: "X-box", 19: "TheLog",
             20: "BarbarianHut", 21: "Witch", 22: "Knight", 23: "Hunter", 24: "Poison",
             25: "GoblinHut", 26: "P.E.K.K.A", 27: "BattleRam", 28: "Barbarians", 29: "Tombstone",
             30: "GiantSnowball", 31: "Executioner", 32: "CannonCart", 33: "Musketeer", 34: "Princess",
             35: "Archers", 36: "DartGoblin", 37: "InfernoDragon", 38: "Rascals", 39: "MegaKnight",
             40: "ThreeMusketeer", 41: "RamRider", 42: "IceWizard", 43: "SkeletonBarrel", 44: "MegaMinion",
             45: "Tesla", 46: "Goblins", 47: "SkeletonArmy", 48: "GoblinGang", 49: "Bats",
             50: "Lumberjack", 51: "HogRider", 52: "Golem", 53: "CloneSpell", 54: "Fireball",
             55: "Valkyrie", 56: "Zap", 57: "Guards", 58: "IceSpirit", 59: "NightWitch",
             60: "Tornado", 61: "BarbarianBarrel", 62: "WallBreakers", 63: "Rage", 64: "miniP.E.K.K.A",
             65: "Miner", 66: "Wizard", 67: "GiantSkeleton", 68: "Cannon", 69: "Zappies",
             70: "Graveyard", 71: "BabyDragon", 72: "LavaPups", 73: "Sparky", 74: "FlyingMachine",
             75: "ElectroDragon", 76: "InfernoTower", 77: "Balloon", 78: "BomberTower", 79: "Minions",
             80: "MagicArcher", 81: "MinionHorde", 82: "LavaHound", 83: "Rocket", 84: "ElectorWizard",
             85: "remain0", 86: "remain1", 87: "remain2", 88: "remain3", 89: "remain4",
             90: "remain5", 91: "remain6", 92: "remain7"
             }

elixir_dict = {"empty": 0.0, "Lightning": 0.6, "Furnace": 0.4, "GoblinBarrel": 0.2, "DarkPrince": 0.4,
               "Prince": 0.5, "RoyalHogs": 0.5, "Freeze": 0.4, "Giant": 0.5, "Bowler": 0.5,
               "Mortar": 0.4, "Arrows": 0.3, "FireSpirit": 0.2, "Bomber": 0.3, "Bandit": 0.0,
               "GoblinGiant": 0.6, "SpearGoblin": 0.2, "IceGolem": 0.2, "X-box": 0.6, "TheLog": 0.2,
               "BarbarianHut": 0.7, "Witch": 0.5, "Knight": 0.3, "Hunter": 0.4, "Poison": 0.4,
               "GoblinHut": 0.5, "P.E.K.K.A": 0.7, "BattleRam": 0.4, "Barbarians": 0.5, "Tombstone": 0.3,
               "GiantSnowball": 0.2, "Executioner": 0.0, "CannonCart": 0.5, "Musketeer": 0.4, "Princess": 0.3,
               "Archers": 0.3, "DartGoblin": 0.3, "InfernoDragon": 0.4, "Rascals": 0.5, "MegaKnight": 0.7,
               "ThreeMusketeer": 0.9, "RamRider": 0.5, "IceWizard": 0.3, "SkeletonBarrel": 0.3, "MegaMinion": 0.3,
               "Tesla": 0.4, "Goblins": 0.2, "SkeletonArmy": 0.3, "GoblinGang": 0.3, "Bats": 0.2,
               "Lumberjack": 0.0, "HogRider": 0.4, "Golem": 0.8, "CloneSpell": 0.0, "Fireball": 0.4,
               "Valkyrie": 0.4, "Zap": 0.2, "Guards": 0.3, "IceSpirit": 0.1, "NightWitch": 0.0,
               "Tornado": 0.3, "BarbarianBarrel": 0.2, "WallBreakers": 0.3, "Rage": 0.2, "miniP.E.K.K.A": 0.4,
               "Miner": 0.3, "Wizard": 0.5, "GiantSkeleton": 0.6, "Cannon": 0.3, "Zappies": 0.0,
               "Graveyard": 0.5, "BabyDragon": 0.4, "LavaPups": 0.0, "Sparky": 0.0, "FlyingMachine": 0.4,
               "ElectroDragon": 0.5, "InfernoTower": 0.5, "Balloon": 0.5, "BomberTower": 0.4, "Minions": 0.3,
               "MagicArcher": 0.0, "MinionHorde": 0.5, "LavaHound": 0.0, "Rocket": 0.6, "ElectorWizard": 0.0,
               "remain0": 0.0, "remain1": 0.0, "remain2": 0.0, "remain3": 0.0, "remain4": 0.0,
               "remain5": 0.0, "remain6": 0.0, "remain7": 0.0
               }


def parse_frame_state(result):
    card_type = np.array(result.card_type)
    card_available = np.array(result.available)
    remain_elixir = np.ones(1, dtype=np.float32)
    spent_time = np.ones(1, dtype=np.float32)
    double_elixir = np.ones(1, dtype=np.float32)
    dead_im = np.ones(1, dtype=np.float32)
    hp = np.ones(2, dtype=np.float32)  # 双方血量

    remain_elixir[0] = result.remain_elixir / 10
    spent_time[0] = result.time / 60
    double_elixir[0] = 1 if result.time > 60 * 2 - 1 else 0
    dead_im[0] = 1 if result.time > 60 * 3 - 1 else 0
    state = np.concatenate([card_type, card_available, hp, remain_elixir, spent_time, double_elixir, dead_im], axis=0)
    return state


def parse_running_state(state):
    card_type_state = parse_card_state(state[:4].astype(np.int32), state[4:8].astype(np.int32))
    state_vector = np.concatenate([card_type_state, state[8:]], axis=0)
    return state_vector


def parse_card_state(card_type, available):
    card_type_state = np.zeros(1 + 92 * 3, dtype=np.float32)
    card_type_state[0] = 1
    for i in range(4):
        if card_type[i] != 0:
            card_type_state[card_type[i]] = available[i]
            card_type_state[92 + 1 + card_type[i] - 1] = 1
            card_type_state[92 * 2 + 1 + card_type[i] - 1] = elixir_dict[card_dict[card_type[i]]]
    return card_type_state
