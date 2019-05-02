import numpy as np

card_dict = {0: "empty", 1: "Furnace", 2: "GoblinBarrel", 3: "DarkPrince", 4: "Prince",
             5: "RoyalHogs", 6: "Giant", 7: "Arrows", 8: "FireSpirit", 9: "Bomber",
             10: "IceGolem", 11: "X-box", 12: "BarbarianHut", 13: "Witch",
             14: "Knight", 15: "Hunter", 16: "Poison", 17: "GoblinHut",
             18: "P.E.K.K.A", 19: "BattleRam", 20: "GiantSnowball", 21: "Musketeer",
             22: "Princess", 23: "Archers", 24: "DartGoblin", 25: "InfernoDragon",
             26: "MegaKnight", 27: "ThreeMusketeer", 28: "IceWizard", 29: "SkeletonArmy",
             30: "HogRider", 31: "Golem", 32: "Fireball", 33: "Valkyrie",
             34: "Zap", 35: "Guards", 36: "IceSpirit", 37: "Tornado",
             38: "BarbarianBarrel", 39: "Rage", 40: "miniP.E.K.K.A", 41: "Miner",
             42: "Wizard", 43: "GiantSkeleton", 44: "BabyDragon", 45: "ElectroDragon",
             46: "InfernoTower", 47: "Balloon", 48: "Minions", 49: "MinionHorde"}

elixir_dict = {"empty": 0, "Furnace": 4, "GoblinBarrel": 3, "DarkPrince": 4, "Prince": 5,
               "RoyalHogs": 5, "Giant": 5, "Arrows": 3, "FireSpirit": 2, "Bomber": 3,
               "IceGolem": 2, "X-box": 6, "BarbarianHut": 7, "Witch": 5,
               "Knight": 3, "Hunter": 4, "Poison": 4, "GoblinHut": 5,
               "P.E.K.K.A": 7, "BattleRam": 4, "GiantSnowball": 2, "Musketeer": 4,
               "Princess": 3, "Archers": 3, "DartGoblin": 3, "InfernoDragon": 4,
               "MegaKnight": 7, "ThreeMusketeer": 10, "IceWizard": 3, "SkeletonArmy": 3,
               "HogRider": 4, "Golem": 8, "Fireball": 4, "Valkyrie": 4,
               "Zap": 2, "Guards": 3, "IceSpirit": 1, "Tornado": 3,
               "BarbarianBarrel": 2, "Rage": 2, "miniP.E.K.K.A": 4, "Miner": 3,
               "Wizard": 5, "GiantSkeleton": 6, "BabyDragon": 4, "ElectroDragon": 5,
               "InfernoTower": 5, "Balloon": 5, "Minions": 3, "MinionHorde": 5}


def parse_running_state(result):
    card_type_state = np.zeros((len(card_dict.keys()) * 4), dtype=np.int32)
    card_available = np.zeros(4, dtype=np.int32)
    card_elixir = np.zeros(4, dtype=np.int32)
    remain_elixir = np.ones(1, dtype=np.int32)
    # 是否溅射 是否飞行 是非远程 是否spell 是否专对建筑
    remain = np.zeros(20, dtype=np.int32)
    for i in range(4):
        card_type_state[i * 50 + result.card_type[i]] = 1
        card_available[i] = result.available[i]
        card_elixir[i] = elixir_dict[card_dict[result.card_type[i]]]
    remain_elixir[0] = result.remain_elixir
    state = np.concatenate([card_type_state, card_available, card_elixir, remain_elixir, remain], axis=0)
    return state
