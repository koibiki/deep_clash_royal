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

elixir_dict = {"empty": 0, "Furnace": 0.4, "GoblinBarrel": 0.3, "DarkPrince": 0.4, "Prince": 0.5,
               "RoyalHogs": 0.5, "Giant": 0.5, "Arrows": 0.3, "FireSpirit": 0.2, "Bomber": 0.3,
               "IceGolem": 0.2, "X-box": 0.6, "BarbarianHut": 0.7, "Witch": 0.5,
               "Knight": 0.3, "Hunter": 0.4, "Poison": 0.4, "GoblinHut": 0.5,
               "P.E.K.K.A": 0.7, "BattleRam": 0.4, "GiantSnowball": 0.2, "Musketeer": 0.4,
               "Princess": 0.3, "Archers": 0.3, "DartGoblin": 0.3, "InfernoDragon": 0.4,
               "MegaKnight": 0.7, "ThreeMusketeer": 1.0, "IceWizard": 0.3, "SkeletonArmy": 0.3,
               "HogRider": 0.4, "Golem": 0.8, "Fireball": 0.4, "Valkyrie": 0.4,
               "Zap": 0.2, "Guards": 0.3, "IceSpirit": 0.1, "Tornado": 0.3,
               "BarbarianBarrel": 0.2, "Rage": 0.2, "miniP.E.K.K.A": 0.4, "Miner": 0.3,
               "Wizard": 0.5, "GiantSkeleton": 0.6, "BabyDragon": 0.4, "ElectroDragon": 0.5,
               "InfernoTower": 0.5, "Balloon": 0.5, "Minions": 0.3, "MinionHorde": 0.5}


def parse_running_state(result):
    card_type_state = parse_card_state(result.card_type, result.available)
    remain_elixir = np.ones(1, dtype=np.float32)
    spent_time = np.ones(1, dtype=np.float32)
    double_elixir = np.ones(1, dtype=np.float32)
    dead_im = np.ones(1, dtype=np.float32)
    hp = np.ones(2, dtype=np.float32)  # 双方血量

    remain_elixir[0] = result.remain_elixir / 10
    spent_time[0] = result.time / 60
    double_elixir[0] = 1 if result.time > 60 * 2 - 1 else 0
    dead_im[0] = 1 if result.time > 60 * 3 - 1 else 0
    state = np.concatenate([card_type_state, hp, remain_elixir, spent_time, double_elixir, dead_im], axis=0)
    return state


def parse_card_state(card_type, available):
    card_type_state = np.zeros(92 * 3, dtype=np.float32)
    for i in range(4):
        if card_type[i] != 0:
            card_type_state[card_type[i] - 1] = 1
            card_type_state[92 + card_type[i] - 1] = available[i]
            card_type_state[92 * 2 + card_type[i] - 1] = elixir_dict[card_dict[card_type[i]]]
    return card_type_state
