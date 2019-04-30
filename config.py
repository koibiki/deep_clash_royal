from easydict import EasyDict as edict
import os
import os.path as osp

cfg = edict()

CARD_DICT = {0: "empty", 1: "Furnace", 2: "GoblinBarrel", 3: "DarkPrince", 4: "Prince",
             5: "RoyalHogs", 6: "Giant", 7: "Arrows", 8: "FireSpirit", 9: "Bomber",
             10: "IceGolen", 11: "X-box", 12: "BarbarianHut", 13: "Witch",
             14: "Knight", 15: "Hunter", 16: "Poison", 17: "GoblinHut",
             18: "P.E.K.K.A", 19: "BattleRam", 20: "GiantSnowball", 21: "Musketeer",
             22: "Princess", 23: "Archers", 24: "DartGoblin", 25: "InfernoDragon",
             26: "MegaKnight", 27: "ThreeMusketeer", 28: "IceWizard", 29: "SkeletonArmy",
             30: "HogRider", 31: "Golem", 32: "Fireball", 33: "Valkyrie",
             34: "Zap", 35: "Guards", 36: "IceSpirit", 37: "Tornado",
             38: "BarbarianBarrel", 39: "Rage", 40: "miniP.E.K.K.A", 41: "Miner",
             42: "Wizard", 43: "GiantSkeleton", 44: "BabyDragon", 45: "ElectroDragon",
             46: "InfernoTower", 47: "Ballon", 48: "Minions", 49: "MinionHorde"}


cfg.INPUT_SIZE = (224, 224, 3)
cfg.PATH = edict()
cfg.PATH.ROOT_DIR = os.getcwd()
cfg.PATH.TBOARD_SAVE_DIR = osp.abspath(osp.join(os.getcwd(), 'logs'))
cfg.PATH.MODEL_SAVE_DIR = osp.abspath(osp.join(os.getcwd(), 'checkpoints'))
cfg.PATH.TFLITE_MODEL_SAVE_DIR = osp.abspath(osp.join(os.getcwd(), 'tf_lite_model'))

cfg.ACTION_NUM = 5
cfg.ACTION_X = 7
cfg.ACTION_Y = 8

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.BATCH_SIZE = 2
cfg.TRAIN.INPUT_SHAPE = (cfg.TRAIN.BATCH_SIZE, cfg.INPUT_SIZE[0], cfg.INPUT_SIZE[1], cfg.INPUT_SIZE[2])
cfg.TRAIN.LEARNING_RATE = 0.001
cfg.TRAIN.LR_DECAY_STEPS = 10000
cfg.TRAIN.LR_DECAY_RATE = 0.9
cfg.TRAIN.EPOCHS = 500
cfg.TRAIN.DISPLAY_STEP = 1
cfg.TRAIN.SAVE_MODEL_STEP = 500
cfg.TRAIN.GPU_MEMORY_FRACTION = 0.5
cfg.TRAIN.TF_ALLOW_GROWTH = True

# TEST
cfg.TEST = edict()
cfg.TEST.BATCH_SIZE = 2
cfg.TEST.INPUT_SHAPE = (cfg.TEST.BATCH_SIZE, cfg.INPUT_SIZE[0], cfg.INPUT_SIZE[1], cfg.INPUT_SIZE[2])
