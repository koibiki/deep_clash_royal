import os
import os.path as osp

from easydict import EasyDict as edict

cfg = edict()

cfg.NUM_CLASSES = 13
cfg.IMAGE_SHAPE = (64, 64, 3)

cfg.PATH = edict()
cfg.PATH.ROOT_DIR = os.getcwd()
cfg.PATH.MODEL_SAVE_DIR = osp.abspath(osp.join(os.getcwd(), 'checkpoints'))

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.BATCH_SIZE = 128
cfg.TRAIN.LEARNING_RATE = 0.001
cfg.TRAIN.LR_DECAY_STEPS = 10000
cfg.TRAIN.LR_DECAY_RATE = 0.98
cfg.TRAIN.EPOCHS = 50000
cfg.TRAIN.SAVE_MODEL_STEP = 100

# VALID
cfg.VALID = edict()
cfg.VALID.BATCH_SIZE = 4
