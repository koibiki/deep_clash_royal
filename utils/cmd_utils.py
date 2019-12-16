import os
from utils.logger_utils import logger


def execute_cmd(cmd):
    logger.info("execute cmd:{:s}".format(cmd))
    os.system(cmd)


def tap(x, y):
    logger.info("tap :{:d} {:d}".format(x, y))
