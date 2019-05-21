import os


def execute_cmd(cmd):
    print("execute cmd:{:s}".format(cmd))
    os.system(cmd)


def tap(x, y):
    print("tap :{:d} {:d}".format(x, y))
