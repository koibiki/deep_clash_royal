import os


def execute_cmd(cmd):
    print("execute cmd:{:s}".format(cmd))
    os.system(cmd)
