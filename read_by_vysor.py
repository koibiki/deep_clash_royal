import cv2
import subprocess
import time
import os
from multiprocessing import Pool, freeze_support, Process
import random

from game.clash_royal import ClashRoyal


def execute_command(shell_cmd):
    print("execute_command:" + shell_cmd)
    os.system(shell_cmd)


if __name__ == '__main__':
    freeze_support()
    # img = cv2.imread("3437.jpg")  # 加载图片
    p = Pool(4)

    ix, iy = -1, -1


    def check_mouse(event, x, y, flags, param):
        global ix, iy
        global i
        global shell_cmd

        if event == cv2.EVENT_LBUTTONDOWN:
            print("ix:{:d}  iy:{:d}   x:{:d}  y:{:d}".format(ix, iy, x, y))
            if ix == -1 or iy == -1:
                ix, iy = x, y
            else:
                width = x - ix
                height = y - iy
                if abs(width) <= 5 and abs(height) <= 5:
                    print(" i = {:d} tap point:={:d}, {:d}".format(i, x, y))
                    shell_cmd = "adb shell input tap {:d} {:d} ".format(x * 2, y * 2)
                    p = Process(target=execute_command, args=(shell_cmd,))
                    p.start()

                else:
                    print(" i = {:d} swipe {:d}, {:d}, {:d}, {:d}".format(i, ix, iy, x, y))
                    shell_cmd = "adb shell input swipe {:d} {:d} {:d} {:d} 300".format(ix * 2, iy * 2, x * 2, y * 2)
                    p = Process(target=execute_command, args=(shell_cmd,))
                    p.start()
                ix, iy = -1, -1


    address = "http://127.0.0.1:35013/device/cd9faa7f/video.flv"

    capture = cv2.VideoCapture(address)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', check_mouse)
    time_time = time.time()

    root = "../vysor/"
    clash_royal = ClashRoyal(root, "cd9faa7f")
    gameId = int(time_time)
    clash_royal._init_game(gameId)
    i = 0
    while True:
        state, img = capture.read()

        if state:
            img = cv2.resize(img, (540, 960))
            # result = clash_royal.frame_step(img)

            cv2.imshow('image', img)
            cv2.waitKey(1)

        else:
            break

    cv2.destroyAllWindows()
