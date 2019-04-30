import ctypes
import cv2
import numpy as np
import time

from config import CARD_DICT
from utils.c_lib_utils import convert2pymat, Result, STATE_DICT
import os
import os.path as osp

if __name__ == '__main__':

    address = "http://127.0.0.1:35013/device/cd9faa7f/video.flv"
    # address = "./scan/s1.mp4"
    capture = cv2.VideoCapture(address)
    cv2.namedWindow('image')
    ll = ctypes.cdll.LoadLibrary
    lib = ll("./lib/libc_opencv.so")

    lib.detect_frame.restype = Result
    i = 0

    root = "../vysor/"


    def init_game(gameId):

        root_error = osp.join(root, str(gameId) + "/error")
        root_run = osp.join(root, str(gameId) + "/run")
        root_finish = osp.join(root, str(gameId) + "/finish")

        os.makedirs(root_error)
        os.makedirs(root_run)
        os.makedirs(root_finish)

        lib.init_game(gameId)

        return root_error, root_run, root_finish


    gameId = int(time.time() * 1000)
    root_error, root_run, root_finish = init_game(gameId)

    while True:

        state, img = capture.read()

        if state:
            if i % 5 != 0:
                i = i + 1
                continue

            img = cv2.resize(img, (540, 960))

            result = Result()

            pymat = convert2pymat(img)

            result = lib.detect_frame(pymat, result)

            cv2.imshow('image', img)
            print("frame-------------" + str(i))
            if result.frame_state == STATE_DICT["ERROR_STATE"]:
                print("error   spent:" + str(result.milli))
                cv2.imwrite(osp.join(root_error, "{:d}.jpg".format(i)), img)
            elif result.frame_state == STATE_DICT["MENU_STATE"]:
                print("id:" + str(gameId) + "  in hall:" + str(result.index) + "  spent:" + str(result.milli))
            elif result.frame_state == STATE_DICT["RUNNING_STATE"]:
                type_array = np.array(result.card_type)
                available = np.array(result.available)
                cv2.imwrite(osp.join(root_run, "{:d}.jpg".format(result.frame_index)), img)
                print("id:" + str(gameId) + "  running:" + str(result.frame_index) + "  spent:" + str(result.milli))
                print("{:s}:{:d}--{:s}:{:d}--{:s}:{:d}--{:s}:{:d}".format(CARD_DICT[type_array[0]],
                                                                          available[0],
                                                                          CARD_DICT[type_array[1]],
                                                                          available[1],
                                                                          CARD_DICT[type_array[2]],
                                                                          available[2],
                                                                          CARD_DICT[type_array[3]],
                                                                          available[3],
                                                                          ))
            elif result.frame_state == STATE_DICT["FINISH_STATE"]:
                cv2.imwrite(osp.join(root_finish, "{:d}.jpg".format(result.frame_index)), img)
                print("id:" + str(gameId) + "  is_finish:" + str(result.win) + "  spent:" + str(result.milli))
                root_error, root_run, root_finish = init_game(int(time.time() * 1000))

            cv2.waitKey(1)
            i += 1
        else:
            break

    cv2.destroyAllWindows()
