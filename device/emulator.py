from device.device import Device
from multiprocessing.pool import Pool
import sys
import time
import cv2
import numpy as np
import win32gui
from PyQt5.QtWidgets import QApplication

from utils.cmd_utils import execute_cmd
from utils.logger_utils import logger


class Emulator(Device):

    def __init__(self, device_id, window_name):
        super().__init__(device_id)
        self.p = Pool(6)
        self.offset_w = 0
        self.offset_h = 0
        self.scale = 0.5
        self.hwnd = win32gui.FindWindow(None, window_name)
        self.mode = 0
        self.pre_time = time.time() * 1000

    def tap_button(self, button):
        cmd = "adb  -s {:s} shell input tap {:d} {:d}".format(self.device_id, button[0] // 2, button[1] // 2)
        self.p.apply_async(execute_cmd, args={cmd})

    def swipe(self, action):

        cmd = "adb -s {:s} shell input swipe {:d} {:d} {:d} {:d} 300".format(self.device_id,
                                                                             action[0] // 2,
                                                                             action[1] // 2,
                                                                             action[2] // 2,
                                                                             action[3] // 2)
        self.p.apply_async(execute_cmd, args={cmd})

    def update_locate(self, window_rect, img_shape):
        self.offset_w = window_rect[0] + 2
        self.offset_h = window_rect[1] + img_shape[0] - 2 - 960

    def get_frame(self):
        time_mill = time.time() * 1000
        if time_mill - self.pre_time < 500:
            return [None, 1]
        else:
            self.pre_time = time_mill
        app = QApplication(sys.argv)
        screens = QApplication.screens()
        if len(screens) == 0:
            logger.info("screen :0")
            return [None, -1]
        window = screens[0].grabWindow(self.hwnd)
        if win32gui.IsWindow(self.hwnd) == 1:
            window_rect = win32gui.GetWindowRect(self.hwnd)
            img = window.toImage()
            img_np = self.convertQImageToMat(img)
            self.update_locate(window_rect, img_np.shape)
            img_np = img_np[-2 - 960:-2, 2:-2, 0:3].copy()
            if img_np.shape[0] != 960:
                raise Exception("错误的分辨率")
            return [img_np, 1]
        else:
            return [None, -1]

    def convertQImageToMat(self, incomingImage):
        #  Converts a QImage into an opencv MAT format
        incomingImage = incomingImage.convertToFormat(4)

        width = incomingImage.width()
        height = incomingImage.height()

        ptr = incomingImage.bits()
        ptr.setsize(incomingImage.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)  # Copies the data
        return arr
