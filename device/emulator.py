from device.device import Device
from multiprocessing.pool import Pool
import sys

import cv2
import numpy as np
import win32gui
from PyQt5.QtWidgets import QApplication
from pymouse import PyMouse

from utils.cmd_utils import execute_cmd


class Emulator(Device):

    def __init__(self, device_id, window_name):
        super().__init__(device_id)
        self.p = Pool(6)
        self.offset_w = 0
        self.offset_h = 0
        self.scale = 0.5
        self.hwnd = win32gui.FindWindow(None, window_name)
        self.mouse = PyMouse()
        self.mode = 0

    def execute_tap(self, x, y):
        print("tap :{:d} {:d}".format(x, y))
        self.mouse.release(self.offset_w + x // 2, self.offset_h + y // 2, button=1)

    def execute_swipe(self, x1, y1, x2, y2):
        print("swipe:{:d} {:d} {:d} {:d}".format(x1, y1, x2, y2))
        self.mouse.press(x1, y1)
        self.mouse.move(x2, y2)
        self.mouse.release(x2, y2)

    def tap_button(self, button):
        if self.mode == 0:
            cmd = "adb  -s {:s} shell input tap {:d} {:d}".format(self.device_id, button[0] // 2, button[1] // 2)
            self.p.apply_async(execute_cmd, args={cmd})
        else:
            self.execute_tap(button[0], button[1])

    def swipe(self, action):
        if self.mode == 0:
            cmd = "adb -s {:s} shell input swipe {:d} {:d} {:d} {:d} 300".format(self.device_id,
                                                                                 action[0] // 2,
                                                                                 action[1] // 2,
                                                                                 action[2] // 2,
                                                                                 action[3] // 2)
            self.p.apply_async(execute_cmd, args={cmd})
        else:
            self.execute_swipe(self.offset_w + action[0] // 2, self.offset_h + action[1] // 2,
                               self.offset_w + action[2] // 2, self.offset_h + action[2] // 2)

    def update_locate(self, window_rect, img_shape):
        self.offset_w = window_rect[0] + 2
        self.offset_h = window_rect[1] + img_shape[0] - 2 - 960

    def get_frame(self):
        app = QApplication(sys.argv)
        screens = QApplication.screens()
        if len(screens) == 0:
            print("screen :0")
            return None
        window = screens[0].grabWindow(self.hwnd)
        if win32gui.IsWindow(self.hwnd) == 1:
            cv2.waitKey(400)
            window_rect = win32gui.GetWindowRect(self.hwnd)
            img = window.toImage()
            img_np = self.convertQImageToMat(img)
            self.update_locate(window_rect, img_np.shape)
            img_np = img_np[-2 - 960:-2, 2:-2, 0:3].copy()
            if img_np.shape[0] != 960:
                raise Exception("错误的分辨率")
            return img_np
        else:
            return None

    def convertQImageToMat(self, incomingImage):
        #  Converts a QImage into an opencv MAT format
        incomingImage = incomingImage.convertToFormat(4)

        width = incomingImage.width()
        height = incomingImage.height()

        ptr = incomingImage.bits()
        ptr.setsize(incomingImage.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)  # Copies the data
        return arr
