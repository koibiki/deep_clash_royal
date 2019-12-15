from multiprocessing.pool import Pool

from device.device import Device
from utils.cmd_utils import execute_cmd
import cv2
import time


class Mobile(Device):

    def __init__(self, device_id, address):
        super().__init__(device_id)
        self.address = address
        self.p = Pool(6)
        self.capture = None
        self.pre_time = time.time() * 1000

    def tap_button(self, button):
        cmd = "adb  -s {:s} shell input tap {:d} {:d}".format(self.device_id, button[0], button[1])
        self.p.apply_async(execute_cmd, args={cmd})

    def swipe(self, action):
        cmd = "adb -s {:s} shell input swipe {:d} {:d} {:d} {:d} 300".format(self.device_id,
                                                                             action[0],
                                                                             action[1],
                                                                             action[2],
                                                                             action[3])
        self.p.apply_async(execute_cmd, args={cmd})

    def get_frame(self):
        if self.capture is None:
            self.capture = cv2.VideoCapture(self.address)

        state, img = self.capture.read()
        if state:
            time_mill = time.time() * 1000
            if time_mill - self.pre_time >= 500:
                self.pre_time = time_mill
                return [cv2.resize(img, (540, 960)), 0]
            else:
                return [None, 0]
        else:
            self.capture.release()
            self.capture = None
            time.sleep(10)
            return [None, -1]
