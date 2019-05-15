from multiprocessing.pool import Pool

from device.device import Device
from utils.cmd_utils import execute_cmd


class Mobile(Device):

    def __init__(self, device_id):
        super().__init__(device_id)
        self.p = Pool(6)

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
