import os
import cv2
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from brain.base_brain import BaseBrain

from device.emulator import Emulator
from device.mobile import Mobile
from game.clash_royal_env import ClashRoyalEnv

if __name__ == '__main__':

    i = 0

    # root = "/home/chengli/data/gym_data/clash_royal"
    root = "F:\\gym_data\\clash_royal"

    device_id = "cd9faa7f"

    # device = Mobile(device_id, host_address)
    device = Emulator(device_id, "one")

    host = ClashRoyalEnv(root, device, mode=ClashRoyalEnv.MODE["friend_battle_host"], name="host")

    while True:
        frame, state_code = device.get_frame()

        if frame is not None:
            host_observation = host.frame_step(frame, None)
            cv2.imshow("img", frame)
            cv2.waitKey(1)
        else:
            if state_code == -1:
                print("没有信号")
