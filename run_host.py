import time

import cv2

from brain.base_brain import BaseBrain
from brain.policy import PolicyGradient
from device.emulator import Emulator
from device.mobile import Mobile
from game.clash_royal import ClashRoyal

if __name__ == '__main__':

    i = 0

    # root = "/home/chengli/data/gym_data/clash_royal"
    root = "D:\\gym_data\\clash_royal"

    # device_id = "cd9faa7f"
    device_id = "127.0.0.1:62001"

    # host_address = "http://127.0.0.1:46539/device/" + device_id + "/video.flv"

    # device = Mobile(device_id, host_address)
    device = Emulator(device_id, "夜神模拟器")

    host = ClashRoyal(root, device, mode=ClashRoyal.MODE["friend_battle_host"], name="host")

    brain = PolicyGradient(host.img_shape, host.state_shape, BaseBrain.BrainType["runner"], "host")

    while True:
        frame = device.get_frame()

        if frame is not None:
            host_observation = host.frame_step(frame)
            if host_observation is not None:
                host_action = brain.choose_action(host_observation)
                host.step(host_observation, host_action)

            if host.game_start and host.game_finish and host.retry <= 1:
                brain.update_episode_result(host.get_rate_of_winning())
                brain.record_battle_result()
                brain.load_model()
        else:
            print("没有信号")
