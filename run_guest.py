import time

import cv2

from brain.ddpg import DDPG
from brain.policy import PolicyGradient
from device.mobile import Mobile
from game.clash_royal import ClashRoyal

if __name__ == '__main__':

    i = 0

    # root = "/home/chengli/data/gym_data/clash_royal"
    root = "D:\\gym_data\\clash_royal"

    device_id = "cd9faa7f"

    address = "http://127.0.0.1:55481/device/" + device_id + "/video.flv"
    mobile = Mobile(device_id, address)

    guest = ClashRoyal(root, device=mobile, mode=ClashRoyal.MODE["friend_battle_guest"], name="guest")

    brain = PolicyGradient(guest.img_shape, guest.state_shape, DDPG.BrainType["runner"], "guest")

    while True:
        frame, state_code = mobile.get_frame()

        if frame is not None:
            host_observation = guest.frame_step(frame)
            if host_observation is not None:
                host_action = brain.choose_action(host_observation)
                guest.step(host_observation, host_action)

            if guest.game_start and guest.game_finish and guest.retry <= 1:
                brain.update_episode_result(guest.get_rate_of_winning())
                brain.record_battle_result()
                brain.load_model()
        else:
            if state_code == -1:
                print("没有信号")
                guest.reset()
