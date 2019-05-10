import time

import cv2

from brain.base_brain2 import BaseBrain
from game.clash_royal import ClashRoyal

if __name__ == '__main__':

    i = 0

    # root = "/home/chengli/data/gym_data/clash_royal"
    root = "/home/holaverse/work/07battle_filed/clash_royal"

    # device_id = "70a7da50"
    device_id = "cd9faa7f"

    host = ClashRoyal(root, device_id=device_id, mode=ClashRoyal.MODE["friend_battle_host"], name="host")

    base_brain = BaseBrain(host, BaseBrain.BrainType["runner"], )

    host_address = "http://127.0.0.1:35013/device/" + device_id + "/video.flv"
    # address = "http://127.0.0.1:46539/device/" + device_id + "/video.flv"

    host_capture = cv2.VideoCapture(host_address)

    while True:
        i += 1
        host_state, host_img = host_capture.read()

        if host_state:

            if i % 20 != 0:
                continue

            host_img = cv2.resize(host_img, (540, 960))

            # cv2.imshow('image', img)
            # cv2.waitKey(0)

            host_observation = host.frame_step(host_img)
            if host_observation is not None:
                host_action = base_brain.choose_action(host_observation)
                host.step(host_observation, host_action)

            if host.game_start and host.game_finish and host.retry <= 1:
                base_brain.update_episode_result(host.get_rate_of_winning())
                base_brain.record_battle_result()
                base_brain.load_model()

        else:
            print("没有信号..")
            host_capture.release()
            time.sleep(30)
            host_capture = cv2.VideoCapture(host_address)
            host.reset()
