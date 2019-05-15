import time

import cv2

from brain.ddpg2 import DDPG
from device.mobile import Mobile
from game.clash_royal import ClashRoyal

if __name__ == '__main__':

    i = 0

    # root = "/home/chengli/data/gym_data/clash_royal"
    root = "/home/holaverse/work/07battle_filed/clash_royal"

    # device_id = "70a7da50"
    device_id = "cd9faa7f"
    # device_id = "YP99IN9PE6V4GES8"

    host = ClashRoyal(root, Mobile(device_id), mode=ClashRoyal.MODE["battle"], name="host")

    brain = DDPG(host.img_shape, host.state_shape, DDPG.BrainType["runner"], "battle")

    host_address = "http://127.0.0.1:35013/device/" + device_id + "/video.flv"
    # host_address = "http://127.0.0.1:46539/device/" + device_id + "/video.flv"

    host_capture = cv2.VideoCapture(host_address)

    while True:
        i += 1
        host_state, img = host_capture.read()

        if host_state:

            if i % 20 != 0:
                continue
            h, w, c = img.shape
            print("h:" + str(h) + " w:" + str(w))
            # if h > 1920 and w == 1080:
            #     img = img[:1920, :, :]
            # if h > 1920 and w == 1080:
            #     img = img[:1920, :, :]

            img = cv2.resize(img, (540, 960))

            # cv2.imshow('image', img)
            # cv2.waitKey(0)

            host_observation = host.frame_step(img)
            if host_observation is not None:
                host_action = brain.choose_action(host_observation)
                host.step(host_observation, host_action)

            if host.game_start and host.game_finish and host.retry <= 1:
                brain.update_episode_result(host.get_rate_of_winning())
                brain.record_battle_result()
                brain.load_model()

        else:
            print("没有信号..")
            host_capture.release()
            time.sleep(30)
            host_capture = cv2.VideoCapture(host_address)
            host.reset()
