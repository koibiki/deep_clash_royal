import time

import cv2

from brain.dqn import DQN
from game.clash_royal import ClashRoyal

if __name__ == '__main__':

    i = 0

    root = "/home/chengli/data/gym_data/clash_royal"
    # root = "/home/holaverse/work/07battle_filed/clash_royal"
    device_id = "70a7da50"
    # device_id = "cd9faa7f"

    guest = ClashRoyal(root, device_id=device_id, mode=ClashRoyal.MODE["friend_battle_guest"], name="guest")

    base_brain = DQN(guest, DQN.BrainType["runner"], )

    # guest_address = "http://127.0.0.1:35013/device/" + device_id + "/video.flv"
    guest_address = "http://127.0.0.1:46539/device/" + device_id + "/video.flv"

    guest_capture = cv2.VideoCapture(guest_address)
    while True:
        i += 1
        guest_state, guest_img = guest_capture.read()

        if guest_state:

            if i % 20 != 0:
                continue
            guest_img = cv2.resize(guest_img, (540, 960))

            guest_observation = guest.frame_step(guest_img)
            if guest_observation is not None:
                guest_action = base_brain.choose_action(guest_observation)
                guest.step(guest_observation, guest_action)

            if guest.game_start and guest.game_finish and guest.retry <= 1:
                base_brain.update_episode_result(guest.get_rate_of_winning())
                base_brain.record_battle_result()
                base_brain.load_model()

        else:
            print("没有信号..")
            guest_capture.release()
            time.sleep(30)
            guest_capture = cv2.VideoCapture(guest_address)
            guest.reset()
