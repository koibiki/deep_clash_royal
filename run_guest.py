import cv2
import time
from brain.base_brain import BaseBrain
from game.clash_royal import ClashRoyal
from utils.c_lib_utils import convert2pymat

if __name__ == '__main__':

    i = 0

    # root = "/home/chengli/data/gym_data/clash_royal"
    root = "/home/holaverse/work/07battle_filed/clash_royal"

    guest = ClashRoyal(root, device_id="70a7da50", mode=ClashRoyal.MODE["friend_battle_guest"], name="guest")

    base_brain = BaseBrain(guest, BaseBrain.BrainType["runner"], )

    guest_address = "http://127.0.0.1:35013/device/70a7da50/video.flv"
    # address = "./scan/s1.mp4"
    # address = "http://127.0.0.1:46539/device/cd9faa7f/video.flv"
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
