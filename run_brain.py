import cv2

from brain.base_brain import BaseBrain
from game.clash_royal import ClashRoyal
from utils.c_lib_utils import convert2pymat

if __name__ == '__main__':

    # address = "http://127.0.0.1:35013/device/cd9faa7f/video.flv"
    address = "http://127.0.0.1:35013/device/70a7da50/video.flv"
    # address = "./scan/s1.mp4"
    # address = "http://127.0.0.1:46539/device/cd9faa7f/video.flv"
    capture = cv2.VideoCapture(address)

    i = 0

    # root = "/home/chengli/data/gym_data/clash_royal"
    root = "/home/holaverse/work/07battle_filed/clash_royal"

    clash_royal = ClashRoyal(root)

    base_brain = BaseBrain(clash_royal.n_card_actions,
                           clash_royal.img_shape,
                           clash_royal.state_shape)

    base_brain.load_memory(root)
    learn = False

    while True:
        i += 1
        state, img = capture.read()

        if state:
            if learn:
                if i % 100 != 0:
                    learn = False
                    continue
            else:
                if i % 20 != 0:
                    continue
            img = cv2.resize(img, (540, 960))
            # cv2.imshow('image', img)
            # cv2.waitKey(0)

            pymat = convert2pymat(img)

            observation = clash_royal.frame_step(img)
            if observation is not None:
                action = base_brain.choose_action(observation)
                clash_royal.step(observation, action)

            if clash_royal.game_start and clash_royal.game_finish and clash_royal.retry <= 1:
                base_brain.store_transition(clash_royal.episode_record)
                base_brain.update_episode_result(clash_royal.get_rate_of_winning())

            if clash_royal.game_start and clash_royal.game_finish and clash_royal.retry <= 1:
                learn = True
                for _ in range(25):
                    base_brain.learn()

        else:
            print("没有信号..")
            # capture = cv2.VideoCapture(address)
            # cv2.destroyAllWindows()
