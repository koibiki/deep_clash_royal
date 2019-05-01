import ctypes
import os
import os.path as osp
import time
from multiprocessing.pool import Pool
import numpy as np
import random
import cv2

from game.parse_result import parse_running_state, card_dict
from utils.c_lib_utils import Result, STATE_DICT, convert2pymat
from utils.cmd_utils import execute_cmd

"""
仅支持 1920 * 1080 分辨率的屏幕
"""


class ClashRoyal:

    def __init__(self, root):
        super().__init__()
        w = 1080
        num_align_width = 7
        num_align_height = 8
        self.w_gap = self.h_gap = w_gap = h_gap = w // num_align_width
        offset_w = w_gap // 2
        ll = ctypes.cdll.LoadLibrary
        self.root = root
        self.lib = ll("./lib/libc_opencv.so")
        self.lib.detect_frame.restype = Result
        self.record = True
        self.real_time = True
        self.game_start = False
        self.game_finish = False
        self.frame_count = 0
        self.log = True
        self.p = Pool(4)
        self.retry = 0
        self.loc_x_action_choices = [offset_w + x * w_gap + w_gap // 2 for x in range(num_align_width)]
        self.loc_y_action_choices = [(y + 1) * h_gap + h_gap * 3 // 4 for y in range(num_align_height)]
        self.card_choices = [[340, 1720], [560, 1702], [738, 1698], [938, 1718]]
        self.n_loc_x_actions = len(self.loc_x_action_choices)
        self.n_loc_y_actions = len(self.loc_y_action_choices)
        self.n_card_actions = len(self.card_choices)
        self.img_shape = (256, 192, 3)
        # 4个位置卡牌的 种类 是否可用 消耗圣水量 剩余圣水量
        self.state_shape = len(card_dict.keys()) * 4 + 4 + 4 + 1

    def _init_game(self, gameId):
        self.game_start = True
        self.game_finish = False
        self.frame_count = 0
        self.running_frame_count = 0
        self.game_id = gameId
        self.lib.init_game(gameId)
        self.error_dir = osp.join(self.root, "{:d}/error".format(gameId))
        self.running_dir = osp.join(self.root, "{:d}/running".format(gameId))
        self.finish_dir = osp.join(self.root, "{:d}/finish".format(gameId))
        self.rewards = np.zeros(2000, dtype=np.object)
        self.actions = np.zeros(2000, dtype=np.object)
        self.imgs = np.zeros(2000, dtype=np.object)
        self.states = np.zeros(2000, dtype=np.object)
        self.pre_mine_crown = 0
        self.pre_opp_crown = 0
        os.makedirs(self.error_dir)
        os.makedirs(self.running_dir)
        os.makedirs(self.finish_dir)

    def frame_step(self, img):
        self.frame_count += 1
        result = Result()

        pymat = convert2pymat(img)
        result = self.lib.detect_frame(pymat, result)

        self._process_result(result, img)
        if self.game_start and result.frame_state == STATE_DICT["RUNNING_STATE"]:
            print("game in running")
            observation = [result.frame_index, self.imgs[result.frame_index], self.states[result.frame_index]]
        else:
            observation = None

        return observation

    def _process_result(self, result, img):
        state = None
        if result.frame_state == STATE_DICT["ERROR_STATE"]:
            self._action_on_error(result, img)

        elif result.frame_state == STATE_DICT["MENU_STATE"]:
            self._action_on_hall(result)

        elif result.frame_state == STATE_DICT["RUNNING_STATE"]:
            state = self._action_on_running(result, img)

        elif result.frame_state == STATE_DICT["FINISH_STATE"]:
            self._action_on_finish(result, img)

        return state

    def _action_on_error(self, result, img):
        if not self.game_start:
            return
        if self.log:
            print("error   spent:" + str(result.milli))
        if self.record:
            cv2.imwrite(osp.join(self.error_dir, "{:d}.jpg".format(self.frame_count)), img)

    def _action_on_running(self, result, img):
        if not self.game_start:
            return
        if self.log:
            print("id:" + str(self.game_id) + "  running:" + str(result.frame_index) + "  " + str(
                self.running_frame_count) + "  spent:" + str(result.milli))
        self.running_frame_count += 1
        reward = 0
        if result.opp_crown > self.pre_opp_crown:
            reward = -0.1
            self.pre_opp_crown = result.opp_crown
        if result.mine_crown > self.pre_mine_crown:
            reward = 0.1
            self.pre_mine_crown = result.mine_crown
        if self.record:
            cv2.imwrite(osp.join(self.running_dir, "{:d}.jpg".format(result.frame_index)), img)

        if reward != 0:
            self._update_reward(reward, result.frame_index - 50, 50)
        else:
            self._update_reward(reward, result.frame_index, 1)

        state = parse_running_state(result)

        self.states[result.frame_index] = state
        img_state = \
            img[self.h_gap + self.h_gap // 4: 9 * self.h_gap + self.h_gap // 4, self.h_gap // 2:-self.h_gap // 2, :]
        self.imgs[result.frame_index] = cv2.resize(img_state, (192, 256)) / 255.

        return state

    def _action_on_finish(self, result, img):
        self._finish_game()
        if self.log:
            print("game in finish:" + str(result.win) + "  spent:" + str(result.milli))
        if not self.game_start:
            return
        self._record_reward(result, img)

    def _action_on_hall(self, result):
        if self.log:
            print("game in hall:" + str(result.index) + "  spent:" + str(result.milli))
        if self.game_start and self.game_finish:
            self.game_start = False
            self.game_finish = False
        if not self.game_start:
            self._init_game(int(time.time() * 1000))
        if self.real_time and self.retry % 50 == 0:
            self.retry = 0
            cmd = "adb shell input tap 344 1246"
            self.p.apply_async(execute_cmd, args={cmd})
        self.retry += 1

    def _record_reward(self, result, img):
        if self.game_start and not self.game_finish:
            self.game_finish = True
            reward = 1 - result.frame_index * 0.0001 if result.win else (-1 + result.frame_index * 0.0001)
            self._update_reward(reward, result.frame_index - 60, 50)

            if self.record:
                cv2.imwrite(osp.join(self.finish_dir, "{:d}.jpg".format(result.frame_index)), img)
                with open(osp.join(self.root, str(self.game_id) + "/state.txt"), "w") as f:
                    for i in range(len(self.states[:self.running_frame_count])):
                        f.write(str(i) + ":" + str(self.states[i]))
                        f.write("\n")
                with open(osp.join(self.root, str(self.game_id) + "/action.txt"), "w") as f:
                    for i in range(len(self.actions[:self.running_frame_count])):
                        f.write(str(i) + ":" + str(self.actions[i]))
                        f.write("\n")
                with open(osp.join(self.root, str(self.game_id) + "/reward.txt"), "w") as f:
                    for i in range(len(self.rewards[:self.running_frame_count])):
                        f.write(str(i) + ":" + str(self.rewards[i]))
                        f.write("\n")

    def _finish_game(self):
        if self.real_time and self.retry % 50 == 0:
            self.retry = 0
            cmd = "adb shell input tap 536 1684"
            self.p.apply_async(execute_cmd, args={cmd})
        self.retry += 1

    def _update_reward(self, reward_value, start_step, update_steps):
        if self.log:
            print("update  step {}  reward {:f} ".format(start_step + update_steps, reward_value) + (
                "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
                if reward_value > 0 else "-------------------------------------------------------------------------"))
        for i in range(update_steps):
            self.rewards[start_step + i] += reward_value * (i + 1) / update_steps

    def step(self, index, action):
        if action[0] != 0:
            card = self.card_choices[action[0]]
            loc_x = self.loc_y_action_choices[action[1]]
            loc_y = self.loc_y_action_choices[action[2]]
            cmd = "adb shell input swipe {:d} {:d} {:d} {:d} 300".format(card[0], card[1], loc_x, loc_y)
            self.p.apply_async(execute_cmd, args={cmd})
            self.actions[index] = action
        else:
            self.actions[index] = [0, 0, 0]

    def episode_statistics(self):
        episode_record = []
        skip_step = 10
        max_step = self.running_frame_count - skip_step
        for i in range(self.running_frame_count - skip_step):
            next_step = (i + skip_step) if (i + skip_step) < max_step else max_step - 1
            next_img = self.imgs[next_step]
            next_state = self.states[next_step]
            episode_record.append([self.imgs[i], self.states[i], self.actions[i], self.rewards[i], next_img, next_state])
        return episode_record
