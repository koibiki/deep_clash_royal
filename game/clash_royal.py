import ctypes
import os
import os.path as osp
import time
from multiprocessing.pool import Pool

import cv2
import numpy as np

from config import CARD_DICT
from game.parse_result import parse_frame_state
from utils.c_lib_utils import Result, STATE_DICT, convert2pymat
from utils.cmd_utils import execute_cmd

"""
仅支持 1920 * 1080 分辨率的屏幕
"""


class ClashRoyal:
    MODE = {"battle": 0,
            "friend_battle_host": 1,
            "friend_battle_guest": 2}

    def __init__(self, root, device_id, mode=MODE["battle"], name="gamer0"):
        super().__init__()
        w = 1080

        # 1080 - 54 * 4   每个格子48


        num_align_width = 7
        num_align_height = 8
        self.w_gap = self.h_gap = w_gap = h_gap = w // num_align_width
        offset_w = w_gap // 2
        ll = ctypes.cdll.LoadLibrary
        self.device_id = device_id
        self.mode = mode
        self.name = name
        self.root = root
        self.record = True
        self.scale = True
        self.real_time = True
        self.game_start = False
        self.game_finish = False
        self.frame_count = 0
        self.log = True
        self.p = Pool(6)
        self.retry = 0
        self.loc_x_action_choices = [offset_w + x * w_gap + w_gap // 2 for x in range(num_align_width - 1)]
        self.loc_y_action_choices = [(y + 1) * h_gap + h_gap * 3 // 4 for y in range(num_align_height)]
        self.card_choices = [[340, 1720], [560, 1702], [738, 1698], [938, 1718]]
        self.n_card_actions = 92 * len(self.loc_x_action_choices) * len(self.loc_y_action_choices)  # 92 种牌 x y
        self.img_shape = (256, 192, 3 * 4)
        # 是否有92种card 92种card是否可用 92种card消耗圣水量 剩余圣水量 耗时 双倍圣水 即死 (我方血量 对方血量)
        self.state_shape = 93 * 4 + 4 + 4 + 1 + 1 + 1 + 1 + 2

        self.memory_record = []
        self.rate_of_winning = []
        self.reward_mean = []
        self.lib = ll("./lib/libc_opencv.so")
        self.lib.detect_frame.restype = Result

    def _init_game(self, gameId):
        self.game_start = True
        self.game_finish = False
        self.frame_count = 0
        self.skip_step = 0
        self.running_frame_count = 0
        self.game_id = gameId
        self.lib.init_game(gameId)
        self.error_dir = osp.join(self.root, "{:d}/error".format(gameId))
        self.running_dir = osp.join(self.root, "{:d}/running".format(gameId))
        self.finish_dir = osp.join(self.root, "{:d}/finish".format(gameId))
        self.rewards = np.zeros(1500, dtype=np.object)
        self.actions = np.zeros(1500, dtype=np.object)
        self.imgs = np.zeros(1500, dtype=np.object)
        self.states = np.zeros(1500, dtype=np.object)
        self.pre_mine_crown = 0
        self.pre_opp_crown = 0
        self.memory_record.append(gameId)
        os.makedirs(self.error_dir)
        os.makedirs(self.running_dir)
        os.makedirs(self.finish_dir)

    def frame_step(self, img):
        self.frame_count += 1
        result = Result()

        pymat = convert2pymat(img)
        result = self.lib.detect_frame(pymat, result)

        self._process_result(result, img)
        if self.game_start and result.frame_state == STATE_DICT["RUNNING_STATE"] and result.frame_index >= 0:
            img_range = []
            for ii in range(4):
                index = 0 if result.frame_index - ii * 5 < 0 else result.frame_index - ii * 5
                img_range.append(self.imgs[index] / 255.)

            step_imgs = np.concatenate(img_range, axis=-1)
            state = self.states[result.frame_index]
            observation = [result.frame_index, step_imgs, state, np.array(result.card_type), np.array(result.available)]
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
            print("{:s} error   spent:{:f}".format(self.device_id, result.milli))
        if self.record:
            cv2.imwrite(osp.join(self.error_dir, "{:d}.jpg".format(self.frame_count)), img)

    def _action_on_running(self, result, img):
        self.retry = 0
        if not self.game_start:
            return
        if self.log:
            print(str(self.device_id) + "  running:" + str(result.frame_index) + "  " + str(
                self.running_frame_count) + "  elixir:" + str(result.remain_elixir) + "  spent:" + str(result.milli))

            print("{:s}:{:f}-{:s}:{:f}-{:s}:{:f}-{:s}:{:f}".format(CARD_DICT[result.card_type[0]],
                                                                   result.prob[0],
                                                                   CARD_DICT[result.card_type[1]],
                                                                   result.prob[1],
                                                                   CARD_DICT[result.card_type[2]],
                                                                   result.prob[2],
                                                                   CARD_DICT[result.card_type[3]],
                                                                   result.prob[3], ))
        if result.frame_index < 0:
            return
        self.running_frame_count += 1
        self.skip_step = self.skip_step - 1 if self.skip_step > 0 else 0
        reward = 0
        if result.opp_crown > self.pre_opp_crown:
            reward = -0.5
            self.pre_opp_crown = result.opp_crown
        if result.mine_crown > self.pre_mine_crown:
            reward = 0.3
            self.pre_mine_crown = result.mine_crown

        if reward != 0:
            self._update_reward(reward, result.frame_index - 10, 5)
        else:
            self._update_reward(reward, result.frame_index, 1)

        state = parse_frame_state(result)

        self.states[result.frame_index] = state
        img_state = img[self.h_gap // 2 + self.h_gap // 8: 9 * self.h_gap // 2 + self.h_gap // 8,
                    self.h_gap // 4:-self.h_gap // 4, :]
        self.imgs[result.frame_index] = cv2.resize(img_state, (192, 256))

        if self.record:
            img_path = osp.join(self.running_dir, "{:d}.jpg".format(result.frame_index))
            if self.scale:
                cv2.imwrite(img_path, self.imgs[result.frame_index])
            else:
                cv2.imwrite(img_path, img)
        return state

    def _action_on_finish(self, result, img):
        self._finish_game()
        if self.log:
            print("game in finish:" + str(result.battle_result) + "  spent:" + str(result.milli))
        if not self.game_start:
            return
        self._record_reward(result, img)

    def _action_on_hall(self, result):
        if self.log:
            print(
                "game in hall:" + str(result.index) + " grey:" + str(result.is_grey) + "  spent:" + str(result.milli))
        if self.game_start and self.game_finish:
            self.game_start = False
            self.game_finish = False
        if not self.game_start:
            self._init_game(int(time.time() * 1000))

        if self.mode == self.MODE["battle"] and result.index == 2:
            if self.retry > 25 and self.retry % 10 == 0:
                self.retry = 0
                cmd = "adb -s {:s} shell input tap 344 1246".format(self.device_id)
                self.p.apply_async(execute_cmd, args={cmd})

        elif self.mode == self.MODE["friend_battle_host"] and result.index == 3:
            if result.is_grey:
                if self.retry > 0 and self.retry % 5 == 0:
                    self.retry = 0
                    # normal 548 544     548 944
                    cmd = "adb  -s {:s} shell input tap 548 944".format(self.device_id)
                    self.p.apply_async(execute_cmd, args={cmd})

            else:
                if result.purple_loc[0] != 0:
                    if self.retry > 0 and self.retry % 5 == 0:
                        self.retry = 0
                        cmd = "adb -s {:s} shell input tap {:d} {:d}".format(self.device_id,
                                                                             result.purple_loc[0],
                                                                             result.purple_loc[1])
                        self.p.apply_async(execute_cmd, args={cmd})

        elif self.mode == self.MODE["friend_battle_guest"] and result.index == 3:
            if not result.is_grey:
                if result.yellow_loc[0] != 0:
                    if self.retry > 0 and self.retry % 5 == 0:
                        self.retry = 0
                        cmd = "adb -s {:s} shell input tap {:d} {:d}".format(self.device_id,
                                                                             result.yellow_loc[0],
                                                                             result.yellow_loc[1])
                        self.p.apply_async(execute_cmd, args={cmd})
        self.retry += 1

    def _record_reward(self, result, img):
        if self.game_start and not self.game_finish:
            self.game_finish = True
            reward = 0
            if result.battle_result == 1:
                reward = 1 - result.frame_index * 0.0001
            elif result.battle_result == -1:
                reward = -1 + result.frame_index * 0.0001
            self._update_reward(reward, result.frame_index - 20, 10)

            if self.record:

                cv2.imwrite(osp.join(self.finish_dir, "{:d}.jpg".format(result.frame_index)), img)
                with open(osp.join(self.root, str(self.game_id) + "/state.txt"), "w") as f:
                    for i in range(len(self.states[:self.running_frame_count - 10])):
                        state_str = ""
                        for item in self.states[i]:
                            state_str += str(item) + ","
                        state_str = state_str[:-1]
                        f.write(str(i) + ":" + state_str)
                        f.write("\n")
                with open(osp.join(self.root, str(self.game_id) + "/action.txt"), "w") as f:
                    for i in range(len(self.actions[:self.running_frame_count - 10])):
                        f.write(str(i) + ":" + str(self.actions[i]))
                        f.write("\n")
                with open(osp.join(self.root, str(self.game_id) + "/reward.txt"), "w") as f:
                    for i in range(len(self.rewards[:self.running_frame_count - 10])):
                        f.write(str(i) + ":" + str(self.rewards[i]))
                        f.write("\n")

                old_path = osp.join(self.root, str(self.game_id))
                result_path = "fail"
                if result.battle_result == 1:
                    result_path = "win"
                elif result.battle_result == 0:
                    result_path = "draw"

                new_path = osp.join(self.root, result_path + "/" + str(self.game_id))
                os.rename(old_path, new_path)

            if len(self.rate_of_winning) > 20:
                self.rate_of_winning.pop(0)
            self.rate_of_winning.append(1 if result.battle_result == 1 else 0)

            if len(self.reward_mean) > 50:
                self.reward_mean.pop(0)

            self.reward_mean.append(np.sum(self.rewards[:self.running_frame_count - 10]))

            self.episode_record = self._episode_statistics(result)

    def _finish_game(self):
        if self.retry % 50 == 0:
            self.retry = 0
            cmd = "adb -s {:s} shell input tap 536 1684".format(self.device_id)
            self.p.apply_async(execute_cmd, args={cmd})
        self.retry += 1

    def _update_reward(self, reward_value, start_step, update_steps):
        if self.log and reward_value != 0:
            print("update  step {}  reward {:f} ".format(start_step + update_steps, reward_value) + (
                "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
                if reward_value > 0 else "-------------------------------------------------------------------------"))
        for i in range(update_steps):
            self.rewards[start_step + i] += reward_value * (i + 1) / update_steps

    def step(self, observation, action):
        index = observation[0]
        if self.skip_step == 0 and action[0] != 0:
            # 更新 选择了不可用 card 的 reward
            if action[0] in observation[3]:
                card_index = np.argmax([action[0] == item for item in observation[3]])
                if observation[4][card_index] == 0:
                    self._update_reward(-0.1, index, 1)
                else:
                    self.skip_step = 5
                    self._update_reward(0.01, index, 1)
                    card = self.card_choices[card_index]
                    loc_x = self.loc_y_action_choices[action[1]]
                    loc_y = self.loc_y_action_choices[action[2]]
                    if self.real_time:
                        cmd = "adb -s {:s} shell input swipe {:d} {:d} {:d} {:d} 300".format(self.device_id,
                                                                                             card[0],
                                                                                             card[1],
                                                                                             loc_x,
                                                                                             loc_y)
                        self.p.apply_async(execute_cmd, args={cmd})
            else:
                self._update_reward(-0.01, index, 1)
            self.actions[index] = action
        else:
            print("do nothing or skip step.")
            self.actions[index] = [0, 0, 0]

    def _episode_statistics(self, result):
        skip_step = 10
        max_step = self.running_frame_count - skip_step

        img_paths = []
        type_dir = "fail"
        if result.battle_result == 1:
            type_dir = "win"
        elif result.battle_result == 0:
            type_dir = "draw"

        save_dir = osp.join(self.root, type_dir + "/" + str(self.game_id)) + "/running"
        for index in range(max_step):
            indices = [index + 5, index, index - 5, index - 5 * 2, index - 5 * 3]
            indices = [max(0, item) for item in indices]
            indices = [min(item, max_step - 1) for item in indices]
            img_path = [osp.join(save_dir, str(i) + ".jpg") for i in indices]
            img_paths.append(img_path)
        episode_record = [self.game_id, img_paths, self.states[:max_step], self.actions[:max_step],
                          self.rewards[:max_step]]
        return episode_record

    def get_rate_of_winning(self):
        return (np.sum(self.rate_of_winning) / (0.0001 + len(self.rate_of_winning)), \
                np.sum(self.reward_mean) / (0.0001 + len(self.reward_mean)))

    def reset(self):
        self.game_start = False


if __name__ == '__main__':
    royal = ClashRoyal("./", "id")
