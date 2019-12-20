import ctypes
import os
import os.path as osp
import platform

from game.game_record import Record
from utils.logger_utils import logger

if "Windows" in platform.platform():
    os.environ["path"] = \
        "F:\\opencv-4.1.2\\build\\install;F:\\opencv-4.1.2\\build\\install\\x64\\mingw\\bin;" \
        "F:\\opencv-4.1.2\\build\\install\\x64\\mingw\\lib;C:\\Windows\\system32;C:\\Windows;" \
        "C:\\Windows\\System32\\Wbem;C:\\Windows\\System32\\WindowsPowerShell\\v1.0;" \
        "C:\\Windows\\System32\\OpenSSH;C:\\Program Files (x86)\\NVIDIA Corporation\\PhysX\\Common;" \
        "$HAVA_HOME\\bin;C:\\Program Files\\Git\\cmd;C:\\mingw64\\bin;F:\\adb;C:\\Program Files\\CMake\\bin;" \
        "D:\\Anaconda3;C:\\Users\\orient\\AppData\\Local\\Microsoft\\WindowsApps;C:\\Program Files (x86)\\CMake\\bin"

import sys
import time

import cv2
import numpy as np

from config import CARD_DICT
from game.parse_result import parse_frame_state
from utils.c_lib_utils import Result, STATE_DICT, convert2pymat

"""
仅支持 1920 * 1080 分辨率的屏幕
"""


class ClashRoyalEnv:
    MODE = {"battle": 0,
            "friend_battle_host": 1,
            "friend_battle_guest": 2}

    def __init__(self, root, device, mode=MODE["battle"], name="gamer0"):
        super().__init__()
        w = 1080
        self.running_frame_count = 0
        # 1080 - 54 * 4   每个格子48
        self.offset_w = 39

        self.offset_h = 94

        self.width = 1080 // 2 - self.offset_w * 2
        self.height = 62 * 10

        self.device_id = device.device_id if device is not None else ""
        self.device = device
        self.mode = mode
        self.name = name
        self.root = root
        self.scale = True
        self.real_time = True
        self.game_start = False
        self.game_finish = False
        self.frame_count = 0
        self.log = True
        self.retry = 0
        self.card_choices = [[340, 1720], [560, 1702], [738, 1698], [938, 1718]]
        self.img_shape = (256, 192, 3 * 4)

        self.memory_record = []

        if sys.platform == 'win32':
            lib_path = "F:\\\\PyCharmProjects\\\\deep_clash_royal\\\\lib\\\\libc_opencv.dll"
        else:
            lib_path = "./lib/libc_opencv.so"

        self.lib = ctypes.cdll.LoadLibrary(lib_path)
        self.lib.detect_frame.restype = Result

        self.record = Record(root)
        self.card_location = {}

    def _init_game(self, gameId):
        logger.info("init:" + str(gameId))
        self.game_start = True
        self.game_finish = False
        self.frame_count = 0
        self.skip_step = 0
        self.game_id = gameId
        self.record.init_record(gameId)
        self.lib.init_game(gameId)

        self.imgs = []

        self.rewards = []
        self.actions = []
        # 我方3塔血量 敌方3塔血量 剩余圣水量 双倍圣水 即死
        self.env_states = []
        # 0 ~ 93 可用的card 92种card消耗圣水量
        self.card_types = []
        self.card_properties = []

        self.pre_mine_crown = 0
        self.pre_opp_crown = 0
        self.memory_record.append(gameId)

    def frame_step(self, img, actor_hidden):
        self.frame_count += 1
        result = Result()

        pymat = convert2pymat(img)
        result = self.lib.detect_frame(pymat, result)

        state = self._process_result(result, img)
        if result.frame_state == STATE_DICT["RUNNING_STATE"] and result.frame_index >= 0 and self.game_start:
            img_state, env_state, card_type, card_properties = state
            observation = [result.frame_index, img_state, env_state, card_type, card_properties, actor_hidden]
        else:
            observation = None

        return observation

    def _process_result(self, result, img):
        if result.frame_state == STATE_DICT["ERROR_STATE"]:
            self._action_on_error(result, img)

        elif result.frame_state == STATE_DICT["MENU_STATE"]:
            self._action_on_hall(result)

        elif result.frame_state == STATE_DICT["RUNNING_STATE"]:
            return self._action_on_running(result, img)

        elif result.frame_state == STATE_DICT["FINISH_STATE"]:
            self._action_on_finish(result, img)

        return None

    def _action_on_error(self, result, img):
        if not self.game_start:
            return
        if self.log:
            logger.debug("{:s} error   spent:{:f}".format(self.device_id, result.milli))
        if self.record:
            self.record.record_error_img(self.frame_count, img)

    def _action_on_running(self, result, img):
        self.retry = 0
        if not self.game_start:
            return
        if self.log:
            logger.info(str(self.device_id) + "  running:" + str(result.frame_index) + "  " +
                        str(len(self.env_states)) + "  elixir:" + str(result.remain_elixir)
                        + "  spent:" + str(result.milli))

            logger.info("{:s}:{:f}-{:s}:{:f}-{:s}:{:f}-{:s}:{:f}".format(CARD_DICT[result.card_type[0]],
                                                                         result.prob[0],
                                                                         CARD_DICT[result.card_type[1]],
                                                                         result.prob[1],
                                                                         CARD_DICT[result.card_type[2]],
                                                                         result.prob[2],
                                                                         CARD_DICT[result.card_type[3]],
                                                                         result.prob[3], ))

            logger.info("hp:{:f}-{:f}-{:f}-{:f}-{:f}-{:f}".format(result.opp_hp[0],
                                                                  result.opp_hp[1],
                                                                  result.opp_hp[2],
                                                                  result.mine_hp[0],
                                                                  result.mine_hp[1],
                                                                  result.mine_hp[2], ))
        if result.frame_index < 0:
            return
        self.skip_step = self.skip_step - 1 if self.skip_step > 0 else 0

        reward = 0
        reward += result.opp_hp[0] * 0.2
        reward += result.opp_hp[1] * 0.1
        reward += result.opp_hp[2] * 0.1
        reward -= result.mine_hp[0] * 0.2
        reward -= result.mine_hp[1] * 0.1
        reward -= result.mine_hp[2] * 0.1

        if result.opp_crown > self.pre_opp_crown:
            reward -= 0.6
            self.pre_opp_crown = result.opp_crown
        if result.mine_crown > self.pre_mine_crown:
            reward += 0.6
            self.pre_mine_crown = result.mine_crown

        self._append_reward(reward, result.frame_index)

        env_state, card_type, card_property = parse_frame_state(result)

        card_type = np.array(result.card_type)

        self.card_location = {card_type[0]: self.card_choices[0], card_type[1]: self.card_choices[1],
                              card_type[2]: self.card_choices[2], card_type[3]: self.card_choices[3]}

        self.env_states.append(env_state)
        self.card_types.append(card_type)
        self.card_properties.append(card_property)

        img_state = img[self.offset_h: self.offset_h + self.height, self.offset_w: - self.offset_w, :]
        self.imgs.append(cv2.resize(img_state, (192, 256)))

        if self.record:
            self.record.record_running_img(self.frame_count, self.imgs[result.frame_index])

        return [img_state, env_state, card_type, card_property]

    def _action_on_finish(self, result, img):
        self._finish_game()
        if self.log:
            logger.debug("game in finish:" + str(result.battle_result) + "  spent:" + str(result.milli))
        if not self.game_start:
            return
        self._record_reward(result, img)

    def _action_on_hall(self, result):
        if self.log:
            logger.info(
                "game in hall:" + str(result.index) + " grey:" + str(result.is_grey) + "  spent:" + str(result.milli))
        if self.game_start and self.game_finish:
            self.game_start = False
            self.game_finish = False
        if not self.game_start:
            self._init_game(int(str(int(time.time() * 100))[-9:]))

        if self.mode == self.MODE["battle"] and result.index == 2:
            if self.retry > 25 and self.retry % 10 == 0:
                self.retry = 0
                self.device.tap_button([344, 1246])

        elif self.mode == self.MODE["friend_battle_host"] and result.index == 3:
            if result.is_grey:
                if self.retry > 0 and self.retry % 5 == 0:
                    self.retry = 0
                    # normal 548 544     548 944
                    self.device.tap_button([548, 544])

            else:
                if result.purple_loc[0] != 0:
                    if self.retry > 0 and self.retry % 5 == 0:
                        self.retry = 0
                        self.device.tap_button([result.purple_loc[0], result.purple_loc[1]])

        elif self.mode == self.MODE["friend_battle_guest"] and result.index == 3:
            if not result.is_grey:
                if result.yellow_loc[0] != 0:
                    if self.retry > 0 and self.retry % 5 == 0:
                        self.retry = 0
                        self.device.tap_button([result.yellow_loc[0], result.yellow_loc[1]])

        self.retry += 1

    def _record_reward(self, result, img):
        if self.game_start and not self.game_finish:
            self.game_finish = True
            reward = 0
            if result.battle_result == 1:
                reward = 1 - result.time * 0.001
            elif result.battle_result == -1:
                reward = -1 + result.time * 0.001
            self._update_reward(reward, result.frame_index - 10)

            if self.record:
                self.record.record_finish_img(result.frame_index, img)

                self.record.record_state(self.env_states[:result.frame_index - 9],
                                         self.card_types[:result.frame_index - 9],
                                         self.card_properties[:result.frame_index - 9])

                self.record.record_actions(self.actions[:result.frame_index - 9])
                self.record.record_rewards(self.rewards[:result.frame_index - 9])
                self.record.finish_record(result.battle_result)

    def _finish_game(self):
        if self.retry % 50 == 0:
            self.retry = 0
            self.device.tap_button([536, 1684])

        self.retry += 1

    def _update_reward(self, reward_value, index):
        if self.log and reward_value != 0:
            logger.info("update  step {}  reward {:f} ".format(index, reward_value) +
                        ("++++++++++" if reward_value > 0 else "----------"))
        self.rewards[index] += reward_value

    def _append_reward(self, reward_value, index):
        if self.log and reward_value != 0:
            logger.info("append  step {}  reward {:f} ".format(index, reward_value) +
                        ("++++++++++" if reward_value > 0 else "----------"))
        self.rewards.append(reward_value)

    def step(self, action):
        """
        {"card": (action_card.item(), card_prob[:, action_card.item()].item()),
        "pos_x": (action_pos_x.item(), pos_x_prob[:, action_pos_x.item()].item()),
        "pos_y": (action_pos_y.item(), pos_y_prob[:, action_pos_y.item()].item()),
        "choice_card": choice_card}, actor_hidden
        :param action:
        :return:
        """
        if action[0] != 0:
            if action["use_card"][0] == 0:
                logger.info("do nothing or skip step.")
            else:
                try:
                    card = action["card"][0]
                    loc_x = int(action["pos_x"][0] * self.width * 2) + self.offset_w * 2
                    loc_y = int(action["pos_y"][0] * self.height * 2) + self.offset_h * 2

                    start_x = self.card_location[card]
                    start_y = self.card_location[card]
                    if self.real_time:
                        self.device.swipe([start_x, start_y, loc_x, loc_y])
                except Exception as e:
                    print(e)
        self.actions.append(action)

    def reset(self):
        self.game_start = False


if __name__ == '__main__':
    royal = ClashRoyalEnv("./", "id")
