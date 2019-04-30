import ctypes
import os
import os.path as osp
import time
from multiprocessing.pool import Pool
import numpy as np
import random
import cv2

from utils.c_lib_utils import Result, STATE_DICT, convert2pymat


class ClashRoyal:

    def __init__(self, root):
        super().__init__()
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
        self.rewards = {}
        self.actions = {}
        self.p = Pool(4)
        self.retry = 0

    def _init_game(self, gameId):
        self.game_start = True
        self.game_finish = False
        self.frame_count = 0
        self.game_id = gameId
        self.lib.init_game(gameId)
        self.error_dir = osp.join(self.root, "{:d}/error".format(gameId))
        self.running_dir = osp.join(self.root, "{:d}/running".format(gameId))
        self.finish_dir = osp.join(self.root, "{:d}/finish".format(gameId))
        self.rewards.clear()
        self.actions.clear()
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
        return result

    def _process_result(self, result, img):
        if result.frame_state == STATE_DICT["ERROR_STATE"]:
            self._action_on_error(result, img)

        elif result.frame_state == STATE_DICT["MENU_STATE"]:
            self._action_on_hall(result)

        elif result.frame_state == STATE_DICT["RUNNING_STATE"]:
            self._action_on_running(result, img)

        elif result.frame_state == STATE_DICT["FINISH_STATE"]:
            self._action_on_finish(result, img)

    def _action_on_error(self, result, img):
        if self.log:
            print("error   spent:" + str(result.milli))
        if not self.game_start:
            return
        if self.record:
            cv2.imwrite(osp.join(self.error_dir, "/{:d}.jpg".format(self.frame_count)), img)

    def _action_on_running(self, result, img):
        if self.log:
            print("id:" + str(self.game_id) + "  running:" + str(result.frame_index) + "  spent:" + str(result.milli))
        if not self.game_start:
            return
        reward = 1
        if result.opp_crown > self.pre_opp_crown:
            reward = -1000 if result.opp_crown == 3 else -100
            self.pre_opp_crown = result.opp_crown
        if result.mine_crown > self.pre_mine_crown:
            reward = 1000 if result.mine_crown == 3 else 100
            self.pre_mine_crown = result.mine_crown
        if self.record:
            cv2.imwrite(osp.join(self.running_dir, "/{:d}.jpg".format(result.frame_index)), img)
        self._update_reward(reward, result.frame_index)
        if random.uniform(0, 1) > 0.9:
            randint = random.randint(0, 3)
            if randint == 0:
                cmd = "adb shell input swipe 340 1720 260 932 300"
            elif randint == 0:
                cmd = "adb shell input swipe 560 1702 678 1058 300"
            elif randint == 0:
                cmd = "adb shell input swipe 738 1698 812 1046 300"
            else:
                cmd = "adb shell input swipe 938 1718 254 902 300"
            self.p.apply_async(self._execute_cmd, args={cmd})

    def _action_on_finish(self, result, img):
        self._finish_game()
        if self.log:
            print("game in finish:" + str(result.win) + "  spent:" + str(result.milli))
        if not self.game_start:
            return
        self._record_reward(result, img)

    def _record_reward(self, result, img):
        if self.game_start and not self.game_finish:
            self.game_finish = True
            reward = 1000 if result.win else -1000
            self._update_reward(reward, result.frame_index - 1)

            if self.record:
                cv2.imwrite(osp.join(self.finish_dir, "/{:d}.jpg".format(self.frame_count)), img)
                # with open(osp.join(self.root, str(self.game_id) + "/action.txt")) as f:
                #     for i in range(len(self.actions.keys())):
                #         pass
                with open(osp.join(self.root, str(self.game_id) + "/reward.txt"), "w") as f:
                    for i in range(len(self.rewards.keys())):
                        f.write(str(i) + ":" + str(self.rewards[i]))
                        f.write("\n")

    def _action_on_hall(self, result):
        if self.log:
            print("game in hall:" + str(result.index) + "  spent:" + str(result.milli))
        if not self.game_start:
            self._init_game(int(time.time() * 1000))
        if self.real_time and self.retry % 50 == 0:
            self.retry = 0
            cmd = "adb shell input tap 344 1246"
            self.p.apply_async(self._execute_cmd, args={cmd})
        self.retry += 1

    def _finish_game(self):
        if self.real_time and self.retry % 50 == 0:
            self.retry = 0
            cmd = "adb shell input tap 536 1684"
            self.p.apply_async(self._execute_cmd, args={cmd})
        self.retry += 1

    def _update_reward(self, reward_value, step):
        if self.log:
            print("update  step {}  reward {:d} ".format(step, reward_value) + (
                "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
                if reward_value > 0 else "-------------------------------------------------------------------------"))
        self.rewards[step] = reward_value

    @staticmethod
    def _execute_cmd(cmd):
        print("execute cmd:{:s}".format(cmd))
        os.system(cmd)
