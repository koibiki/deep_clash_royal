import ctypes
import os
import os.path as osp

import cv2

from utils.c_lib_utils import Result, STATE_DICT, convert2pymat


class ClashRoyal:

    def __init__(self, root):
        super().__init__()
        ll = ctypes.cdll.LoadLibrary
        self.root = root
        self.lib = ll("./lib/libc_opencv.so")
        self.lib.detect_frame.restype = Result
        self.record = False

    def init_game(self, gameId):
        self.game_id = gameId
        self.lib.init_game(gameId)
        self.error_dir = osp.join(self.root, "{:d}/error".format(gameId))
        self.running_dir = osp.join(self.root, "{:d}/running".format(gameId))
        self.finish_dir = osp.join(self.root, "{:d}/finish".format(gameId))
        self.rewards = []
        self.steps = []
        self.pre_mine_crown = 0
        self.pre_opp_crown = 0
        os.makedirs(self.error_dir)
        os.makedirs(self.running_dir)
        os.makedirs(self.finish_dir)

    def frame_step(self, img):
        result = Result()

        pymat = convert2pymat(img)
        result = self.lib.detect_frame(pymat, result)

        self.process_result(result, img)
        return result

    def process_result(self, result, img):
        if result.frame_state == STATE_DICT["ERROR_STATE"]:
            print("error   spent:" + str(result.milli))
            if self.record:
                cv2.imwrite(osp.join(self.error_dir, "/{:d}.jpg".format(result.frame_index)), img)

        elif result.frame_state == STATE_DICT["MENU_STATE"]:
            print("id:" + str(self.game_id) + "  in hall:" + str(result.index) + "  spent:" + str(result.milli))

        elif result.frame_state == STATE_DICT["RUNNING_STATE"]:
            print("id:" + str(self.game_id) + "  running:" + str(result.frame_index) + "  spent:" + str(result.milli))
            if result.opp_crown > self.pre_opp_crown:
                if result.opp_crown == 3:
                    self._update_reward(-100, result.frame_index)
                else:
                    self._update_reward(-10, result.frame_index)

                self.pre_opp_crown = result.opp_crown
            if result.mine_crown > self.pre_mine_crown:
                if result.mine_crown == 3:
                    self._update_reward(100, result.frame_index)
                else:
                    self._update_reward(10, result.frame_index)
                self.pre_mine_crown = result.mine_crown
            if self.record:
                cv2.imwrite(osp.join(self.running_dir, "/{:d}.jpg".format(result.frame_index)), img)

        elif result.frame_state == STATE_DICT["FINISH_STATE"]:
            print("id:" + str(self.game_id) + "  is_finish:" + str(result.win) + "  spent:" + str(result.milli))

            if result.win:
                self._update_reward(100, result.frame_index)
            else:
                self._update_reward(-100, result.frame_index)

            if self.record:
                cv2.imwrite(osp.join(self.finish_dir, "/{:d}.jpg".format(0)), img)

    def _update_reward(self, reward_value, step):
        print("update  step {}  reward {:d} ".format(step, reward_value) + (
            "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
            if reward_value > 0 else "-------------------------------------------------------------------------"))
        self.rewards.append(reward_value)
        self.steps.append(step)
