import json

from utils.logger_utils import logger
import os.path as osp
import os
import cv2


class Record(object):

	def __init__(self, root):
		self.game_id = -1
		self.game_episodes = 0
		self.win_episodes = 0
		self.fail_episodes = 0
		self.draw_episodes = 0
		self.root = root
		self.scale = True
		self.reward_amount = 0

	def init_record(self, game_id):
		self.error_dir = osp.join(self.root, "{:d}/error".format(game_id))
		self.running_dir = osp.join(self.root, "{:d}/running".format(game_id))
		self.finish_dir = osp.join(self.root, "{:d}/finish".format(game_id))
		self.save_dir = osp.join(self.root, "{:d}".format(game_id))
		os.makedirs(self.error_dir)
		os.makedirs(self.running_dir)
		os.makedirs(self.finish_dir)
		self.game_id = game_id

	def reset(self):
		self.game_episodes = 0
		self.win_episodes = 0
		self.fail_episodes = 0
		self.draw_episodes = 0

	def check_game_valid(self):
		if self.game_id == -1:
			raise Exception("game id == -1.")

	def get_win_rate(self):
		return self.win_episodes / self.game_episodes

	def get_mean_reward(self):
		return self.reward_amount / self.game_episodes

	def record_episode_result(self, result):
		if result == -1:
			self.fail_episodes += 1
		elif result == 0:
			self.draw_episodes += 1
		elif result == 1:
			self.win_episodes += 1
		else:
			logger.error("错误的游戏结果.")
		self.game_episodes += 1

	def record_error_img(self, index, img):
		self.check_game_valid()
		cv2.imwrite(osp.join(self.error_dir, "{:d}.jpg".format(index)), img)

	def record_running_img(self, index, img):
		self.check_game_valid()
		cv2.imwrite(osp.join(self.running_dir, "{:d}.jpg".format(index)), img)

	def record_finish_img(self, index, img):
		self.check_game_valid()
		cv2.imwrite(osp.join(self.finish_dir, "{:d}.jpg".format(index)), img)

	def record_state(self, env_state, card_type, card_properties):
		self.check_game_valid()
		self._record_json(osp.join(self.save_dir, "env_state.json"), env_state)
		self._record_json(osp.join(self.save_dir, "card_type.json"), card_type)
		self._record_json(osp.join(self.save_dir, "card_properties.json"), card_properties)

	def record_actions(self, actions):
		self.check_game_valid()
		self._record_json(osp.join(self.save_dir, "actions.json"), actions)

	def record_rewards(self, rewards):
		self.check_game_valid()
		self.reward_amount += sum(rewards)
		self._record_json(osp.join(self.save_dir, "rewards.json"), rewards)

	def _record_json(self, path, data):
		with open(path, "w") as f:
			json.dump(data, f)

	def finish_record(self, result):
		self.check_game_valid()
		old_path = osp.join(self.root, str(self.game_id))
		result_path = "fail"
		if result == 1:
			result_path = "win"
		elif result == 0:
			result_path = "draw"

		os.makedirs(os.path.join(self.root, result_path), exist_ok=True)
		new_path = osp.join(self.root, result_path + "/" + str(self.game_id))
		os.rename(old_path, new_path)
		self.record_episode_result(result)
		self.game_id = -1
