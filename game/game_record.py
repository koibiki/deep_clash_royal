from utils.logger_utils import logger
import os.path as  osp
import os
import cv2


class Record(object):

	def __init__(self, root):
		self.game_episodes = 0
		self.win_episodes = 0
		self.fail_episodes = 0
		self.draw_episodes = 0
		self.root = root
		self.scale = True

	def init_record(self, game_id):
		self.error_dir = osp.join(self.root, "{:d}/error".format(game_id))
		self.running_dir = osp.join(self.root, "{:d}/running".format(game_id))
		self.finish_dir = osp.join(self.root, "{:d}/finish".format(game_id))
		os.makedirs(self.error_dir)
		os.makedirs(self.running_dir)
		os.makedirs(self.finish_dir)

	def reset(self):
		self.game_episodes = 0
		self.win_episodes = 0
		self.fail_episodes = 0
		self.draw_episodes = 0

	def record_episode_result(self, result):
		if result == -1:
			self.fail_episodes += 1
		elif result == 0:
			self.draw_episodes += 1
		elif result == 1:
			self.win_episodes += 1
		else:
			logger.error("错误的游戏结果.")

	def record_error_img(self, index, img):
		cv2.imwrite(osp.join(self.error_dir, "{:d}.jpg".format(index)), img)

	def record_running_img(self, index, img):
		img_path = osp.join(self.running_dir, "{:d}.jpg".format(index))
		if self.scale:
			cv2.imwrite(img_path, img)
		else:
			cv2.imwrite(img_path, img)

		cv2.imwrite(osp.join(self.running_dir, "{:d}.jpg".format(index)), img)

	def record_finish_img(self, index, img):
		cv2.imwrite(osp.join(self.finish_dir, "{:d}.jpg".format(index)), img)
