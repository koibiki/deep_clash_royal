import json
import os
import random

import cv2
import torch.utils.data as data


class ClayRoyalDataset(data.Dataset):

	def __init__(self, root, transforms=None, ):
		self.root = root
		self.transforms = transforms

		self.data = self.read_game_data(root)

	def update(self, ):
		self.data = self.read_game_data(self.root)

	def __getitem__(self, index):
		img_id = self.data[index]
		start_index = random.randint(0, img_id[1] - 10)

		imgs = []
		for i in range(start_index, start_index + 10):
			img_path = os.path.join(img_id[0], "running/{}.jpg".format(i))
			imgs.append(cv2.imread(img_path))

		env_state_path = os.path.join(img_id[0], "env_state.json")
		with open(env_state_path) as f:
			env_states = json.load(f)
		env_states = env_states[start_index: start_index + 10]

		card_type_path = os.path.join(img_id[0], "card_type.json")
		with open(card_type_path) as f:
			card_type = json.load(f)
		card_type = card_type[start_index: start_index + 10]

		card_property_path = os.path.join(img_id[0], "card_property.json")
		with open(card_property_path) as f:
			card_property = json.load(f)
		card_property = card_property[start_index: start_index + 10]

		action_path = os.path.join(img_id[0], "action.json")
		with open(action_path) as f:
			actions = json.load(f)
		actions = actions[start_index: start_index + 10]

		reward_path = os.path.join(img_id[0], "reward.json")
		with open(reward_path) as f:
			reward = json.load(f)
		reward = reward[start_index: start_index + 10]

		return imgs, env_states, card_type, card_property, actions, reward

	def __len__(self):
		return len(self.data)

	def read_game_data(self, root):
		win = self._read_game_data_by_dir(os.path.join(root, "win"))
		fail = self._read_game_data_by_dir(os.path.join(root, "fail"))
		draw = self._read_game_data_by_dir(os.path.join(root, "draw"))
		data = win + fail + draw
		return data

	@staticmethod
	def _read_game_data_by_dir(root):
		game_ids = os.listdir(root)
		data = []
		for game_id in game_ids:
			game_id_path = os.path.join(root, game_id)
			data_dir_names = os.listdir(game_id_path)
			data_dir_paths = [os.path.join(game_id_path, data_dir_name) for data_dir_name in data_dir_names]
			for data_dir_path in data_dir_paths:
				data.append([data_dir_path, len(os.listdir(data_dir_path))])
		return data


if __name__ == '__main__':
	pass
