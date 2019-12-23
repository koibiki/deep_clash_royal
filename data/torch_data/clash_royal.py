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

		env_states_path = os.path.join(img_id[0], "env_states.json")
		with open(env_states_path) as f:
			env_states = json.load(f)
		env_states = env_states[start_index: start_index + 10]

		card_types_path = os.path.join(img_id[0], "card_types.json")
		with open(card_types_path) as f:
			card_types = json.load(f)
		card_types = card_types[start_index: start_index + 10]

		card_properties_path = os.path.join(img_id[0], "card_properties.json")
		with open(card_properties_path) as f:
			card_properties = json.load(f)
		card_properties = card_properties[start_index: start_index + 10]

		actions_path = os.path.join(img_id[0], "actions.json")
		with open(actions_path) as f:
			actions = json.load(f)
		actions = actions[start_index: start_index + 10]

		reward_path = os.path.join(img_id[0], "rewards.json")
		with open(reward_path) as f:
			reward = json.load(f)
		reward = reward[start_index: start_index + 10]

		return imgs, env_states, card_types, card_properties, actions, reward

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
			data.append([game_id_path, len(os.listdir(os.path.join(game_id_path, "running")))])
		return data


if __name__ == '__main__':
	root = "F:\\gym_data\\clash_royal\\"
	ClayRoyalDataset(root)
