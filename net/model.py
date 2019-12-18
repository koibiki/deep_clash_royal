import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import numpy as np

"""
借鉴openai five 使用embedding 及 动作softmax, 位置softmax
"""


class BattleFieldFeature(nn.Module):

	def __init__(self):
		super().__init__()
		resnet = torchvision.models.resnet18(False)
		self.conv1 = resnet.conv1
		self.bn1 = resnet.bn1
		self.relu = resnet.relu
		self.maxpool = resnet.maxpool

		self.layer1 = resnet.layer1
		self.layer2 = resnet.layer2
		self.layer3 = resnet.layer3
		self.layer4 = resnet.layer4
		self.downsample = nn.Conv2d(512, 8, 3, 1)
		self.dp = nn.Dropout()

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.downsample(x)
		x = self.relu(x)

		if self.training:
			x = self.dp(x)

		b, c, h, w = x.size()
		x = torch.Tensor.view(x, (b, c * h * w))

		return x


class Actor(nn.Module):
	def __init__(self):
		super().__init__()

		self.hidden_size = 1024
		self.embed_size = 128

		# card indices
		self.card_indices = torch.from_numpy(np.array([i for i in range(94)]))

		# img feature
		self.battle_field_feature = BattleFieldFeature()

		# card index embed
		self.card_embed = nn.Embedding(93, self.embed_size)
		self.dense = nn.Linear(1024, 1024)

		# actor
		self.actor_gru = nn.GRU(1024, self.hidden_size)
		self.use_card = nn.Linear(1024, 2)
		self.intent = nn.Linear(1024, self.embed_size)
		self.pos_x = nn.Linear(1024, 6)
		self.pos_y = nn.Linear(1024, 8)

	def forward(self, img, env_state, card_state, actor_hidden):
		battle_field_feature = self.battle_field_feature(img)
		cat = torch.cat([battle_field_feature, env_state, card_state], dim=1)

		actor_feature = F.relu(self.dense(cat))
		actor_feature = torch.Tensor.unsqueeze(actor_feature, 0)

		# actor run
		actor_output, actor_hidden = self.actor_gru(actor_feature, actor_hidden)

		use_card = F.softmax(self.use_card(actor_output), dim=1)

		action_intent = self.intent(actor_output)

		card_embed = self.card_embed(self.card_indices)
		card = action_intent.float().mm(card_embed.t())

		pos_x = F.softmax(self.pos_x(actor_output))
		pos_y = F.softmax(self.pos_y(actor_output))

		choice_index = torch.argmax(card).cpu().numpy()

		choice_card = card_embed[choice_index]

		return use_card, card, pos_x, pos_y, choice_card, actor_hidden

	def initHidden(self, batch_size):
		result = torch.zeros(1, batch_size, self.hidden_size)
		if torch.cuda.is_available():
			return result.cuda()
		else:
			return result


class CriticPPO(nn.Module):

	def __init__(self):
		super().__init__()

		self.hidden_size = 1024

		# img feature
		self.battle_field_feature = BattleFieldFeature()
		self.dense = nn.Linear(1024, 1024)

		# critic
		self.critic_gru = nn.GRU(1024, self.hidden_size)
		self.q = nn.Linear(1024, 1)

	def forward(self, img, env_state, card_state, critic_hidden):
		battle_field_feature = self.battle_field_feature(img)

		cat = torch.cat([battle_field_feature, env_state, card_state], dim=1)

		feature = F.relu(self.dense(cat))
		critic_feature = torch.Tensor.unsqueeze(feature, 0)

		# critic run
		critic_output, critic_hidden = self.critic_gru(critic_feature, critic_hidden)
		critic_q = self.q(critic_output)

		return critic_q, critic_hidden

	def initHidden(self, batch_size):
		result = torch.zeros(1, batch_size, self.hidden_size)
		if torch.cuda.is_available():
			return result.cuda()
		else:
			return result


class CriticDDPG(nn.Module):
	def __init__(self):
		super().__init__()

		self.hidden_size = 1024

		# img feature
		self.battle_field_feature = BattleFieldFeature()
		self.dense = nn.Linear(1024, 1024)

		# critic
		self.critic_gru = nn.GRU(1024, self.hidden_size)
		self.q = nn.Linear(1024, 1)

	def forward(self, img, env_state, card_state, critic_hidden, use_card, intent, pos_x, pos_y):
		battle_field_feature = self.battle_field_feature(img)

		action_intent = intent * use_card
		action_pos_x = pos_x * use_card
		action_pos_y = pos_y * use_card

		cat = torch.cat([battle_field_feature, env_state, card_state, action_intent, action_pos_x, action_pos_y], dim=1)

		feature = F.relu(self.dense(cat))
		critic_feature = torch.Tensor.unsqueeze(feature, 0)

		# critic run
		critic_output, critic_hidden = self.critic_gru(critic_feature, critic_hidden)
		critic_q = self.q(critic_output)

		return critic_q

	def initHidden(self, batch_size):
		result = torch.zeros(1, batch_size, self.hidden_size)
		if torch.cuda.is_available():
			return result.cuda()
		else:
			return result
