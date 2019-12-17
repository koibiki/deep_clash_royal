import torch
from torch import nn
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


class AlphaNet(nn.Module):
	def __init__(self):
		super().__init__()

		self.hidden_size = 1024

		self.card_indices = torch.from_numpy(np.array([i for i in range(94)]))

		self.battle_field_feature = BattleFieldFeature()

		self.card_embeded = nn.Embedding(94, 64)

		self.dense1 = nn.Linear(1024, 1024)

		self.actor_gru = nn.GRU(1024, self.hidden_size)
		self.critic_gru = nn.GRU(1024, self.hidden_size)

		self.relu = nn.ReLU()

		# actor
		self.intent = nn.Linear(1024, 64)
		self.pos_x = nn.Linear(1024, 6)
		self.pos_y = nn.Linear(1024, 8)

		# critic
		self.q = nn.Linear(1024, 1)

	def forward(self, img, state, actor_hidden, critic_hidden):
		battle_field_feature = self.battle_field_feature(img)
		cat = torch.cat([battle_field_feature, state], dim=1)

		actor_feature = self.dense1(cat)
		actor_feature = self.relu(actor_feature)

		actor_feature = torch.Tensor.unsqueeze(actor_feature, 0)

		# actor run
		actor_output, actor_hidden = self.actor_gru(actor_feature, actor_hidden)

		action_intent = self.intent(actor_output)

		card_embed = self.card_embeded(self.card_indices)

		intent = action_intent.float().mm(card_embed.t())

		pos_x = self.pos_x(actor_output)
		pos_y = self.pos_y(actor_output)

		# critic run
		critic_feature = torch.cat([battle_field_feature, state, intent, pos_x, pos_y], dim=1)

		critic_output, critic_hidden = self.critic_gru(critic_feature, critic_hidden)

		critic_q = self.q(critic_output)

		return intent, pos_x, pos_y, critic_q

	def initHidden(self, batch_size):
		result = torch.zeros(1, batch_size, self.hidden_size)
		if torch.cuda.is_available():
			return result.cuda()
		else:
			return result