import torch
from torch import optim
import torch.nn.functional as F

from net.model import Actor, Critic
from tensorboardX import SummaryWriter


class DDPG(object):
	def __init__(self, ):

		device = torch.device("")

		self.actor = Actor().to(device)
		self.actor_target = Actor().to(device)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_optimizer = optim.Adam(self.actor.parameters(), 0.001)

		self.critic = Critic().to(device)
		self.critic_target = Critic().to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = optim.Adam(self.critic.parameters(), 0.001)
		self.replay_buffer = Replay_buffer()
		self.writer = SummaryWriter(directory)
		self.num_critic_update_iteration = 0
		self.num_actor_update_iteration = 0
		self.num_training = 0

	def select_action(self, img, state):
		state = torch.FloatTensor(img, state).to(device)
		return self.actor(state).cpu().data.numpy().flatten()

	def update(self):

		for it in range(1000):
			# Sample replay buffer
			x, y, u, r, d = self.replay_buffer.sample()
			state = torch.FloatTensor(x).to(device)
			action = torch.FloatTensor(u).to(device)
			next_state = torch.FloatTensor(y).to(device)
			done = torch.FloatTensor(d).to(device)
			reward = torch.FloatTensor(r).to(device)

			# Compute the target Q value
			target_Q = self.critic_target(next_state, self.actor_target(next_state))
			target_Q = reward + ((1 - done) * 0.99 * target_Q).detach()

			# Get current Q estimate
			current_Q = self.critic(state, action)

			# Compute critic loss
			critic_loss = F.mse_loss(current_Q, target_Q)
			self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
			# Optimize the critic
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

			# Compute actor loss
			actor_loss = -self.critic(state, self.actor(state)).mean()
			self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)

			# Optimize the actor
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(0.01 * param.data + (1 - 0.01) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(0.01 * param.data + (1 - 0.01) * target_param.data)

			self.num_actor_update_iteration += 1
			self.num_critic_update_iteration += 1

	def save(self):
		torch.save(self.actor.state_dict(), directory + 'actor.pth')
		torch.save(self.critic.state_dict(), directory + 'critic.pth')
		print("====================================")
		print("Model has been saved...")
		print("====================================")

	def load(self):
		self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
		self.critic.load_state_dict(torch.load(directory + 'critic.pth'))
		print("====================================")
		print("model has been loaded...")
		print("====================================")
