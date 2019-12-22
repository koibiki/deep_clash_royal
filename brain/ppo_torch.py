import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.distributions import Categorical
# from torch.utils.tensorboard import SummaryWriter
import time
import os
import numpy as np

from net.torch_model import PpoNet
from utils.func_call import func_time
from utils.torch_util import gen_torch_tensor


class PPO(object):
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 10
    buffer_capacity = 8000
    batch_size = 32
    gamma = 0.99

    def __init__(self, ):
        super(PPO, self).__init__()
        self.device = torch.device("cpu")

        self.ppo_net = PpoNet().to(self.device)

        self.counter = 0
        self.training_step = 0
        # self.writer = SummaryWriter('../exp')

        self.optimizer = optim.Adam(self.ppo_net.parameters(), 1e-3)

        if not os.path.exists('./param'):
            os.makedirs('./param/net_param')

    def _img_transform(self, img):
        return np.array(img).astype(np.float) / 255.

    @func_time
    def select_action(self, img, env_state, card_type, card_property, actor_hidden=None):
        if actor_hidden is None:
            actor_hidden = self.ppo_net.init_hidden(len(img), self.device)

        img, env_state, card_type, card_property = gen_torch_tensor(self._img_transform(img),
                                                                    env_state,
                                                                    card_type,
                                                                    card_property,
                                                                    self.device)
        with torch.no_grad():
            actor = self.ppo_net(img, env_state, card_type, card_property, actor_hidden=actor_hidden)['actor']
        action, choice_card, actor_hidden = actor

        return action, choice_card, actor_hidden

    def get_value(self, img, env_state, card_type, card_property, critic_hidden=None):
        if critic_hidden is None:
            critic_hidden = self.ppo_net.init_hidden(len(img), self.device)

        img, env_state, card_type, card_property = gen_torch_tensor(img, env_state, card_type, card_property,
                                                                    self.device)
        with torch.no_grad():
            critic = self.ppo_net(img, env_state, card_type, card_property, critic_hidden=critic_hidden)['critic']
            value, critic_hidden = critic
        return value.numpy(), critic_hidden

    def save_param(self):
        torch.save(self.ppo_net.state_dict(), '../param/net_param' + str(time.time())[:10] + '.pth')

    def update(self, i_ep):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1)
        reward = [t.reward for t in self.buffer]
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + self.gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float)
        print("The agent is updateing....")
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                if self.training_step % 1000 == 0:
                    print('I_ep {} ，train {} times'.format(i_ep, self.training_step))
                # with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1)
                V = self.ppo_net(state[index])
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                action_prob = self.ppo_net(state[index]).gather(1, action[index])  # new policy

                ratio = (action_prob / old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor critic network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                self.optimizer.zero_grad()
                action_loss.backward()
                value_loss = F.mse_loss(Gt_index, V)
                all_loss = action_loss + value_loss

                nn.utils.clip_grad_norm_(self.ppo_net.parameters(), self.max_grad_norm)

                # self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)
                # self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
                all_loss.backward()

                self.optimizer.step()
                self.training_step += 1
