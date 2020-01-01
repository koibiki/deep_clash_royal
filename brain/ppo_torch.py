import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import os
import numpy as np
import random

from data.torch_data.clash_royal import ClayRoyalDataset
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

    def __init__(self, device="cpu"):
        super(PPO, self).__init__()
        self.device = torch.device(device)

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
    def select_action(self, img, env_state, card_type, card_property, actor_hidden=None, choice_index=None):
        if actor_hidden is None:
            actor_hidden = self.ppo_net.init_hidden(len(img), self.device)

        img, env_state, card_type, card_property = gen_torch_tensor(self._img_transform(img),
                                                                    env_state,
                                                                    card_type,
                                                                    card_property,
                                                                    self.device)
        with torch.no_grad():
            actor = self.ppo_net(img, env_state, card_type, card_property,
                                 actor_hidden=actor_hidden, choice_index=choice_index)['actor']
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

    def save(self):
        torch.save(self.ppo_net.state_dict(), './param/params.pth')

    def load(self):
        self.ppo_net.load_state_dict(torch.load('./param/params.pth', map_location=self.device))

    def save_param(self):
        torch.save(self.ppo_net.state_dict(), '../param/' + str(time.time())[:10] + '.pth')

    def learn(self, ):

        root = "F:\\gym_data\\clash_royal\\"
        dataset = ClayRoyalDataset(root)

        dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

        total_loss = 0
        total_value = 0
        iter = 0

        for epoch in range(1000):
            for i, (imgs, env_states, card_types, card_properties, action_index, action_prob, reward) in enumerate(
                    dataloader):

                imgs = imgs.to(self.device)
                env_states = env_states.to(self.device)
                card_types = card_types.to(self.device)
                card_properties = card_properties.to(self.device)
                action_index = action_index.to(self.device)
                action_prob = action_prob.to(self.device)
                reward = reward.to(self.device)

                actor_hidden = self.ppo_net.init_hidden(len(imgs), self.device)
                critic_hidden = self.ppo_net.init_hidden(len(imgs), self.device)
                step_count = random.randint(5, 10)
                gt_reward = reward[:, step_count - 1]
                gt_reward = torch.Tensor.view(gt_reward, (-1, 1))
                for step in range(step_count):
                    result = self.ppo_net(imgs[:, step, :, :, :],
                                          env_states[:, step, :],
                                          card_types[:, step, :],
                                          card_properties[:, step, :],
                                          actor_hidden, critic_hidden, choice_index=action_index[:, step, 0])

                    action, _, actor_hidden = result['actor']
                    critic, critic_hidden = result['critic']

                advantage = gt_reward - critic

                advantage = (advantage - torch.mean(advantage)) / (torch.std(advantage) + 1e-8)

                new_card_prob = torch.from_numpy(np.array(action['card_prob'])).to(self.device)
                old_card_log_prob = action_prob[:, step_count - 1, 0]
                ratio = torch.div(new_card_prob, old_card_log_prob).view((-1, 1))
                card_surr1 = ratio * advantage
                card_surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                new_pos_x_prob = torch.from_numpy(np.array(action['pos_x_prob'])).to(self.device)
                old_pos_x_log_prob = action_prob[:, step_count - 1, 1]
                pos_x_ratio = torch.div(new_pos_x_prob, old_pos_x_log_prob).view((-1, 1))
                pos_x_surr1 = pos_x_ratio * advantage
                pos_x_surr2 = torch.clamp(pos_x_ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                new_pos_y_prob = torch.from_numpy(np.array(action['pos_y_prob'])).to(self.device)
                old_pos_y_log_prob = action_prob[:, step_count - 1, 2]
                pos_y_ratio = torch.div(new_pos_y_prob, old_pos_y_log_prob).view((-1, 1))
                pos_y_surr1 = pos_y_ratio * advantage
                pos_y_surr2 = torch.clamp(pos_y_ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                self.optimizer.zero_grad()
                # update actor critic network
                card_action_loss = -torch.min(card_surr1, card_surr2).mean()  # MAX->MIN desent
                pos_x_action_loss = -torch.min(pos_x_surr1, pos_x_surr2).mean()
                pos_y_action_loss = -torch.min(pos_y_surr1, pos_y_surr2).mean()

                value_loss = F.mse_loss(gt_reward, critic)
                all_loss = card_action_loss + pos_x_action_loss + pos_y_action_loss + value_loss

                nn.utils.clip_grad_norm_(self.ppo_net.parameters(), self.max_grad_norm)

                # self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)
                # self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
                all_loss.backward()

                total_loss += all_loss.item()
                total_value += value_loss.item()
                iter += 1

                self.optimizer.step()
                print("epoch {}:{} all loss:{} value_loss:{} mean:{} mean value:{} ratio：{}".format(epoch, i,
                                                                                                    all_loss.item(),
                                                                                                    value_loss.item(),
                                                                                                    total_loss / iter,
                                                                                                    total_value / iter,
                                                                                                    ratio.mean().item()))
            if epoch % 100 == 0:
                self.save()

        self.save()
