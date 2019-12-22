import os
import numpy as np

from net.tf_model import PpoNet
from utils.func_call import func_time
from utils.tf_util import gen_tf_tensor


class PPO(object):
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 10
    buffer_capacity = 8000
    batch_size = 32
    gamma = 0.99

    def __init__(self, ):
        super(PPO, self).__init__()

        self.ppo_net = PpoNet()

        self.counter = 0
        self.training_step = 0
        # self.writer = SummaryWriter('../exp')

        # self.optimizer = optim.Adam(self.ppo_net.parameters(), 1e-3)

        if not os.path.exists('./param'):
            os.makedirs('./param/net_param')

    def _img_transform(self, img):
        return np.array(img).astype(np.float) / 255.

    @func_time
    def select_action(self, img, env_state, card_type, card_property, actor_hidden=None):
        if actor_hidden is None:
            actor_hidden = self.ppo_net.initialize_hidden_state()

        img, env_state, card_type, card_property = \
            gen_tf_tensor(self._img_transform(img), env_state, card_type, card_property)

        action = self.ppo_net(img, env_state, card_type, card_property, actor_hidden=actor_hidden)['actor']
        card_prob, pos_x_prob, pos_y_prob, choice_card, actor_hidden = action

        action_card = np.argmax(card_prob, -1)
        action_card_prob = np.max(card_prob, -1)
        action_pos_x = np.argmax(pos_x_prob, -1)
        action_pos_x_prob = np.max(pos_x_prob, -1)
        action_pos_y = np.argmax(pos_y_prob, -1)
        action_pos_y_prob = np.max(pos_y_prob, -1)

        return {"card": [int(i) for i in list(action_card)],
                'card_prob': [float(i) for i in list(action_card_prob)],
                "pos_x": [int(i) for i in list(action_pos_x)],
                "pos_x_prob": [float(i) for i in list(action_pos_x_prob)],
                "pos_y": [int(i) for i in list(action_pos_y)],
                "pos_y_prob": [float(i) for i in list(action_pos_y_prob)]}, choice_card, actor_hidden
