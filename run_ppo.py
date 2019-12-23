from brain.ppo_torch import PPO
import numpy as np

brain = PPO()

actor_hidden = None
critic_hidden = None
while True:
    img, env_state, card_type, card_property = \
        np.random.rand(1, 192, 256, 3).astype(np.uint8), np.random.rand(1, 3).astype(np.float), \
        np.random.rand(1, 4).astype(np.int), np.random.rand(1, 8).astype(np.float),
    action, _, actor_hidden = brain.select_action(img, env_state, card_type, card_property, actor_hidden)
    critic, critic_hidden = brain.get_value(img, env_state, card_type, card_property, critic_hidden=critic_hidden)
    print(action)
