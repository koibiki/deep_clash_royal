from brain.ppo_tf import PPO
import numpy as np
# import tensorflow.contrib.eager as tfe
# tfe.enable_eager_execution()
brain = PPO()

actor_hidden = None
critic_hidden = None
while True:
    img, env_state, card_type, card_property = \
        np.random.rand(4, 192, 256, 3).astype(np.uint8), np.random.rand(4,4).astype(np.float), \
        np.random.rand(4, 4).astype(np.int), np.random.rand(4, 8).astype(np.float),
    action, _, actor_hidden = brain.select_action(img, env_state, card_type, card_property, actor_hidden)

    print(action)
