from brain.ppo_torch import PPO
import numpy as np
import random
reward = [0, 0, 0, 0.1, 0, 0, 0.2, 0, 0, 0, 0, 0, 0.6, 0, 0, 0.1, 0, 0.1, -0.3, 0, 0.2, 1]

Gt = []
R = 0
for r in reward[::-1]:
    R = r + 0.5 * R
    Gt.insert(0, R)


def roulette_wheel_selection(data):
    m = random.uniform(0, 1)
    total_prob = 0
    i = 0
    for i, prob in data:
        total_prob += prob
        if total_prob >= m:
            return i
    return i

roulette_wheel_selection([0.1, 0.3, 0.2 ,0.4])


brain = PPO()

actor_hidden = None
critic_hidden = None
while True:
    img, env_state, card_type, card_property = \
        np.random.rand(4, 192, 256, 3).astype(np.uint8), np.random.rand(4, 3).astype(np.float), \
        np.random.rand(4, 4).astype(np.int), np.random.rand(4, 8).astype(np.float),
    action, _, actor_hidden = brain.select_action(img, env_state, card_type, card_property, actor_hidden, choice_index=[0])
    critic, critic_hidden = brain.get_value(img, env_state, card_type, card_property, critic_hidden=critic_hidden)
    print(action)
