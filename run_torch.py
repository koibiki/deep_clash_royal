import os
import cv2
from brain.ppo_torch import PPO
from device.emulator import Emulator
import numpy as np

if __name__ == '__main__':

    brain = PPO()

    # root = "/home/chengli/data/gym_data/clash_royal"
    root = "F:\\gym_data\\clash_royal"

    host_id = "127.0.0.1:62001"
    guest_id = "127.0.0.1:62025"
    host_device = Emulator(host_id, "one")
    guest_device = Emulator(guest_id, "two")

    from game.agent import Agent

    host = Agent(root, host_device, mode=Agent.MODE["friend_battle_host"], name="host")
    guest = Agent(root, guest_device, mode=Agent.MODE["friend_battle_guest"], name="guest")

    host_actor_hidden = None
    host_critic_hidden = None

    guest_actor_hidden = None
    guest_critic_hidden = None
    host_skip = False
    guest_skip = False
    while True:
        host_frame, host_state_code = host_device.get_frame()

        if host_frame is not None:

            host_observation = host.frame_step(host_frame, host_actor_hidden)
            if host_observation is not None:
                img, env_state, card_type, card_property, host_actor_hidden = \
                    [host_observation[1]], [host_observation[2]], [host_observation[3]], [host_observation[4]], \
                    host_observation[5]
                host_action, _, host_actor_hidden = brain.select_action(img, env_state, card_type, card_property,
                                                                        actor_hidden=host_actor_hidden, skip=host_skip)
                host_skip = host.step(host_action)

            if host.game_start and host.game_finish and host.retry <= 1:
                # brain.load_model()
                host_actor_hidden = None
                host_skip = False

            cv2.imshow("host", host_frame)
        else:
            if host_state_code == -1:
                print("host 没有信号")
        guest_frame, guest_state_code = guest_device.get_frame()

        if guest_frame is not None:
            guest_observation = guest.frame_step(guest_frame, guest_actor_hidden)
            if guest_observation is not None:
                img, env_state, card_type, card_property, guest_actor_hidden = \
                    [guest_observation[1]], [guest_observation[2]], [guest_observation[3]], [guest_observation[4]], \
                    guest_observation[5]
                guest_action, _, guest_actor_hidden = brain.select_action(img, env_state, card_type, card_property,
                                                                          actor_hidden=guest_actor_hidden, skip=guest_skip)
                guest_skip = guest.step(guest_action)

            if guest.game_start and guest.game_finish and guest.retry <= 1:
                # brain.load_model()
                guest_actor_hidden = None
                guest_skip = False

            cv2.imshow("guest", guest_frame)
        else:
            if host_state_code == -1:
                print("host 没有信号")

        cv2.waitKey(1)
