from brain.ppo_torch import PPO
from device.emulator import Emulator
from game.clash_royal_env import ClashRoyalEnv

if __name__ == '__main__':

    i = 0

    # root = "/home/chengli/data/gym_data/clash_royal"
    root = "D:\\gym_data\\clash_royal"

    # device_id = "cd9faa7f"

    # address = "http://127.0.0.1:55481/device/" + device_id + "/video.flv"
    # mobile = Mobile(device_id, address)

    device_id = "127.0.0.1:62025"
    device = Emulator(device_id, "two")
    guest = ClashRoyalEnv(root, device=device, mode=ClashRoyalEnv.MODE["friend_battle_guest"], name="guest")

    brain = PPO()

    actor_hidden = None

    while True:
        frame, state_code = device.get_frame()

        if frame is not None:
            observation = guest.frame_step(frame, actor_hidden)
            if observation is not None:
                img, env_state, card_type, card_property, actor_hidden = \
                    [observation[1]], [observation[2]], [observation[3]], [observation[4]], observation[5]
                action, actor_hidden = brain.select_action(img, env_state, card_type, card_property, actor_hidden)
                guest.step(action)

            if guest.game_start and guest.game_finish and guest.retry <= 1:
                # brain.load_model()
                actor_hidden = None
        else:
            if state_code == -1:
                print("没有信号")
                guest.reset()
