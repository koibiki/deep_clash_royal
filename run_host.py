# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from brain.ppo_torch import PPO
from device.emulator import Emulator
from game.clash_royal_env import ClashRoyalEnv

if __name__ == '__main__':

    i = 0

    # root = "/home/chengli/data/gym_data/clash_royal"
    root = "F:\\gym_data\\clash_royal"

    # device_id = "cd9faa7f"
    device_id = "127.0.0.1:62001"

    host_address = "http://127.0.0.1:55481/device/" + device_id + "/video.flv"

    device = Emulator(device_id, "one")
    host = ClashRoyalEnv(root, device=device, mode=ClashRoyalEnv.MODE["friend_battle_host"], name="host")

    brain = PPO()

    actor_hidden = None

    while True:
        frame, state_code = device.get_frame()

        if frame is not None:
            host_observation = host.frame_step(frame, actor_hidden)
            if host_observation is not None:
                host_action, actor_hidden = brain.select_action(host_observation)
                host.step(host_action)

            if host.game_start and host.game_finish and host.retry <= 1:
                # brain.load_model()
                actor_hidden = None
        else:
            if state_code == -1:
                print("没有信号")
                host.reset()
