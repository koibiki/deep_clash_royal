from brain.ddpg2 import DDPG
from device.mobile import Mobile
from game.clash_royal import ClashRoyal

# root = "/home/chengli/data/gym_data/clash_royal"
root = "/home/holaverse/work/07battle_filed/clash_royal"

host = ClashRoyal(root, device=Mobile(""), name="trainer")

base_brain = DDPG(host.img_shape, host.state_shape, DDPG.BrainType["trainer"], "trainer")

base_brain.load_memory(root)
for i in range(5000000):
    base_brain.learn()
    # base_brain.load_memory(root)
