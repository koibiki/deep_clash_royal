from brain.ddpg import DDPG
from game.clash_royal import ClashRoyal

# root = "/home/chengli/data/gym_data/clash_royal"
root = "/home/holaverse/work/07battle_filed/clash_royal"

host = ClashRoyal(root, device_id="cd9faa7f", name="trainer")

base_brain = DDPG(host.img_shape, host.state_shape, DDPG.BrainType["trainer"], "trainer")

base_brain.load_memory(root)
for i in range(500000):
    base_brain.learn()
    if i > 0 and i % 100 == 0:
        base_brain.load_memory(root)
