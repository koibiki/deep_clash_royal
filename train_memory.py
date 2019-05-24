from brain.policy import PolicyGradient
from device.emulator import Emulator
from device.mobile import Mobile
from game.clash_royal import ClashRoyal

# root = "/home/chengli/data/gym_data/clash_royal"
root = "F:\\gym_data\\clash_royal"

host = ClashRoyal(root, None, name="trainer")

base_brain = PolicyGradient(host.img_shape, host.state_shape, PolicyGradient.BrainType["trainer"], "trainer")

base_brain.load_memory(root)
for i in range(5000000):
    base_brain.learn()
    if i > 0 and i % 100 == 0:
        base_brain.load_memory(root)
