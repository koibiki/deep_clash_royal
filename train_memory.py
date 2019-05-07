from brain.base_brain import BaseBrain
from game.clash_royal import ClashRoyal

# root = "/home/chengli/data/gym_data/clash_royal"
root = "/home/holaverse/work/07battle_filed/clash_royal"

clash_royal = ClashRoyal(root)

base_brain = BaseBrain(clash_royal,
                       BaseBrain.BrainType["trainer"])

base_brain.load_memory(root)
for i in range(5000):
    base_brain.learn()
    if i > 0 and i % 1000:
        base_brain.load_memory(root)
