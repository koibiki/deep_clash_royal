from brain.base_brain import BaseBrain
from game.clash_royal import ClashRoyal

root = "/home/chengli/data/gym_data/clash_royal"

clash_royal = ClashRoyal(root)

base_brain = BaseBrain(clash_royal.n_loc_x_actions,
                       clash_royal.n_loc_y_actions,
                       clash_royal.n_card_actions,
                       clash_royal.img_shape,
                       clash_royal.state_shape)

base_brain.load_memory(root)
for i in range(5000):
    base_brain.learn()
