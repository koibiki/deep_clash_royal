import os
import os.path as osp

root = "../../vysor/fail"

listdir = os.listdir(root)

for dir_name in listdir:
    path = osp.join(osp.join(root, dir_name), "state.txt")
    new_path = osp.join(osp.join(root, dir_name), "states.txt")

    indices = []
    states = []
    with open(path, "r") as f:
        readlines = f.readlines()
        readlines = [line.replace("\n", " ") for line in readlines]
        all_line = "".join(readlines)
        all_line = all_line.replace("[", "").replace("]", ";")
        split = all_line.split(";")
        for item in split[:-1]:
            item_split = item.split(":")
            index = item_split[0]
            state = item_split[1].split(" ")
            indices.append(index)
            states.append(state)
    with open(new_path, "w") as f:
        for i in range(len(indices)):
            f.write(indices[i])
            f.write(":")
            state_str = ""
            for item in states[i]:
                state_str += item + ","
            state_str = state_str[:-1]
            f.write(state_str)
            f.write("\n")
