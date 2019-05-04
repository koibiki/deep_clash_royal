import numpy as np
import os
import os.path as osp
from tqdm import *


class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:  # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:  # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1  # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.img_path_dict = {}
        self.state_dict = {}
        self.action_dict = {}
        self.reward_dict = {}

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)  # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n,), dtype=np.object), np.empty((n, 1))
        pri_seg = self.tree.total_p / n  # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1
        # for later calculate ISweight
        min_prob = (np.min(self.tree.tree[-self.tree.capacity:]) + 0.001) / self.tree.total_p
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i], b_memory[i] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    def update_dict(self):
        data = self.tree.data

        keys = set()
        for item in data:
            if type(item) is not np.str:
                continue
            split = item.split("_")
            if len(split) == 2:
                game_id = split[0]
                index = int(split[1])
                for i in range(5):
                    step = 0 if index + 5 - i * 5 < 0 else index + 5 - i * 5
                    keys.add(game_id + "_" + str(step))

        keys = list(keys)
        self.state_dict = self._clear(self.state_dict, keys)
        self.action_dict = self._clear(self.action_dict, keys)
        self.reward_dict = self._clear(self.reward_dict, keys)
        self.img_path_dict = self._clear(self.img_path_dict, keys)

    @staticmethod
    def _clear(dict_collection, keys):
        c_keys = dict_collection.keys()
        delete_keys = [key for key in c_keys if key not in keys]
        for key in delete_keys:
            dict_collection.pop(key)
        return dict_collection

    def load_memory(self, root):
        self._load_fail_memory(osp.join(root, "fail"))
        self._load_win_memory(osp.join(root, "win"))

    def _load_fail_memory(self, root):
        listdir = os.listdir(root)
        for dir_name in tqdm(listdir, "load fail memory"):
            game_id = int(dir_name)
            dir_path = osp.join(root, dir_name)
            self._parse_memory(dir_path, game_id)

    def _load_win_memory(self, root):
        listdir = os.listdir(root)
        for dir_name in tqdm(listdir, desc="load win memory"):
            game_id = int(dir_name)
            dir_path = osp.join(root, dir_name)
            self._parse_memory(dir_path, game_id)

    def _parse_memory(self, root, game_id):
        state_keys = self._parse_states(osp.join(root, "state.txt"), game_id)
        reward_keys = self._parse_rewards(osp.join(root, "reward.txt"), game_id)
        action_keys = self._parse_actions(osp.join(root, "action.txt"), game_id)
        assert self._check_keys(state_keys, reward_keys, action_keys)
        self._parse_imgs_path(osp.join(root, "running"), state_keys)
        self._load_keys(state_keys)

    def _load_keys(self, keys):
        for key in keys:
            self.store(key)

    def _parse_states(self, path, game_id):
        keys = []
        with open(path, "r") as f:
            lines = f.readlines()
            for line in lines:
                split = line.strip().split(":")
                index = int(split[0])
                state_str = split[1].split(",")
                state = [float(item) for item in state_str]
                key = str(game_id) + "_" + str(index)
                self.state_dict[key] = state
                keys.append(key)
        return keys

    def _parse_rewards(self, path, game_id):
        keys = []
        with open(path, "r") as f:
            lines = f.readlines()
            for line in lines:
                split = line.strip().split(":")
                index = int(split[0])
                reward = float(split[1])
                key = str(game_id) + "_" + str(index)
                self.reward_dict[key] = reward
                keys.append(key)
        return keys

    def _parse_actions(self, path, game_id):
        keys = []
        with open(path, "r") as f:
            lines = f.readlines()
            for line in lines:
                split = line.strip().split(":")
                index = int(split[0])
                action_str = split[1].replace("[", "").replace("]", "").replace(" ", "").split(",")
                action = [int(item) for item in action_str]
                key = str(game_id) + "_" + str(index)
                self.action_dict[key] = action
                keys.append(key)
        return keys

    @staticmethod
    def _check_keys(keys0, keys1, keys2):
        if len(keys0) == len(keys1) == len(keys2):
            for item in keys0:
                if item not in keys1:
                    return False
                if item not in keys2:
                    return False
        return True

    def _parse_imgs_path(self, root, keys):
        for key in keys:
            index = int(key.split("_")[-1])
            indices = [index + 5, index, index - 5, index - 5 * 2, index - 5 * 3]
            indices = [max(0, item) for item in indices]
            indices = [min(item, len(keys) - 1) for item in indices]
            img_paths = [osp.join(root, str(item) + ".jpg") for item in indices]
            self.img_path_dict[key] = img_paths
