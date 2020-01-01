import os
import os.path as osp
import random

import cv2
import numpy as np
from PIL import Image
from tqdm import *

from card_net.config import cfg


class DataProvider(object):

    def __init__(self):
        root = "F:\\gym_data\\card"
        self._batch_size = cfg.TRAIN.BATCH_SIZE
        self.data = self._create_dataset_from_dir(root)
        self.indices = [i for i in range(len(self.data[0]))]

    def _create_dataset_from_dir(self, root):
        img_paths = []
        type_label = []
        available_label = []
        for dir_name in tqdm(os.listdir(root), desc="read dir"):
            img_dir = os.path.join(root, dir_name)

            img_paths0 = [osp.join(osp.join(img_dir, "0"), img_name) for img_name in os.listdir(osp.join(img_dir, "0"))]
            type_label0 = [int(dir_name.split("_")[0]) for _ in img_paths0]
            available_label0 = [0 for _ in img_paths0]

            img_paths1 = [osp.join(osp.join(img_dir, "1"), img_name) for img_name in os.listdir(osp.join(img_dir, "1"))]
            type_label1 = [int(dir_name.split("_")[0]) for _ in img_paths1]
            available_label1 = [1 for _ in img_paths1]

            img_paths += img_paths0 + img_paths1
            type_label += type_label0 + type_label1
            available_label += available_label0 + available_label1
        return np.array(img_paths), np.array(type_label), np.array(available_label)

    def _map_func(self, img_path):
        imread = cv2.imread(img_path)

        if random.uniform(0, 1) < 0.5:
            h, w, c = imread.shape
            fromarray = Image.fromarray(imread)
            fromarray = fromarray.crop((0, random.randint(0, h // 4), w, h))
            imread = np.array(fromarray)

        imread = cv2.resize(imread, (64, 64))
        if random.uniform(0, 1) < 0.5:
            imread = imread * (np.random.randint(85, 115, imread.shape) / 100)

        imread = np.array(imread, np.float32) / 255.
        return imread

    def _sample_data(self, indices):
        return [self._map_func(item) for item in self.data[0][indices]], self.data[1][indices], self.data[2][indices]

    def generate_data(self):
        samples = random.sample(self.indices, self._batch_size)
        return self._sample_data(samples)
