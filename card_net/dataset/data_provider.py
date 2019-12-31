import os.path as osp
import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import *

from card_net.dataset.parse_tfrecords import parse_item
from card_net.config import cfg


class DataProvider(object):

    def __init__(self):
        pass

    @staticmethod
    def _create_dataset_from_dir(root):
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
        return img_paths, type_label, available_label

    def _map_func(self, img_path_tensor, type_label, available_label):
        imread = cv2.imread(img_path_tensor.decode('utf-8'))
        imread = cv2.resize(imread, (64, 64))
        imread = np.expand_dims(imread, axis=-1)
        imread = np.array(imread, np.float32) / 255.
        return imread, type_label, available_label

    def generate_train_input_fn(self):
        root = "F:\\gym_data\\card"
        batch_size = cfg.TRAIN.BATCH_SIZE

        def _input_fn():
            img_paths, type_label, available_label = self._create_dataset_from_dir(root)

            dataset = tf.data.Dataset.from_tensor_slices((img_paths, type_label, available_label)) \
                .map(lambda item1, item2, item3: tf.py_func(self._map_func, [item1, item2, item2],
                                                            [tf.float32, tf.int32, tf.int32])) \
                .shuffle(100)
            dataset = dataset.repeat()
            dataset = dataset.prefetch(32 * batch_size)
            dataset = dataset.batch(batch_size)
            iterator = dataset.make_one_shot_iterator()
            images, labels, labels_len = iterator.get_next()

            features = {'images': images}
            return features, (labels, labels_len)

        return _input_fn


class TfDataProvider(object):

    def __init__(self, lang_dict):
        self.lang_dict = lang_dict

    def generate_train_input_fn(self):
        def _input_fn():
            root = ""
            batch_size = cfg.TRAIN.BATCH_SIZE
            tf_records_path = [osp.join(root, path) for path in os.listdir(root)]
            dataset = tf.data.TFRecordDataset(tf_records_path)
            dataset = dataset.map(parse_item, num_parallel_calls=12).shuffle(32)
            dataset = dataset.repeat()
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(2 * batch_size)
            iterator = dataset.make_one_shot_iterator()
            images, labels = iterator.get_next()

            features = {'images': images}
            return features, labels

        return _input_fn
