import tensorflow as tf
import numpy as np
import random


def gen_tf_tensor(img, env_state, card_type, card_property):
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    env_state = tf.convert_to_tensor(np.array(env_state).astype(np.float), dtype=tf.float32)
    card_type = tf.convert_to_tensor(np.array(card_type).astype(np.int))
    card_property = tf.convert_to_tensor(np.array(card_property).astype(np.float), dtype=tf.float32)
    return img, env_state, card_type, card_property


def sample_action(card_prob, pos_x_prob, pos_y_prob, choice_card):
    pass


def roulette_sample(datas):
    choice_indices = []
    choice_probs = []
    for data in datas:
        rate_thresh = random.uniform(0, 1)
        all_rate = 0
        scale_data = data / np.sum(data)
        choice_index = 0
        for i, rate in enumerate(scale_data):
            all_rate += rate
            if all_rate >= rate_thresh:
                choice_index = i
        choice_indices.append(choice_index)
        choice_probs.append(data[choice_index])
    return choice_indices, choice_probs
