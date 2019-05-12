import os.path as osp
import random
import time

import cv2
import numpy as np
import tensorflow as tf

from brain.memory import Memory
from game.parse_result import parse_running_state
from net.mobilenet_v2 import build_mobilenetv2
from utils.img_utils import add_salt_and_pepper, add_gaussian_noise


class DDPG(object):
    def __init__(self,
                 a_dim,
                 s_dim,
                 a_bound,
                 gama):
        self.gamma = gama

        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.a = self._build_a(self.S, )
        q = self._build_c(self.S, self.a, )
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - 0.01)  # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]  # soft update operation
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)  # replaced target parameters
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):  # soft replacement happened at here
            q_target = self.R + self.gamma * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

    def _build_actor_net(self, s_img, s_card_elixir, is_train):
        with tf.variable_scope('cnn'):
            # out = resnet_v2.resnet_v2_50(s_img)[1][scope + "/cnn/resnet_v2_50/block4"]
            out = build_mobilenetv2(s_img, is_train)

            loc = tf.layers.average_pooling2d(out, (2, 2), (2, 2), )

            cnn_out = tf.layers.conv2d_transpose(loc, 512, (3, 3), (2, 2), padding="same")

            cnn_concat = tf.concat([out, cnn_out], axis=-1)

            cnn_concat = tf.layers.dropout(cnn_concat, 0.5)

        with tf.variable_scope('executor'):
            s_card_elixir = tf.cast(s_card_elixir, dtype=tf.float32)
            dense = tf.layers.dense(s_card_elixir, 512, kernel_regularizer=self.reg, activation=tf.nn.relu)

            dense = tf.expand_dims(dense, axis=1)
            dense = tf.expand_dims(dense, axis=1)

            h_dense = tf.concat([dense for _ in range(cnn_concat.shape[1].value)], axis=1)
            h_w_dense = tf.concat([h_dense for _ in range(cnn_concat.shape[2].value)], axis=2)

            concat = tf.concat([cnn_concat, h_w_dense], axis=-1)

            global_ave = tf.layers.average_pooling2d(concat, (8, 6))

            card_value = tf.layers.conv2d(global_ave, 93, 1, 1, padding="same")

            # loc 的预测只与已选出的 card 有关
            select_card = tf.argmax(card_value, axis=-1, output_type=tf.int32)
            index_constant = tf.constant([i for i in range(93)], dtype=tf.int32)

            card_type = tf.cast(tf.equal(1, index_constant / select_card), dtype=tf.int32)

            cnn_flatten = tf.layers.flatten(cnn_concat)

            cnn_card = tf.concat([cnn_flatten, card_type], axis=-1)

            loc_x = tf.layers.dense(cnn_card, 1, activation=tf.nn.sigmoid, name="loc_x")

            loc_y = tf.layers.dense(cnn_card, 1, activation=tf.nn.sigmoid, name="loc_y")

        return card_value, loc_x, loc_y
