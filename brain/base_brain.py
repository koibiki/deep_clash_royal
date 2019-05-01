import random
from multiprocessing.pool import Pool

import tensorflow as tf
import numpy as np
import time

from brain.memory import Memory


class BaseBrain:
    def __init__(self,
                 n_loc_x_actions,
                 n_loc_y_actions,
                 n_card_actions,
                 img_shape,
                 state_shape,
                 lr=0.0001,
                 reward_decay=0.8,
                 memory_size=5000,
                 batch_size=32,
                 replace_target_iter=30,
                 output_graph=True):
        self.p = Pool(4)
        self.retry = 0
        self.n_loc_x_actions = n_loc_x_actions
        self.n_loc_y_actions = n_loc_y_actions
        self.n_card_actions = n_card_actions
        self.img_shape = img_shape
        self.state_shape = state_shape

        self.reg = tf.contrib.layers.l2_regularizer(0.00001)
        self.lr = lr
        self.gamma = reward_decay
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon = 0.9

        # total learning step
        self.learn_step_counter = 0

        self.memory = Memory(capacity=memory_size)
        self.memory_size = memory_size

        # consist of [target_net, evaluate_net]
        self._build_evaluate_and_target_net()

        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        eval_var = [val for val in vars if 'eval_net' in val.name]
        target_var = [val for val in vars if 'target_net' in val.name]

        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(target_var, eval_var)]

        self.sess = tf.Session()

        if output_graph:
            tf.summary.FileWriter("./logs/" + str(int(time.time() * 1000)), self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self, s_img, s_card_elixir):
        with tf.variable_scope('cnn'):
            conv1 = tf.layers.conv2d(s_img, 32, 3, padding="same", kernel_regularizer=self.reg, activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(conv1, (2, 2), (2, 2), padding="same")
            conv2 = tf.layers.conv2d(pool1, 64, 3, padding="same", kernel_regularizer=self.reg, activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(conv2, (2, 2), (2, 2), padding="same")
            conv3 = tf.layers.conv2d(pool2, 64, 3, padding="same", kernel_regularizer=self.reg, activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(conv3, (2, 2), (2, 2), padding="same")
            conv4 = tf.layers.conv2d(pool3, 128, 3, padding="same", kernel_regularizer=self.reg, activation=tf.nn.relu)
            pool4 = tf.layers.max_pooling2d(conv4, (2, 2), (2, 2), padding="same")
            conv5 = tf.layers.conv2d(pool4, 512, 3, padding="same", kernel_regularizer=self.reg, activation=tf.nn.relu)
            pool5 = tf.layers.max_pooling2d(conv5, (2, 2), (2, 2), padding="same")

        with tf.variable_scope('executor'):
            flatten = tf.layers.flatten(pool5)
            s_card_elixir = tf.cast(s_card_elixir, dtype=tf.float32)
            concat = tf.concat([flatten, s_card_elixir], axis=-1)
            dense1 = tf.layers.dense(concat, 128, kernel_regularizer=self.reg, activation=tf.nn.relu)
            dense2 = tf.layers.dense(dense1, 156, kernel_regularizer=self.reg, activation=tf.nn.relu)

            card_logit = tf.layers.dense(dense2, self.n_card_actions)
            x_logit = tf.layers.dense(dense2, self.n_loc_x_actions)
            y_logit = tf.layers.dense(dense2, self.n_loc_y_actions)

        return card_logit, x_logit, y_logit

    def _build_evaluate_and_target_net(self):

        # ------------------ build evaluate_net ------------------
        # input State
        self.s_img = \
            tf.placeholder(tf.float32, [None, self.img_shape[0], self.img_shape[1], self.img_shape[2]], name='image')
        self.s_card_elixir = tf.placeholder(tf.int32, [None, self.state_shape], name='state')

        self.q_card_target = tf.placeholder(tf.float32, [None, self.n_card_actions], name='Q_card_target')
        self.q_x_target = tf.placeholder(tf.float32, [None, self.n_loc_x_actions], name='Q_x_target')
        self.q_y_target = tf.placeholder(tf.float32, [None, self.n_loc_y_actions], name='Q_y_target')

        self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')

        with tf.variable_scope('eval_net'):
            self.q_card_eval, self.q_x_eval, self.q_y_eval = self._build_net(self.s_img, self.s_card_elixir)

        with tf.variable_scope('loss'):
            reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

            self.abs_errors = tf.reduce_sum(tf.abs(self.q_card_target - self.q_card_eval), axis=1) + \
                              tf.reduce_sum(tf.abs(self.q_x_target - self.q_x_eval), axis=1) + \
                              tf.reduce_sum(tf.abs(self.q_y_target - self.q_y_eval), axis=1)

            card_loss = tf.reduce_mean(tf.squared_difference(self.q_card_target, self.q_card_eval))
            eval_empty = tf.not_equal(0, tf.arg_max(self.q_card_eval, dimension=-1, output_type=tf.int32))
            eval_empty = tf.cast(eval_empty, dtype=tf.float32)
            target_empty = tf.not_equal(0, tf.arg_max(self.q_card_target, dimension=-1, output_type=tf.int32))
            target_empty = tf.cast(target_empty, dtype=tf.float32)
            x_loss = tf.reduce_mean(tf.squared_difference(self.q_x_target, self.q_x_eval) * eval_empty * target_empty)
            y_loss = tf.reduce_mean(tf.squared_difference(self.q_y_target, self.q_y_eval) * eval_empty * target_empty)
            self.loss = card_loss + x_loss + y_loss + reg_loss

        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        # input Next State
        self.s_img_ = \
            tf.placeholder(tf.float32, [None, self.img_shape[0], self.img_shape[1], self.img_shape[2]], name='image_')
        self.s_card_elixir_ = tf.placeholder(tf.int32, [None, self.state_shape], name='state_')
        with tf.variable_scope('target_net'):
            self.q_card_next, self.q_x_next, self.q_y_next = self._build_net(self.s_img_, self.s_card_elixir_)

    def store_transition(self, episode_record):
        for item in episode_record:
            self.memory.store(item)

    def choose_action(self, observation):

        if np.random.uniform() < self.epsilon:
            action = [0, 0, 0]
        else:
            if np.random.uniform() < self.epsilon:
                # forward feed the observation and get q value for every actions
                card_value, x_value, y_value = self.sess.run([self.q_card_eval, self.q_x_eval, self.q_y_eval],
                                                             feed_dict={self.s_img: [observation[0]],
                                                                        self.s_card_elixir: [observation[1]]})
                action = [np.argmax(card_value), np.argmax(x_value), np.argmax(y_value)]
            else:
                card = random.choice(range(self.n_card_actions))
                x_loc = random.choice(range(self.n_loc_x_actions))
                y_loc = random.choice(range(self.n_loc_y_actions))
                action = [card, x_loc, y_loc]
        print("choose action:" + str(action))
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)

        next_imgs = [item[-2] for item in batch_memory]
        next_states = [item[-1] for item in batch_memory]

        imgs = [item[0] for item in batch_memory]
        states = [item[1] for item in batch_memory]
        reward = np.array([item[3] for item in batch_memory])
        q_card_next, q_x_next, q_y_next, q_card_eval, q_x_eval, q_y_eval = self.sess.run(
            [self.q_card_next, self.q_x_next, self.q_y_next, self.q_card_eval, self.q_x_eval, self.q_y_eval],
            feed_dict={self.s_img_: next_imgs,
                       self.s_card_elixir_: next_states,
                       self.s_img: imgs,
                       self.s_card_elixir: states})

        q_card_target = q_card_eval.copy()
        q_x_target = q_x_eval.copy()
        q_y_target = q_y_eval.copy()

        q_card_target = reward + self.gamma * np.max(q_card_target, axis=1)
        q_x_target = reward + self.gamma * np.max(q_x_target, axis=1)
        q_y_target = reward + self.gamma * np.max(q_y_target, axis=1)

        _, abs_errors, self.loss = self.sess.run([self._train_op, self.abs_errors, self.loss],
                                                 feed_dict={self.s_img: imgs,
                                                            self.s_card_elixir: states,
                                                            self.q_card_target: q_card_target,
                                                            self.q_x_target: q_x_target,
                                                            self.q_y_target: q_y_target,
                                                            self.ISWeights: ISWeights})
        self.memory.batch_update(tree_idx, abs_errors)  # update priority

        self.learn_step_counter += 1
