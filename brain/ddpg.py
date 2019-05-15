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
    BrainType = {"double": 0,
                 "runner": 1,
                 "trainer": 2}

    def __init__(self,
                 img_shape,
                 state_shape,
                 brain_type,
                 name,
                 lr=0.001,
                 gama=0.9,
                 memory_size=50000,
                 batch_size=16, ):

        self.brain_type = brain_type

        self.img_shape = img_shape
        self.state_shape = state_shape

        self.reg = tf.contrib.layers.l2_regularizer(0.00001)
        self.lr = lr
        self.gamma = gama

        self.batch_size = batch_size if brain_type != self.BrainType["runner"] else 1
        self.epsilon = 0.9

        self.memory = Memory(capacity=memory_size)
        self.memory_size = memory_size

        self.rate_of_winning = 0
        self.reward_sum = 0
        self.learn_step_counter = 0

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4
        sess_config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=sess_config)

        with self.sess.as_default():
            self._build_eval_target_net()

            self.saver = tf.train.Saver(max_to_keep=3)
            train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            model_name = 'net_{:s}.ckpt'.format(str(train_start_time))
            self.model_save_path = osp.join("./checkpoints", model_name)

            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())

            self.load_model()

            self.writer = tf.summary.FileWriter("./logs/" + name)
            self.writer.add_graph(self.sess.graph)
            self.merge_summary = tf.summary.merge_all()

    def _build_actor_critic_net(self, s_img, s_card_elixir, is_train=True, ):
        with tf.variable_scope('feature', ):
            # out = resnet_v2.resnet_v2_50(s_img)[1][scope + "/cnn/resnet_v2_50/block4"]
            out = build_mobilenetv2(s_img, is_train)

            loc = tf.layers.average_pooling2d(out, (2, 2), (2, 2), )

            cnn_out = tf.layers.conv2d_transpose(loc, 512, (3, 3), (2, 2), padding="same")

            cnn_concat = tf.concat([out, cnn_out], axis=-1)

            cnn_concat = tf.layers.dropout(cnn_concat, 0.5)

            cnn_flatten = tf.layers.flatten(cnn_concat)

            env_state = s_card_elixir[:, 92 * 2:]

            s_card_elixir = tf.cast(s_card_elixir, dtype=tf.float32)
            dense = tf.layers.dense(s_card_elixir, 512, kernel_regularizer=self.reg, activation=tf.nn.relu)

            global_ave = tf.layers.average_pooling2d(cnn_concat, (8, 6), 1)

            h_dense = tf.expand_dims(dense, axis=1)
            h_w_dense = tf.expand_dims(h_dense, axis=1)

            card_state = tf.concat([global_ave, h_w_dense], axis=-1)
            squeeze = tf.squeeze(card_state, axis=[1, 2])
        with tf.variable_scope('actor', ):
            card_dense = tf.layers.dense(squeeze, 512, activation=tf.nn.relu)

            card_value = tf.layers.dense(card_dense, 93, name="card_value")

            card_prob = tf.nn.softmax(card_value, name="card_prob")

            # loc 的预测只与已选出的 card 有关
            select_card = tf.argmax(card_value, axis=-1, output_type=tf.int32)
            card_type = tf.one_hot(select_card, 93)

            cnn_card = tf.concat([cnn_flatten, card_type, env_state], axis=-1)

            loc_x = tf.layers.dense(cnn_card, 1, activation=tf.nn.sigmoid, name="loc_x")

            loc_y = tf.layers.dense(cnn_card, 1, activation=tf.nn.sigmoid, name="loc_y")

        with tf.variable_scope("critic", ):
            card_state_type = tf.concat([squeeze, card_type], axis=-1)

            cnn_card_loc_x = tf.concat([cnn_card, loc_x], axis=-1)
            cnn_card_loc_y = tf.concat([cnn_card, loc_y], axis=-1)

            q_card = tf.layers.dense(card_state_type, 1, name="q_card")

            q_loc_x = tf.layers.dense(cnn_card_loc_x, 1, name="q_loc_x")

            q_loc_y = tf.layers.dense(cnn_card_loc_y, 1, name="q_loc_y")

        return card_prob, card_value, loc_x, loc_y, q_card, q_loc_x, q_loc_y

    def _build_eval_target_net(self):

        self._global_step = tf.Variable(0, name='global_step', trainable=False)
        self._rate_of_winning = tf.Variable(0.0, name='rate_of_winning', dtype=tf.float32, trainable=False)
        self._reward = tf.Variable(0.0, name='reward', dtype=tf.float32, trainable=False)

        if self.brain_type != self.BrainType["trainer"]:
            with tf.variable_scope('rate_of_winning'):
                tf.summary.scalar(name='rate_of_winning', tensor=self._rate_of_winning)
                tf.summary.scalar(name='reward_sum', tensor=self._reward)
        # ------------------ build evaluate_net ------------------
        # input State
        self.s_img = \
            tf.placeholder(tf.float32, [None, self.img_shape[0], self.img_shape[1], self.img_shape[2]], name='image')
        self.s_card_elixir = tf.placeholder(tf.float32, [None, self.state_shape], name='state')

        self.card_action = tf.placeholder(tf.int32, [None, ], name='card_action')
        self.x_action = tf.placeholder(tf.float32, [None, ], name='x_action')
        self.y_action = tf.placeholder(tf.float32, [None, ], name='y_action')

        self.q_card_target = tf.placeholder(tf.float32, [None, 1], name='Q_card_target')
        self.q_x_target = tf.placeholder(tf.float32, [None, 1], name='Q_x_target')
        self.q_y_target = tf.placeholder(tf.float32, [None, 1], name='Q_y_target')

        self.action_reward = tf.placeholder(tf.float32, [None], name='action_reward')

        # eval_net
        with tf.variable_scope('eval_net'):
            self.card_prob, self.card_value, self.loc_x, self.loc_y, self.q_card_eval, self.q_x_eval, self.q_y_eval = \
                self._build_actor_critic_net(self.s_img, self.s_card_elixir, is_train=True)

        with tf.variable_scope('loss'):
            reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

            # has card select
            select_card = tf.argmax(self.card_value, axis=-1, output_type=tf.int32)

            self.accuracy = tf.reduce_mean(tf.metrics.accuracy(select_card, self.card_action))

            same_card = tf.cast(tf.equal(select_card, self.card_action), dtype=tf.float32) * \
                        tf.cast(tf.not_equal(self.card_action, 0), dtype=tf.float32)

            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.card_value,
                                                                          labels=self.card_action)

            card_loss = tf.reduce_mean(neg_log_prob * self.action_reward)

            loc_x_loss = -tf.reduce_mean(
                tf.squared_difference(self.loc_x, self.x_action) * same_card * self.action_reward)
            loc_y_loss = -tf.reduce_mean(
                tf.squared_difference(self.loc_y, self.y_action) * same_card * self.action_reward)

            self.abs_errors = tf.reduce_sum(tf.abs(self.q_card_target - self.q_card_eval), axis=1) + \
                              tf.reduce_sum(tf.abs(self.q_x_target - self.q_x_eval), axis=1) * same_card + \
                              tf.reduce_sum(tf.abs(self.q_y_target - self.q_y_eval), axis=1) * same_card

            q_card_loss = tf.reduce_mean((tf.squared_difference(self.q_card_target, self.q_card_eval)))

            q_x_loss = tf.reduce_mean(tf.squared_difference(self.q_x_target, self.q_x_eval) * same_card)
            q_y_loss = tf.reduce_mean(tf.squared_difference(self.q_y_target, self.q_y_eval) * same_card)

            self.a_loss = card_loss + loc_x_loss + loc_y_loss + reg_loss / 2

            self.c_loss = q_card_loss + q_x_loss + q_y_loss + reg_loss / 2

            self.loss = self.c_loss

            if self.brain_type != self.BrainType["runner"]:
                tf.summary.scalar(name="accuracy", tensor=self.accuracy)
                tf.summary.scalar(name='loss', tensor=self.loss)
                tf.summary.scalar(name='actor_loss', tensor=self.a_loss)
                tf.summary.scalar(name='critic_loss', tensor=self.c_loss)
                tf.summary.scalar(name='card_loss', tensor=card_loss)
                tf.summary.scalar(name='loc_x_loss', tensor=loc_x_loss)
                tf.summary.scalar(name='loc_y_loss', tensor=loc_y_loss)
                tf.summary.scalar(name='card_q_loss', tensor=q_card_loss)
                tf.summary.scalar(name='loc_x_q_loss', tensor=q_x_loss)
                tf.summary.scalar(name='loc_y_q_loss', tensor=q_y_loss)

        # ------------------ build target_net ------------------
        # input Next State
        self.s_img_ = \
            tf.placeholder(tf.float32, [None, self.img_shape[0], self.img_shape[1], self.img_shape[2]], name='image_')
        self.s_card_elixir_ = tf.placeholder(tf.float32, [None, self.state_shape], name='state_')

        # target_net
        with tf.variable_scope('target_net'):
            _, _, _, _, self.q_card_next, self.q_x_next, self.q_y_next = \
                self._build_actor_critic_net(self.s_img_, self.s_card_elixir_, False)

        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        eval_vars = [val for val in vars if 'eval_net' in val.name]
        target_vars = [val for val in vars if 'target_net' in val.name]

        a_eval_params = [val for val in eval_vars if "critic" not in val.name]
        c_eval_params = [val for val in eval_vars if "actor" not in val.name]

        self.soft_replace = [tf.assign(t, (1 - 0.01) * t + 0.01 * e) for t, e in zip(target_vars, eval_vars)]

        with tf.variable_scope('train'):
            self._train_actor_op = tf.train.AdamOptimizer(self.lr).minimize(self.a_loss, var_list=a_eval_params)
            self._train_critic_op = tf.train.AdamOptimizer(self.lr).minimize(self.c_loss, var_list=c_eval_params)

    def store_transition(self, episode_record):
        game_id = episode_record[0]
        img_paths = episode_record[1]
        states = episode_record[2]
        actions = episode_record[3]
        rewards = episode_record[4]
        for index in range(len(img_paths)):
            self.memory.store(str(game_id) + "_" + str(index))
            self.memory.img_path_dict[str(game_id) + "_" + str(index)] = img_paths[index]
            self.memory.state_dict[str(game_id) + "_" + str(index)] = states[index]
            self.memory.action_dict[str(game_id) + "_" + str(index)] = actions[index]
            self.memory.reward_dict[str(game_id) + "_" + str(index)] = rewards[index]
        self.memory.update_dict()

    def update_episode_result(self, result):
        self.rate_of_winning = result[0]
        self.reward_sum = result[1]

    def choose_action(self, observation):
        uniform = np.random.uniform()
        if uniform <= 0.6:
            # forward feed the observation and get q value for every actions
            card_prob, card, loc_x, loc_y = self.sess.run([self.card_prob, self.card_value, self.loc_x, self.loc_y],
                                                          feed_dict={self.s_img: [observation[1]],
                                                                     self.s_card_elixir: [
                                                                         parse_running_state(observation[2])]})
            card_index = np.argmax(card_prob[0])
            print("dqn play:" + str(card_index) + " " + str(card_prob[0][card_index]))
            if card_index != 0:
                action = [card_index, loc_x[0][0], loc_y[0][0]]
                print("dqn play has action:" + str(action))
            else:
                action = [0, 0, 0]
            print("dqn choose action:" + str(action) + "  " + str(observation[3]) + " " + str(observation[4]))
        elif uniform < 0.7:
            action = [0, 0, 0]
            print("random choose action:" + str(action) + "  " + str(observation[3]) + " " + str(observation[4]))
        elif uniform < 0.85:
            card = random.choice(range(1, 93))
            x_loc = np.random.uniform()
            y_loc = np.random.uniform()
            action = [card, x_loc, y_loc]
            print("random choose action:" + str(action) + "  " + str(observation[3]) + " " + str(observation[4]))
        else:
            card = random.choice(observation[3])
            x_loc = np.random.uniform()
            y_loc = np.random.uniform()
            action = [card, x_loc, y_loc]
            print("random choose action:" + str(action) + "  " + str(observation[3]) + " " + str(observation[4]))
        return action

    @staticmethod
    def _process_img(img_path, flip):
        img = cv2.imread(img_path)
        if img is None:
            print(img_path)
        h, w, c = img.shape
        if h != 256 or w != 192:
            w = 1080
            num_align_width = 7
            h_gap = w // num_align_width
            img = img[h_gap // 2 + h_gap // 8: 9 * h_gap // 2 + h_gap // 8, h_gap // 4:-h_gap // 4, :]
            img = cv2.resize(img, (192, 256))

        randint = random.randint(0, 2)
        if randint == 1:
            img = add_salt_and_pepper(img, random.randint(1, 5) * 0.01)
        elif randint == 2:
            img = add_gaussian_noise(img, random.randint(1, 5) * 0.01)

        if flip:
            img = cv2.flip(img, 1)
        return img / 255.

    def _prepare_data(self, batch_memory):
        imgs = []
        next_imgs = []
        x_action = []
        for item in batch_memory:
            img_paths = self.memory.img_path_dict[item]

            flip = random.randint(0, 1) == 0
            step_img = [self._process_img(img_path, flip) for img_path in img_paths[1:]]
            next_step_img = [self._process_img(img_path, flip) for img_path in img_paths[:4]]

            step_img = np.concatenate(step_img, axis=-1)
            next_step_img = np.concatenate(next_step_img, axis=-1)
            imgs.append(step_img)
            next_imgs.append(next_step_img)
            if flip:
                action = 1.0 - self.memory.action_dict[item][1]
            else:
                action = self.memory.action_dict[item][1]
            x_action.append(action)

        imgs = np.array(imgs)
        next_imgs = np.array(next_imgs)

        next_states = []
        for item in batch_memory:
            game_id = item.split("_")[0]
            index = int(item.split("_")[1])
            next_index = index + 5
            next_state = self.memory.state_dict.get(game_id + "_" + str(next_index))
            if next_state is None:
                next_state = self.memory.state_dict[item]
            next_states.append(next_state)

        next_states = np.array(next_states)

        states = np.array([self.memory.state_dict[item] for item in batch_memory])
        card_type_action = [self.memory.action_dict[item][0] for item in batch_memory]
        y_action = [self.memory.action_dict[item][2] for item in batch_memory]
        reward = np.array([self.memory.reward_dict[item] for item in batch_memory])
        card_action = np.array([[card_type_action[i], x_action[i], y_action[i]] for i in range(len(card_type_action))])

        return imgs, states, card_action, reward, next_imgs, next_states

    def _discount_and_norm_rewards(self, reward):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(reward)
        running_add = 0
        for t in reversed(range(0, len(reward))):
            running_add = running_add * self.gamma + reward[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

    def learn(self):
        with self.sess.as_default():
            start_time = time.time() * 1000
            self.sess.run(self.soft_replace)

            if self.brain_type != self.BrainType["trainer"]:
                self.sess.run(tf.assign(self._rate_of_winning, self.rate_of_winning))
                self.sess.run(tf.assign(self._reward, self.reward_sum))

            while True:
                tree_idx, batch_memory, weights = self.memory.sample(self.batch_size)
                imgs, states, actions, reward, next_imgs, next_states \
                    = self._prepare_data(batch_memory)
                undo_count = np.sum([1 for action in actions if action[0] == 0])
                if undo_count < self.batch_size / 3:
                    break

            states_vector = [parse_running_state(state) for state in states]
            next_states_vector = [parse_running_state(state) for state in next_states]

            q_card_next, q_card_eval, q_x_next, q_x_eval, q_y_next, q_y_eval = self.sess.run(
                [self.q_card_next, self.q_card_eval, self.q_x_next, self.q_x_eval, self.q_y_next, self.q_y_eval],
                feed_dict={self.s_img_: next_imgs,
                           self.s_card_elixir_: next_states_vector,
                           self.s_img: imgs,
                           self.s_card_elixir: states_vector,
                           self.card_action: actions[:, 0].astype(np.int32),
                           self.x_action: actions[:, 1],
                           self.y_action: actions[:, 2]})

            q_card_target = q_card_eval.copy()
            q_x_target = q_x_eval.copy()
            q_y_target = q_y_eval.copy()

            # reward = self._discount_and_norm_rewards(reward)

            for i in range(len(q_card_target)):
                action_item = actions[i]
                q_card_target[i] = reward[i] + self.gamma * q_card_next[i]
                if action_item[0] != 0:
                    state_card = states[i][:4]
                    state_available = states[i][4:8]
                    available_card = state_card * state_available
                    if action_item[0] in available_card:
                        q_x_target[i] = reward[i] + self.gamma * q_x_next[i]
                        q_y_target[i] = reward[i] + self.gamma * q_y_next[i]

            _, _, _, abs_errors, loss, summary = self.sess.run(
                [self._train_actor_op, self._train_critic_op, self.accuracy,
                 self.abs_errors, self.loss, self.merge_summary],
                feed_dict={self.s_img: imgs,
                           self.s_card_elixir: states_vector,
                           self.q_card_target: q_card_target,
                           self.q_x_target: q_x_target,
                           self.q_y_target: q_y_target,
                           self.action_reward: reward,
                           self.card_action: actions[:, 0].astype(np.int32),
                           self.x_action: actions[:, 1],
                           self.y_action: actions[:, 2]
                           })

            self.memory.batch_update(tree_idx, abs_errors)  # update priority

            self.writer.add_summary(summary=summary, global_step=self.learn_step_counter)

            self.learn_step_counter += 1

            if self.learn_step_counter > 0 and self.learn_step_counter % 25 == 0:
                print('\nSave weights {:d}\n'.format(self.learn_step_counter))
                self.saver.save(self.sess, self.model_save_path, global_step=self.learn_step_counter)
            print("Train spent {:d}  {:f}".format(self.learn_step_counter, time.time() * 1000 - start_time))

    def record_battle_result(self):
        with self.sess.as_default():
            self.sess.run(tf.assign(self._rate_of_winning, self.rate_of_winning))
            self.sess.run(tf.assign(self._reward, self.reward_sum))
            summary = self.sess.run([self.merge_summary], )

            self.writer.add_summary(summary=summary[0], global_step=self.learn_step_counter)

            self.learn_step_counter += 1

    def load_model(self):
        ckpt = tf.train.get_checkpoint_state("./checkpoints")
        if ckpt is not None:
            weight_path = ckpt.model_checkpoint_path
            print('Restoring from {}...'.format(weight_path), end=' ')
            self.saver.restore(self.sess, weight_path)
            print('done')

    def load_memory(self, root):
        self.memory.load_memory(root)
