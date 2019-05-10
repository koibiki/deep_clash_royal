import os.path as osp
import random
import time
from multiprocessing.pool import Pool

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets import resnet_v2

from brain.memory import Memory
from game.clash_royal import ClashRoyal
from net.mobilenet_v2 import build_mobilenetv2
from utils.img_utils import add_salt_and_pepper, add_gaussian_noise


class BaseBrain:
    BrainType = {"double": 0,
                 "runner": 1,
                 "trainer": 2}

    def __init__(self,
                 clash_royal,
                 brain_type,
                 lr=0.0001,
                 reward_decay=0.9,
                 memory_size=50000,
                 batch_size=16,
                 replace_target_iter=200, ):
        self.p = Pool(4)
        self.retry = 0
        self.n_card_actions = clash_royal.n_card_actions + 6 * 8
        self.img_shape = clash_royal.img_shape
        self.state_shape = clash_royal.state_shape

        self.brain_type = brain_type

        self.rate_of_winning = 0
        self.reward_sum = 0

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

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4
        sess_config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=sess_config)

        with self.sess.as_default():
            # consist of [target_net, evaluate_net]
            self._build_evaluate_and_target_net()

            vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            eval_var = [val for val in vars if 'eval_net' in val.name]
            target_var = [val for val in vars if 'target_net' in val.name]

            with tf.variable_scope('hard_replacement'):
                self.target_replace_op = [tf.assign(t, e) for t, e in zip(target_var, eval_var)]

            self.saver = tf.train.Saver(max_to_keep=3)
            train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            model_name = 'net_{:s}.ckpt'.format(str(train_start_time))
            self.model_save_path = osp.join("./checkpoints", model_name)

            self.sess.run(tf.global_variables_initializer())
            self.load_model()

            self.writer = tf.summary.FileWriter("./logs/" + clash_royal.name)
            self.writer.add_graph(self.sess.graph)
            self.merge_summary = tf.summary.merge_all()

    def _build_net(self, s_img, s_card_elixir):

        with tf.variable_scope('cnn'):
            # out = resnet_v2.resnet_v2_50(s_img)[1][scope + "/cnn/resnet_v2_50/block4"]
            out = build_mobilenetv2(s_img, True)

            loc = tf.layers.average_pooling2d(out, (2, 2), (2, 2), )

            cnn_out = tf.layers.conv2d_transpose(loc, 512, (3, 3), (2, 2), padding="same")

            cnn_concat = tf.concat([out, cnn_out], axis=-1)

            cnn_concat = tf.layers.dropout(cnn_concat, 0.5)

        with tf.variable_scope('executor'):
            s_card_elixir = tf.cast(s_card_elixir, dtype=tf.float32)
            dense = tf.layers.dense(s_card_elixir, 512, kernel_regularizer=self.reg, activation=tf.nn.relu)

            dense = tf.expand_dims(dense, axis=1)
            dense = tf.expand_dims(dense, axis=1)

            h_dense = tf.concat([dense for _ in range(out.shape[1].value)], axis=1)
            h_w_dense = tf.concat([h_dense for _ in range(out.shape[2].value)], axis=2)

            concat = tf.concat([cnn_concat, h_w_dense], axis=-1)

            with tf.variable_scope('value'):
                card_value = tf.layers.conv2d(concat, 1, (3, 3), (1, 1), padding="same")

            with tf.variable_scope('advantage'):
                card_advantage = tf.layers.conv2d(concat, 93, (3, 3), (1, 1), padding="same")

            with tf.variable_scope('q'):
                # Q = V(s) + A(s,a)
                card_logit = card_value + (card_advantage - tf.reduce_mean(card_advantage))
                card_logit = tf.layers.flatten(card_logit)

        return card_logit

    def _build_evaluate_and_target_net(self):

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

        self.q_card_target = tf.placeholder(tf.float32, [None, self.n_card_actions], name='Q_card_target')

        self.weights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')

        with tf.variable_scope('eval_net'):
            self.q_card_eval = self._build_net(self.s_img, self.s_card_elixir, )

        with tf.variable_scope('loss'):
            reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

            self.abs_errors = tf.reduce_sum(tf.abs(self.q_card_target - self.q_card_eval), axis=1)

            card_loss = tf.reduce_mean(self.weights * (tf.squared_difference(self.q_card_target, self.q_card_eval)))

            self.loss = card_loss + reg_loss
            if self.brain_type != self.BrainType["runner"]:
                tf.summary.scalar(name='loss', tensor=self.loss)
                tf.summary.scalar(name='card_loss', tensor=card_loss)

        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss, global_step=self._global_step)

        # ------------------ build target_net ------------------
        # input Next State
        self.s_img_ = \
            tf.placeholder(tf.float32, [None, self.img_shape[0], self.img_shape[1], self.img_shape[2]], name='image_')
        self.s_card_elixir_ = tf.placeholder(tf.float32, [None, self.state_shape], name='state_')

        with tf.variable_scope('target_net'):
            self.q_card_next = self._build_net(self.s_img_, self.s_card_elixir_, )

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
        if uniform <= 0.5:
            # forward feed the observation and get q value for every actions
            card_value = self.sess.run([self.q_card_eval],
                                       feed_dict={self.s_img: [observation[1]],
                                                  self.s_card_elixir: [observation[2]]})[0]
            index = np.argmax(card_value)
            print("dqn play:" + str(index % 93 != 0))
            if index % 93 != 0:
                action = [index % 93, index % (93 * 6) // 93, index // (93 * 6)]
            else:
                action = [0, 0, 0]
            print("dqn choose action:" + str(action) + "  " + str(observation[3]) + " " + str(observation[4]))
        elif uniform < 0.9:
            action = [0, 0, 0]
            print("random choose action:" + str(action) + "  " + str(observation[3]) + " " + str(observation[4]))
        else:
            card = random.choice(observation[3])
            x_loc = random.choice(range(6))
            y_loc = random.choice(range(7))
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
                action = 3 + 2 - self.memory.action_dict[item][1]
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
        card_action = [[card_type_action[i], x_action[i], y_action[i]] for i in range(len(card_type_action))]
        return imgs, states, card_action, reward, next_imgs, next_states

    def record_battle_result(self):
        with self.sess.as_default():
            self.sess.run(tf.assign(self._rate_of_winning, self.rate_of_winning))
            self.sess.run(tf.assign(self._reward, self.reward_sum))
            summary = self.sess.run([self.merge_summary], )

            self.writer.add_summary(summary=summary[0], global_step=self.learn_step_counter)

            self.learn_step_counter += 1

    def learn(self):
        with self.sess.as_default():
            # check to replace target parameters
            if self.learn_step_counter > 0 and self.learn_step_counter % self.replace_target_iter == 0:
                self.sess.run(self.target_replace_op)
                print('\ntarget_params_replaced {:d}\n'.format(self.learn_step_counter))

            start_time = time.time() * 1000
            tree_idx, batch_memory, weights = self.memory.sample(self.batch_size)

            if self.brain_type != self.BrainType["trainer"]:
                self.sess.run(tf.assign(self._rate_of_winning, self.rate_of_winning))
                self.sess.run(tf.assign(self._reward, self.reward_sum))

            imgs, states, card_action, reward, next_imgs, next_states \
                = self._prepare_data(batch_memory)

            q_card_next, q_card_eval, = self.sess.run(
                [self.q_card_next, self.q_card_eval],
                feed_dict={self.s_img_: next_imgs,
                           self.s_card_elixir_: next_states,
                           self.s_img: imgs,
                           self.s_card_elixir: states})

            q_card_target = q_card_eval.copy()

            for i in range(len(q_card_target)):
                if card_action[i][0] != 0:
                    action = card_action[i]
                    state_card = states[i][:92]
                    state_available = states[i][92:92 * 2]
                    has_card_index = np.where(state_card == 1)[0] + 1
                    available_card_index = np.where(state_available == 1)[0] + 1
                    available_index = np.intersect1d(has_card_index, available_card_index)

                    if action[0] in available_index:
                        index = action[0] + 93 * (action[2] * 6 + action[1])
                        q_card_target[:, index] = reward[i] + self.gamma * np.max(q_card_next[i])
                    else:
                        random_indices = [i for i in range(1, 93) if i not in available_index]

                        for random_index in random_indices:
                            for ii in range(6 * 8):
                                index = random_index + 93 * ii
                                q_card_target[:, index] = reward[i] + self.gamma * np.max(q_card_next[i])
                else:
                    for ii in range(6 * 8):
                        q_card_target[:, ii * 93] = reward[i] + self.gamma * np.max(q_card_next[i])

            _, abs_errors, loss, summary = self.sess.run(
                [self._train_op, self.abs_errors, self.loss, self.merge_summary],
                feed_dict={self.s_img: imgs,
                           self.s_card_elixir: states,
                           self.q_card_target: q_card_target,
                           self.weights: weights})

            self.memory.batch_update(tree_idx, abs_errors)  # update priority

            self.writer.add_summary(summary=summary, global_step=self.learn_step_counter)

            self.learn_step_counter += 1

            if self.learn_step_counter > 0 and self.learn_step_counter % 25 == 0:
                print('\nSave weights {:d}\n'.format(self.learn_step_counter))
                self.saver.save(self.sess, self.model_save_path, global_step=self.learn_step_counter)
            print("Train spent {:d}  {:f}".format(self.learn_step_counter, time.time() * 1000 - start_time))

    def load_model(self):
        ckpt = tf.train.get_checkpoint_state("./checkpoints")
        if ckpt is not None:
            weight_path = ckpt.model_checkpoint_path
            print('Restoring from {}...'.format(weight_path), end=' ')
            self.saver.restore(self.sess, weight_path)
            print('done')

    def load_memory(self, root):
        self.memory.load_memory(root)


if __name__ == '__main__':
    royal = ClashRoyal("../", "id")
    base_brain = BaseBrain(royal, 0)
    base_brain.load_memory("../../vysor")
