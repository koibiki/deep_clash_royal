import os
import os.path as osp
import time

import tensorflow as tf

from card_net.config import cfg


class BaseNet:

    def build_optimizer(self, cost, global_step):
        starter_learning_rate = cfg.TRAIN.LEARNING_RATE
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 10000, 0.9,
                                                   staircase=True)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate) \
                .minimize(loss=cost, global_step=global_step)
        return optimizer, learning_rate

    def build_sess(self):
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = 0.7
        sess_config.gpu_options.allow_growth = True
        sess = tf.Session(config=sess_config)
        return sess

    def build_saver(self):
        saver = tf.train.Saver(max_to_keep=3)
        os.makedirs('./logs', exist_ok=True)
        train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        model_name = 'net_{:s}.ckpt'.format(str(train_start_time))
        model_save_path = osp.join(cfg.PATH.MODEL_SAVE_DIR, model_name)
        return saver, model_save_path

    def load_model(self, sess, saver, weight_path):
        if weight_path is None:
            ckpt = tf.train.get_checkpoint_state(cfg.PATH.MODEL_SAVE_DIR)
            weight_path = ckpt.model_checkpoint_path
        print('Restoring from {}...'.format(weight_path), end=' ')
        saver.restore(sess, weight_path)
        stem = os.path.splitext(os.path.basename(weight_path))[-1]
        restore_iter = int(stem.split('-')[-1])
        sess.run(self.global_step.assign(restore_iter))
        print('done')
        return restore_iter

    def get_tf_data_amount(self, tf_data_paths):
        sample_count = sum(sum(1 for _ in tf.python_io.tf_record_iterator(tf_path)) for tf_path in tf_data_paths)
        return sample_count
