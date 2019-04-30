from net.base_net import BaseNet
from net.ops import *
from config import cfg
import os
import os.path as osp
import time
import numpy as np
from tqdm import *
import math


class MobileNetV2(BaseNet):

    def __init__(self, num_class, is_train=True, weight_path=None):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self._batch_size = cfg.TRAIN.BATCH_SIZE if is_train else cfg.TEST.BATCH_SIZE
        self._input_shape = cfg.TRAIN.INPUT_SHAPE if is_train else cfg.TEST.INPUT_SHAPE
        self.weight_path = weight_path
        self._input = tf.placeholder(dtype=tf.float32, shape=self._input_shape, name='input')
        self._action = tf.placeholder(dtype=tf.int32, shape=self._batch_size, name='action')
        self._action_y = tf.placeholder(dtype=tf.int32, shape=self._batch_size, name='action_x')
        self._action_x = tf.placeholder(dtype=tf.int32, shape=self._batch_size, name='action_y')

        self.sess = self.build_sess()
        with self.sess.as_default():
            self.out = self.build_mobilenetv2(self._input, num_class, is_train, 0.5 if is_train else 1.0)
            self.global_step = tf.Variable(0, trainable=False)
            self.saver, self.model_save_path = self.build_saver()
            if not is_train:
                self.load_model(self.sess, self.saver, weight_path)

    def build_mobilenetv2(self, _inputs, is_train, keep_prob):
        with tf.variable_scope('mobilenetv2'):
            conv1 = conv2d_block(_inputs, 32, 3, 2, is_train=is_train, name='conv1_1')

            res2 = inverted_res_block(conv1, 16, (3, 3), t=1, s=1, is_train=is_train, n=1, name='res2')
            res3 = inverted_res_block(res2, 24, (3, 3), t=6, s=2, is_train=is_train, n=2, name='res3')
            res4 = inverted_res_block(res3, 32, (3, 3), t=6, s=2, is_train=is_train, n=3, name='res4')
            res5 = inverted_res_block(res4, 64, (3, 3), t=6, s=2, is_train=is_train, n=4, name='res5')
            res6 = inverted_res_block(res5, 96, (3, 3), t=6, s=1, is_train=is_train, n=3, name='res6')
            res7 = inverted_res_block(res6, 160, (3, 3), t=6, s=2, is_train=is_train, n=3, name='res7')
            res8 = inverted_res_block(res7, 320, (3, 3), t=6, s=1, is_train=is_train, n=1, name='res8')

            conv9 = conv2d_block(res8, 1280, 1, 1, is_train, name='conv9_1')
            global_pool = slim.avg_pool2d(conv9, [7, 7])
            dp = slim.dropout(global_pool, keep_prob=keep_prob, scope='dp')
            action_logits = slim.conv2d(dp, cfg.ACTION_NUM, 1, 1, activation_fn=None, scope='action_logits')
            action_logits = slim.conv2d(dp, cfg.ACTION_NUM, 1, 1, activation_fn=None, scope='action_x_logits')
            action_logits = slim.conv2d(dp, cfg.ACTION_NUM, 1, 1, activation_fn=None, scope='action_y_logits')

        out = {"conv1": conv1, "res2": res2, "res3": res3, "res4": res4, "res5": res5, "res6": res6, "res7": res7,
               "res8": res8, "conv9": conv9, "global_pool": global_pool, "logits": logits, "pred": pred}

        return out

    def build_loss(self, out, y):
        cross_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=out["logits"]))

        l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        total_loss = cross_loss + l2_loss

        return total_loss, cross_loss, l2_loss

    def build_evaluator(self, out, y):
        correct_pred = tf.equal(tf.argmax(out["pred"], 1), tf.cast(y, tf.int64))
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return acc

    def build_summary(self, total_cost, cost, learning_rate):
        os.makedirs(cfg.PATH.TBOARD_SAVE_DIR, exist_ok=True)
        tf.summary.scalar(name='Total_Cost', tensor=tf.reduce_sum(total_cost))
        tf.summary.scalar(name='Cost', tensor=tf.reduce_sum(cost))
        tf.summary.scalar(name='Learning_Rate', tensor=learning_rate)
        merge_summary_op = tf.summary.merge_all()
        return merge_summary_op

    def train(self, parse_func, train_tf_record_path, valid_tf_record_path=None):
        with self.sess.as_default():
            total_loss, cross_loss, reg_loss = self.build_loss(self.out, self._y)

            optimizer, lr = self.build_optimizer(total_loss, self.global_step)

            train_epochs = cfg.TRAIN.EPOCHS

            acc_op = self.build_evaluator(self.out, self._y)

            summary_op = self.build_summary(total_loss, cross_loss, lr)

            train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            log_dir = osp.join(cfg.PATH.TBOARD_SAVE_DIR, train_start_time)
            os.mkdir(log_dir)

            summary_writer = tf.summary.FileWriter(log_dir)
            summary_writer.add_graph(self.sess.graph)

            self.sess.run(tf.global_variables_initializer())

            train_example, train_iterator = self.build_train_dataset(self.sess, train_tf_record_path, parse_func)
            train_data_amount = self.get_tf_data_amount(train_tf_record_path)
            train_num_iter = int(math.ceil(train_data_amount / cfg.TRAIN.BATCH_SIZE))

            if valid_tf_record_path is not None:
                valid_example, valid_iterator = self.build_valid_dataset(self.sess, valid_tf_record_path, parse_func)
                valid_data_amount = self.get_tf_data_amount(valid_tf_record_path)

            for epoch in range(train_epochs):
                all_acc = 0
                all_t_c = 0
                all_cross_c = 0
                pbar = tqdm(range(train_num_iter))
                for _step in pbar:
                    input_data, input_label = self.sess.run(train_example)

                    summary, _, t_c, c_c, l1_c, _acc, _lr, pred_prob = self.sess.run(
                        [summary_op, optimizer, total_loss, cross_loss, reg_loss, acc_op, lr, self.out['pred']],
                        feed_dict={self._input: input_data,
                                   self._y: input_label})

                    all_t_c += np.sum(t_c)
                    all_cross_c += np.sum(c_c)
                    all_acc += np.sum(_acc)

                    sesc_str = 'Epoch: {:4d}/{:4d} cost= {:9f} cross_c={:9f} l1_c={:9f} lr={:9f} acc={:9f}'.format(
                        epoch + 1, train_epochs, all_t_c / (_step + 1), all_cross_c / (_step + 1), np.sum(l1_c), _lr,
                        all_acc / (_step + 1))

                    pbar.set_description(sesc_str)

                    _global_step = epoch * train_num_iter + _step

                    if _global_step % cfg.TRAIN.SAVE_MODEL_STEP == 0:
                        tf.train.write_graph(self.sess.graph_def, 'checkpoints', 'net_txt.pb', as_text=True)
                        self.saver.save(sess=self.sess, save_path=self.model_save_path, global_step=_global_step)

                    summary_writer.add_summary(summary=summary, global_step=_global_step)

            print('FINISHED TRAINING.')
