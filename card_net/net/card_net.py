import os
import os.path as osp
import time
import tensorflow as tf
from card_net.config import cfg
from card_net.dataset.data_provider import DataProvider
from card_net.net.base_net import BaseNet


class CardNet(BaseNet):

    def __init__(self, num_class, is_train=True, weight_path=None):
        self._batch_size = cfg.TRAIN.BATCH_SIZE
        self.weight_path = weight_path
        self._input = tf.placeholder(dtype=tf.float32, shape=(None, 64, 64, 3), name='input')
        self._type = tf.placeholder(dtype=tf.int32, shape=(None,), name='type')
        self._avaiable = tf.placeholder(dtype=tf.int32, shape=(None,), name='avaiable')
        self.sess = self.build_sess()
        with self.sess.as_default():
            self.out = self.build_net(self._input, num_class)
            self.global_step = tf.Variable(0, trainable=False)
            self.saver, self.model_save_path = self.build_saver()

    def build_net(self, _inputs, num_classes):
        with tf.variable_scope("card_net"):
            conv1 = tf.layers.conv2d(
                inputs=_inputs, filters=32, kernel_size=(3, 3), padding="same", activation=tf.nn.relu, name='conv1')
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name='pool1')

            conv2 = tf.layers.conv2d(
                inputs=pool1, filters=64, kernel_size=(3, 3), padding="same", activation=tf.nn.relu, name='conv2')
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, padding="same", name='pool2')

            conv3 = tf.layers.conv2d(
                inputs=pool2, filters=128, kernel_size=(3, 3), padding="same", activation=tf.nn.relu, name='conv3')
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=[2, 2], padding="same",
                                            name='pool3')

            cnn_feature = tf.layers.flatten(pool3)

            type_logits = tf.layers.dense(cnn_feature, num_classes, activation=None, name="type_logits")
            available_logits = tf.layers.dense(cnn_feature, 2, activation=None, name="available_logits")

            type_pred = tf.nn.softmax(type_logits, name="type_pred")
            available_pred = tf.nn.softmax(available_logits, name="available_pred")
        out = {"type_logits": type_logits, "available_logits": available_logits,
               "type_pred": type_pred, "available_pred": available_pred}
        return out

    def build_loss(self, out, type, available):
        type_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=type, logits=out["type_logits"]))
        available_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=available, logits=out["available_logits"]))

        total_loss = type_loss + available_loss

        return total_loss

    def build_evaluator(self, out, y):
        correct_pred = tf.equal(tf.argmax(out["pred"], 1), tf.cast(y, tf.int64))
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return acc

    def build_summary(self, total_cost, learning_rate):
        os.makedirs(cfg.PATH.MODEL_SAVE_DIR, exist_ok=True)
        tf.summary.scalar(name='Total_Cost', tensor=tf.reduce_sum(total_cost))
        tf.summary.scalar(name='Learning_Rate', tensor=learning_rate)
        merge_summary_op = tf.summary.merge_all()
        return merge_summary_op

    def train(self, ):
        with self.sess.as_default():
            total_loss = self.build_loss(self.out, self._type, self._avaiable)

            optimizer, lr = self.build_optimizer(total_loss, self.global_step)

            summary_op = self.build_summary(total_loss, lr)

            train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            log_dir = osp.join(cfg.PATH.MODEL_SAVE_DIR, train_start_time)
            os.mkdir(log_dir)

            summary_writer = tf.summary.FileWriter(log_dir)
            summary_writer.add_graph(self.sess.graph)

            self.sess.run(tf.global_variables_initializer())

            dataset = DataProvider()
            for _step in range(100000):
                input_data, input_type, input_available = dataset.generate_data()

                summary, _, t_c, _lr, pred_prob, available_prob = self.sess.run(
                    [summary_op, optimizer, total_loss, lr, self.out['type_pred'], self.out['available_pred']],
                    feed_dict={self._input: input_data,
                               self._type: input_type,
                               self._avaiable: input_available})

                print('_step:{} cost= {:9f} '.format(_step, t_c))

                if _step % cfg.TRAIN.SAVE_MODEL_STEP == 0:
                    tf.train.write_graph(self.sess.graph_def, 'checkpoints', 'net_txt.pb', as_text=True)
                    self.saver.save(sess=self.sess, save_path=self.model_save_path, global_step=_step)

        print('FINISHED TRAINING.')
