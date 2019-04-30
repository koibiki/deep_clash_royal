import tensorflow as tf
from config import cfg
import time
import os.path as osp
import os


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

    def build_train_dataset(self, sess, tf_record_path, parse_func):
        return self.build_tf_dataset(sess, tf_record_path, parse_func, cfg.TRAIN.BATCH_SIZE)

    def build_valid_dataset(self, sess, tf_record_path, parse_func):
        return self.build_tf_dataset(sess, tf_record_path, parse_func, cfg.VALID.BATCH_SIZE)

    def build_tf_dataset(self, sess, tf_record_path, parse_func, batch_size):
        dataset = tf.data.TFRecordDataset(tf_record_path)
        dataset = dataset.map(parse_func)
        dataset = dataset.shuffle(buffer_size=100)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_initializable_iterator()
        next_example = iterator.get_next()
        sess.run(iterator.initializer)
        return next_example, iterator

    def get_tf_data_amount(self, tf_data_paths):
        sample_count = sum(sum(1 for _ in tf.python_io.tf_record_iterator(tf_path)) for tf_path in tf_data_paths)
        return sample_count

    def get_center_loss(self, features, labels, alpha, num_classes):
        """获取center loss及center的更新op

        Arguments:
            features: Tensor,表征样本特征,一般使用某个fc层的输出,shape应该为[batch_size, feature_length].
            labels: Tensor,表征样本label,非one-hot编码,shape应为[batch_size].
            alpha: 0-1之间的数字,控制样本类别中心的学习率,细节参考原文.
            num_classes: 整数,表明总共有多少个类别,网络分类输出有多少个神经元这里就取多少.

        Return：
            loss: Tensor,可与softmax loss相加作为总的loss进行优化.
            centers: Tensor,存储样本中心值的Tensor，仅查看样本中心存储的具体数值时有用.
            centers_update_op: op,用于更新样本中心的op，在训练时需要同时运行该op，否则样本中心不会更新
        """
        # 获取特征的维数，例如256维
        len_features = features.get_shape()[1]
        # 建立一个Variable,shape为[num_classes, len_features]，用于存储整个网络的样本中心，
        # 设置trainable=False是因为样本中心不是由梯度进行更新的
        centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0), trainable=False)
        # 将label展开为一维的，输入如果已经是一维的，则该动作其实无必要
        labels = tf.reshape(labels, [-1])

        # 根据样本label,获取mini-batch中每一个样本对应的中心值
        centers_batch = tf.gather(centers, labels)
        # 计算loss
        loss = tf.nn.l2_loss(features - centers_batch)

        # 当前mini-batch的特征值与它们对应的中心值之间的差
        diff = centers_batch - features

        # 获取mini-batch中同一类别样本出现的次数,了解原理请参考原文公式(4)
        unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])

        diff = diff / tf.cast((1 + appear_times), tf.float32)
        diff = alpha * diff

        centers_update_op = tf.scatter_sub(centers, labels, diff)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, centers_update_op)

        return loss, centers, centers_update_op

    def contrastive_loss(self, y, d):
        margin = 1.5
        y = tf.cast(y, dtype=tf.float32)
        tmp = y * tf.square(d)
        tmp2 = (1 - y) * tf.square(tf.maximum(margin - d, 0))
        return tf.reduce_mean(tmp + tmp2)
