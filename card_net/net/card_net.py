import tensorflow as tf


class CrnnNet(object):

    def __init__(self):
        self.feature_net = CnnFeature()

    def __call__(self, inputs, mode, batch_size, num_classes):
        with tf.variable_scope("card_net"):
            cnn_feature = self.feature_net(inputs, mode == tf.estimator.ModeKeys.TRAIN)

            cnn_feature = tf.layers.flatten(cnn_feature)

            type_logits = tf.layers.dense(cnn_feature, num_classes, activation=None, name="type_logits")
            available_logits = tf.layers.dense(cnn_feature, num_classes, activation=None, name="available_logits")

            type_pred = tf.nn.softmax(type_logits, name="type_pred")
            available_pred = tf.nn.softmax(available_logits, name="available_pred")

        return type_logits, available_logits, type_pred, available_pred


class CnnFeature(tf.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, inputs, training, **kwargs):
        with tf.variable_scope("cnn"):
            conv1 = tf.layers.conv2d(
                inputs=inputs, filters=32, kernel_size=(3, 3), padding="same",
                activation=tf.nn.relu, name='conv1')

            pool1 = tf.layers.max_pooling2d(
                inputs=conv1, pool_size=[2, 2], strides=2, name='pool1')

            conv2 = tf.layers.conv2d(
                inputs=pool1, filters=64, kernel_size=(3, 3), padding="same",
                activation=tf.nn.relu, name='conv2')

            pool2 = tf.layers.max_pooling2d(
                inputs=conv2, pool_size=[2, 2], strides=2, padding="same", name='pool2')

            conv3 = tf.layers.conv2d(
                inputs=pool2, filters=128, kernel_size=(3, 3), padding="same",
                activation=tf.nn.relu, name='conv3')

            pool3 = tf.layers.max_pooling2d(
                inputs=conv3, pool_size=[2, 2], strides=[2, 2], padding="same", name='pool3')

        return pool3
