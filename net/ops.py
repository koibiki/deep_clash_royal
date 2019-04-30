import tensorflow as tf
import tensorflow.contrib.slim as slim

weight_decay = 1e-5

weights_init = tf.glorot_normal_initializer()
bias_init = tf.zeros_initializer()
l2_reg = slim.l2_regularizer(weight_decay)


def decode(inputs, cat_in, out_dim, k, t, s, is_train, n, name):
    with tf.name_scope(name), tf.variable_scope(name):
        net = slim.conv2d_transpose(inputs, out_dim, k, 2, scope="deconv")
        net = tf.concat([net, cat_in], axis=-1, name="concat")
    net = res_block(net, out_dim, k, t, s, is_train, name + '_0')
    for i in range(1, n):
        net = res_block(net, out_dim, k, t, 1, is_train, name + '_' + str(i), shortcut=True)
    return net


def conv2d_block(inputs, out_dim, k, s, is_train, name):
    with tf.name_scope(name), tf.variable_scope(name):
        net = slim.conv2d(inputs, out_dim, k, s, activation_fn=None,
                          weights_initializer=weights_init, biases_initializer=bias_init,
                          weights_regularizer=l2_reg, scope='conv2d')
        net = slim.batch_norm(net, scale=True, is_training=is_train, activation_fn=tf.nn.relu6, scope="bn")

        return net


def inverted_res_block(inputs, out_dim, kernel, t, s, is_train, n, name):
    net = res_block(inputs, out_dim, kernel, t, s, is_train, name + '_0')
    for i in range(1, n):
        net = res_block(net, out_dim, kernel, t, 1, is_train, name + '_' + str(i), shortcut=True)
    return net


def res_block(inputs, filters, kernel, t, strides, is_train, name, shortcut=False):
    with tf.name_scope(name), tf.variable_scope(name):
        bottleneck_dim = round(t * inputs.get_shape().as_list()[-1])
        net = slim.conv2d(inputs, bottleneck_dim, 1, 1, activation_fn=None,
                          weights_initializer=weights_init, biases_initializer=bias_init,
                          weights_regularizer=l2_reg, scope='pw')
        net = slim.batch_norm(net, scale=False, is_training=is_train, activation_fn=tf.nn.relu6, scope="pw_bn")

        net = slim.separable_conv2d(net, None, kernel, 1, stride=strides, activation_fn=None,
                                    weights_initializer=weights_init, biases_initializer=bias_init, scope="sp")
        net = slim.batch_norm(net, scale=False, is_training=is_train, activation_fn=tf.nn.relu6, scope="sp_bn")

        net = slim.conv2d(net, filters, 1, 1, activation_fn=None,
                          weights_initializer=weights_init, biases_initializer=bias_init,
                          weights_regularizer=l2_reg, scope='pw_linear')
        net = slim.batch_norm(net, scale=False, is_training=is_train, scope="pw_linear_bn")

        if shortcut:
            net = inputs + net
        return net
