from net.ops import *


def build_mobilenetv2(_inputs, is_train):
    with tf.variable_scope('mobilenetv2'):
        conv1 = conv2d_block(_inputs, 32, 3, 2, is_train=is_train, name='conv1_1')
        res2 = inverted_res_block(conv1, 16, (3, 3), t=1, s=1, is_train=is_train, n=1, name='res2')
        res3 = inverted_res_block(res2, 24, (3, 3), t=2, s=2, is_train=is_train, n=2, name='res3')
        res4 = inverted_res_block(res3, 32, (3, 3), t=2, s=2, is_train=is_train, n=3, name='res4')
        res5 = inverted_res_block(res4, 64, (3, 3), t=2, s=2, is_train=is_train, n=4, name='res5')
        res6 = inverted_res_block(res5, 96, (3, 3), t=2, s=1, is_train=is_train, n=3, name='res6')
        res7 = inverted_res_block(res6, 128, (3, 3), t=2, s=2, is_train=is_train, n=3, name='res7')
        res8 = inverted_res_block(res7, 256, (3, 3), t=2, s=1, is_train=is_train, n=1, name='res8')
        conv9 = conv2d_block(res8, 512, 1, 1, is_train, name='conv9_1')
    return conv9
