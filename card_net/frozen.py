import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np
import cv2


def freeze_graph(input_checkpoint, output_graph, output_node_names):
    '''
    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :return:
    '''
    # checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    # input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径

    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=sess.graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点


def freeze_graph_test(pb_path, image_path):
    '''
    :param pb_path:pb文件的路径
    :param image_path:测试图片的路径
    :return:
    '''
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # 定义输入的张量名称,对应网络结构的输入张量
            input_image_tensor = sess.graph.get_tensor_by_name("input:0")

            # 定义输出的张量名称
            type_pred = sess.graph.get_tensor_by_name("card_net/type_pred:0")
            available_pred = sess.graph.get_tensor_by_name("card_net/available_pred:0")

            im = cv2.imread(image_path)
            im = cv2.resize(im, (64, 64))
            im = im[np.newaxis, :]

            out = sess.run([type_pred, available_pred], feed_dict={input_image_tensor: im})
            print("out:{}".format(out))



# output_node_names = "card_net/type_pred,card_net/available_pred"
#
# input_checkpoint = './checkpoints/net_2020-01-01-17-33-07.ckpt-11300'
# out_pb_path = "./checkpoints/frozen_model.pb"
# freeze_graph(input_checkpoint, out_pb_path, output_node_names)

img_path = "../0.jpg"
pb_path = "./checkpoints/frozen_model.pb"
freeze_graph_test(pb_path, img_path)
