import tensorflow as tf
import os
from card_net.dataset.data_provider import DataProvider
from card_net.net.card_net import CrnnNet
from card_net.config import cfg

print(tf.__version__)

tf.logging.set_verbosity(tf.logging.INFO)

provider = DataProvider()
train_input_fn = provider.generate_train_input_fn()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_feature_columns():
    feature_columns = {
        'images': tf.feature_column.numeric_column('images', (64, 64, 3)),
    }
    return feature_columns


def model_fn(features, labels, mode, params):
    type_tensor = labels[0]
    available_tensor = labels[1]

    feature_columns = list(get_feature_columns().values())

    images = tf.feature_column.input_layer(
        features=features, feature_columns=feature_columns)

    images = tf.reshape(images, shape=(-1, 64, 64, 3))

    crnn = CrnnNet()
    type_logits, available_logits, type_pred, available_pred = crnn(images, mode, cfg.TRAIN.BATCH_SIZE, cfg.NUM_CLASSES)

    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
        global_step = tf.train.get_or_create_global_step()

        type_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=type_logits,
                                                                                  labels=type_tensor))
        available_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=available_logits,
                                                                                       labels=available_tensor))

        type_loss = tf.identity(type_loss, name='type_loss')
        available_loss = tf.identity(available_loss, name='available_loss')

        tf.summary.scalar('type_loss', type_loss)
        tf.summary.scalar('available_loss', available_loss)

        loss = type_loss + available_loss

        if mode == tf.estimator.ModeKeys.TRAIN:
            start_learning_rate = cfg.TRAIN.LEARNING_RATE
            learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 10000, 0.9,
                                                       staircase=True)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                train_op = optimizer.minimize(loss=loss, global_step=global_step)
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'type': type_pred,
            'available': available_pred,
        }
        export_outputs = {
            'predictions': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions, export_outputs=export_outputs)


tensors_to_log = {"type_loss": "type_loss", "available_loss": "available_loss"}

logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=10, at_end=True)

session_config = tf.ConfigProto()
session_config.gpu_options.per_process_gpu_memory_fraction = 0.9
session_config.gpu_options.allow_growth = True

run_config = tf.estimator.RunConfig(
    save_checkpoints_steps=100,
    tf_random_seed=512,
    model_dir="./checkpoints",
    keep_checkpoint_max=3,
    log_step_count_steps=10,
    session_config=session_config
)

estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config)

estimator.train(input_fn=train_input_fn, steps=2000, hooks=[logging_hook])
