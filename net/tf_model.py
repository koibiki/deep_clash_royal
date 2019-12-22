import tensorflow as tf
from tensorflow.keras import layers, Sequential

from game.parse_result import calu_available_card
from net.resnet import resnet18
import numpy as np


def gru(units):
    return tf.keras.layers.GRU(units,
                               return_sequences=True,
                               return_state=True,
                               recurrent_initializer='glorot_uniform')


class BattleFieldFeature(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.resnet = resnet18()
        self.downsample = layers.Conv2D(8, (3, 3), strides=1, padding='same', activation='relu')
        self.flatten = layers.Flatten()

        self.dp = layers.Dropout(0.5)

    def call(self, x, training=False):
        x = self.resnet(x)
        x = self.downsample(x)

        if training:
            x = self.dp(x)

        x = self.flatten(x)
        return x


class PpoNet(tf.keras.Model):
    """
    enc_units: encoder 隐元数量
    batch_sz: batch size
    """

    def __init__(self, ):
        super(PpoNet, self).__init__()
        self.hidden_size = 512
        self.embed_size = 64

        self.card_amount = 94

        # card indices
        self.card_indices = tf.convert_to_tensor(np.array([i for i in range(self.card_amount)]))

        # img feature
        self.battle_field_feature = BattleFieldFeature()

        # card index embed
        self.card_embed = layers.Embedding(self.card_amount, self.embed_size)
        self.card_dense = Sequential([layers.Dense(256, activation='relu'),
                                      layers.Dense(64, activation='relu')])
        self.card_pooling = layers.MaxPool2D(pool_size=(4, 1), strides=(4, 1), padding='valid')

        self.dense = layers.Dense(self.hidden_size, activation='relu')

        # actor
        self.actor_gru = gru(self.hidden_size)
        self.use_card = layers.Dense(2)
        self.intent = layers.Dense(self.embed_size)
        self.pos_x = layers.Dense(6 * self.embed_size)
        self.pos_y = layers.Dense(5 * self.embed_size)

        # critic
        self.critic_gru = gru(self.hidden_size)
        self.value = layers.Dense(1)

        self.softmax = layers.Softmax()

    def call(self, img, env_state, card_type, card_property, actor_hidden=None, critic_hidden=None, training=False):

        card_embed = self.card_embed(card_type)

        card_embed0 = card_embed[:, 0]
        card_state0 = tf.expand_dims(self.card_dense(tf.concat([card_embed0, card_property[:, :2]], axis=-1)), axis=1)

        card_embed1 = card_embed[:, 1]
        card_state1 = tf.expand_dims(self.card_dense(tf.concat([card_embed1, card_property[:, 2:4]], axis=1)), axis=1)

        card_embed2 = card_embed[:, 2]
        card_state2 = tf.expand_dims(self.card_dense(tf.concat([card_embed2, card_property[:, 4:6]], axis=1)), axis=1)

        card_embed3 = card_embed[:, 3]
        card_state3 = tf.expand_dims(self.card_dense(tf.concat([card_embed3, card_property[:, 6:]], axis=1)), axis=1)

        card_cat = tf.concat([card_state0, card_state1, card_state2, card_state3], axis=1)
        card_cat = tf.expand_dims(card_cat, axis=-1)
        card_state = tf.reshape(self.card_pooling(card_cat), (-1, self.embed_size))

        battle_field_feature = self.battle_field_feature(img, training)
        all_feature = tf.concat([battle_field_feature, env_state, card_state], axis=1)

        feature = self.dense(all_feature)
        feature = tf.expand_dims(feature, 0)

        result = {}
        # actor run
        if actor_hidden is not None:
            actor_output, actor_hidden = self.actor_gru(feature, actor_hidden)

            actor_output = tf.squeeze(actor_output, 0)

            action_intent = self.intent(actor_output)

            card_embed = self.card_embed(self.card_indices)
            card_prob = self.softmax(tf.matmul(action_intent, card_embed, transpose_b=True))

            available_card = tf.convert_to_tensor(calu_available_card(card_type.numpy(), card_property.numpy()),
                                                  dtype=tf.float32)
            card_prob = tf.multiply(card_prob, available_card)

            card_choice_index = tf.argmax(card_prob, axis=-1)
            card_choice_index = tf.one_hot(card_choice_index, self.card_amount)
            choice_card = tf.matmul(card_choice_index, card_embed)

            pos_x_vector = tf.reshape(self.pos_x(actor_output), (-1, 6, self.embed_size))
            pos_y_vector = tf.reshape(self.pos_y(actor_output), (-1, 5, self.embed_size))

            pos_x_vector_transpose = tf.transpose(pos_x_vector, (1, 0, 2))
            pos_y_vector_transpose = tf.transpose(pos_y_vector, (1, 0, 2))

            pos_x_choice = tf.transpose(pos_x_vector_transpose * choice_card, (1, 0, 2))
            pos_y_choice = tf.transpose(pos_y_vector_transpose * choice_card, (1, 0, 2))

            pos_x_prob = self.softmax(tf.reduce_sum(pos_x_choice, axis=-1))
            pos_y_prob = self.softmax(tf.reduce_sum(pos_y_choice, axis=-1))

            result["actor"] = [card_prob.numpy(), pos_x_prob.numpy(), pos_y_prob.numpy(), choice_card.numpy(), actor_hidden]

        # critic run
        if critic_hidden is not None:
            critic_output, critic_hidden = self.critic_gru(feature, critic_hidden)
            critic_v = self.value(critic_output)
            result["critic"] = [critic_v, critic_hidden]

        return result

    def initialize_hidden_state(self):
        return tf.zeros((1, self.hidden_size))
