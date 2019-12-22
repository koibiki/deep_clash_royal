import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential


class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:
            self.downsample = lambda x: x

    def call(self, input, training=None):
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(input)
        output = layers.add([out, identity])
        output = self.relu(output)
        return output


class ResNet(keras.Model):
    def __init__(self, layer_dims,):
        super(ResNet, self).__init__()
        # preprocess
        self.stem = Sequential([
            layers.Conv2D(64, (3, 3), strides=(1, 1)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        ])
        # resblock
        self.layer1 = self._build_resblock(64, layer_dims[0], stride=2)
        self.layer2 = self._build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self._build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self._build_resblock(512, layer_dims[3], stride=2)

    def call(self, input):
        x = self.stem(input)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def _build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        # may down sample
        res_blocks.add(BasicBlock(filter_num, stride))
        # just down sample one time
        for pre in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))
        return res_blocks


def resnet18():
    return ResNet([2, 2, 2, 2])
