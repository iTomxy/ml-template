import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import layers as L
import scipy.io as sio
import numpy as np


class CNN_F(K.Model):
    """CNN-F / VGG-F"""

    def __init__(self, weight_file):
        super(CNN_F, self).__init__()
        layers = sio.loadmat(weight_file)["net"][0][0][0][0]
        self.first = K.Sequential([
            make_conv(layers[0], True),
            L.ReLU(),
            make_lrn(layers[2]),
            make_pool(layers[3])
        ], name="block_1")
        self.second = K.Sequential([
            make_conv(layers[4]),
            L.ReLU(),
            make_lrn(layers[6]),
            make_pool(layers[7])
        ], name="block_2")
        self.third = K.Sequential([
            make_conv(layers[8]),
            L.ReLU()
        ], name="block_3")
        self.fourth = K.Sequential([
            make_conv(layers[10]),
            L.ReLU()
        ], name="block_4")
        self.fifth = K.Sequential([
            make_conv(layers[12]),
            L.ReLU(),
            make_pool(layers[14])
        ], name="block_5")
        self.sixth = K.Sequential([
            make_conv(layers[15]),
            L.ReLU()
        ], name="block_6")
        self.drop6 = L.Dropout(0.5)
        self.seventh = K.Sequential([
            make_conv(layers[17]),
            L.ReLU()
        ], name="block_7")
        self.drop7 = L.Dropout(0.5)

    def call(self, x, training=False):
        x = self.first(x)
        x = self.second(x)
        x = self.third(x)
        x = self.fourth(x)
        x = self.fifth(x)
        x = self.sixth(x)
        x = self.drop6(x, training=training)
        x = self.seventh(x)
        x = self.drop7(x, training=training)
        return tf.reshape(x, [-1, x.shape.as_list()[-1]])


def make_conv(layer, first=False):
    """if first layer, provide `input_shape=(h,w,c)`"""
    layer = layer[0][0]
    # print("name:", layer[0][0])
    # print("type:", layer[1][0])
    k, b = layer[2][0]
    # b = b.flatten()
    # print("kernel:", k.shape, ", bias:", b.shape)
    shape = layer[3][0]
    # print("shape:", shape)
    pad = layer[4][0]
    # print("pad:", pad)
    stride = layer[5][0]
    # print("stride:", stride)

    if first:
        conv = L.Conv2D(shape[3], shape[:2], strides=stride, padding="valid",
                        kernel_initializer=tf.initializers.constant(k),
                        bias_initializer=tf.initializers.constant(b),
                        input_shape=[224, 224, 3])
    else:
        conv = L.Conv2D(shape[3], shape[:2], strides=stride, padding="valid",
                        kernel_initializer=tf.initializers.constant(k),
                        bias_initializer=tf.initializers.constant(b))


    if np.sum(pad) > 0:
        padding = L.Lambda(lambda x: tf.pad(
            x, [[0, 0], [pad[0], pad[1]], [pad[2], pad[3]], [0, 0]], "CONSTANT"))
        conv = K.Sequential([padding, conv], name=layer[0][0])

    return conv


def make_lrn(layer):
    layer = layer[0][0]
    # print("name:", layer[0][0])
    # print("type:", layer[1][0])
    param = layer[2][0]
    # print("local_size/depth_radius:", param[0])
    # print("bias:", param[1])
    # print("alpha:", param[2])
    # print("beta:", param[3])
    return L.Lambda(lambda x: tf.nn.local_response_normalization(
        x, depth_radius=param[0], bias=param[1], alpha=param[2], beta=param[3]))


def make_pool(layer):
    layer=layer[0][0]
    # print("name:", layer[0][0])
    # print("type:", layer[1][0])
    # print("pool type:", layer[2])
    k_size=layer[3][0]
    stride=layer[4][0]
    # print("stride:", stride)
    pad=layer[5][0]
    # print("pad:", pad)

    pool=L.MaxPool2D(pool_size=k_size, strides=stride, padding="valid")
    if np.sum(pad) > 0:
        padding=L.Lambda(lambda x: tf.pad(
            x, [[0, 0], [pad[0], pad[1]], [pad[2], pad[3]], [0, 0]], "CONSTANT"))
        pool=K.Sequential([padding, pool], name=layer[0][0])

    return pool
