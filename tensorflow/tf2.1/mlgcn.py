import pickle
import numpy as np
import scipy.special as scsp
import tensorflow as tf
import tensorflow.keras as K
from args import *
from wheel import *


def get_A(adj_file, tau=0.4):
    """tau: threashold of Eq (7)
    https://github.com/Megvii-Nanjing/ML-GCN/blob/master/util.py#L291
    """
    with open(adj_file, 'rb') as f:
        data = pickle.load(f)
    M = data["adj"]  # Eq (6)
    N = data["nums"][:, np.newaxis]  # #samples per class
    n_class = N.shape[0]
    P = M / N  # Eq (6)
    A = P  # Eq (7)
    A[P < tau] = 0
    A[P >= tau] = 1
    A *= 0.25 / (A.sum(0, keepdims=True) + 1e-6)
    A += np.identity(n_class, np.int)
    return A


class GCN_ResNet(K.Model):
    """https://github.com/Megvii-Nanjing/ML-GCN/blob/master/models.py#L43"""

    def __init__(self, A):
        """input: image, label w2v
        x -> ResNet-101 -> [None, 2048, 14, 14] -> GMP -> [None, 2048] -> f
        y -> GC + leaky relu -> [c, 1024] -> GC -> [c, 2048] -> W.t
        """
        super(GCN_ResNet, self).__init__()
        resnet101 = K.applications.ResNet101(
            include_top=False, weights='imagenet', input_shape=(448, 448, 3))
        self.base_net = resnet101
        self.gmp = K.layers.MaxPool2D(14)
        self.flat = K.layers.Flatten()

        self.gc1 = GraphConv(1024, K.layers.LeakyReLU(0.2), False)
        self.gc2 = GraphConv(2048, bias=False)

        _row_sum = tf.reduce_sum(A, 1)  # [n]
        _rs_rsqrt = tf.math.rsqrt(_row_sum)  # 1 / sqrt(x)
        self.A = tf.convert_to_tensor(
            _rs_rsqrt[:, None] * A * _rs_rsqrt[None, :])  # A^ = D^{-1/2} A D^{-1/2}

    def call(self, x, y):
        x = self.base_net(x)
        x = self.gmp(x)
        F = self.flat(x)

        y = self.gc1(y, self.A)
        y = self.gc2(y, self.A)
        W = tf.transpose(y)  # [d, c]

        return tf.matmul(F, W)


class LR_MLGCN(K.optimizers.schedules.LearningRateSchedule):
    """decay by 10 every 30 epochs
    - https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/optimizer_v2/learning_rate_schedule.py#L410-L511
    - https://github.com/Megvii-Nanjing/ML-GCN/blob/master/engine.py#L313
    """

    def __init__(self, lr, decay_step, decay_rate=0.1):
        """decay_step = args.decay_step * (N // args.batch_size)"""
        super(LR_MLGCN, self).__init__()
        self.lr = lr
        self.decay_step = decay_step
        self.decay_rate = decay_rate

    def __call__(self, step):
        """optimize once, step one"""
        if step % self.decay_step == 0:
            self.lr *= self.decay_rate
        return self.lr

    def get_config(self):
        return {
            "initial_learning_rate": self.lr,
            "decay_step": self.decay_step,
            "decay_rate": self.decay_rate
        }
