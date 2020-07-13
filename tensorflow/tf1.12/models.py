import numpy as np
import tensorflow as tf
import cnnf
import losses
from args import *
from wheel import *


class Some_Model:
    """descriptions"""

    def __init__(self):
        self.in_images = tf.placeholder(
            "float32", [None, 224, 224, 3], name="in_images")
        self.in_labels = tf.placeholder(
            "float32", [None, args.n_class], name="in_labels")
        self.training = tf.placeholder("bool", [], name="training")

        self._build()
        self._add_loss()
        self._add_optim()

    def _build(self):
        fc7 = cnnf.CNN_F(self.in_images, args.cnnf_weight, self.training, top=False)
        fc7 = tf.layers.average_pooling2d(fc7, 6, 6)
        self.fea = tf.reshape(fc7, [-1, fc7.shape.as_list()[-1]])
        x = tf.layers.dense(self.fea, 2048, tf.nn.relu, tf.initializers.he_normal(), name="itom1")
        x = tf.layers.dropout(x, rate=0.5, training=self.training)
        x = tf.layers.dense(x, 1024, tf.math.tanh, tf.initializers.glorot_normal(), name="itom2")
        self.logit = tf.layers.dense(x, args.n_class, None, False,
            kernel_initializer=tf.truncated_normal(stddev=0.01),
            bias_initializer=tf.constant_initializer(0.1),
            name="itom3")

    def _add_loss(self):
        self.loss_xent = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.in_labels, logits=self.logit))

        var_list_w = [v for v in tf.trainable_variables() if 'kernel' in v.name]
        loss_reg = args.weight_decay * \
            tf.reduce_mean([tf.nn.l2_loss(v) for v in var_list_w])

        self.loss = self.loss_xent + loss_reg

    def _add_optim(self, decay_step):
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.lr = tf.train.exponential_decay(
            args.lr, self.global_step, decay_step, args.decay_rate, staircase=True, name="learning_rate")

        # var_base, var_new = [], []
        # for v in tf.trainable_variables():
        #     if "itom" in v.name:
        #         var_new.append(v)
        #     else:
        #         var_base.append(v)

        # optim_base = tf.train.AdamOptimizer(self.lr * args.lrp, beta1=args.momentum)
        # optim_new = tf.train.AdamOptimizer(self.lr, beta1=args.momentum)

        # train_base = optim_base.minimize(self.loss, var_list=var_base)
        # train_new = optim_new.minimize(self.loss, var_list=var_new)
        # self.train_op = tf.group(train_base, train_new)

        var_list_w = [v for v in tf.trainable_variables() if 'kernel' in v.name]
        var_list_b = [v for v in tf.trainable_variables() if 'bias' in v.name]

        optim_w = tf.train.MomentumOptimizer(self.lr, args.momentum)
        optim_b = tf.train.MomentumOptimizer(self.lr * 2.0, args.momentum)

        train_w = optim_w.minimize(self.loss, var_list=var_list_w)
        train_b = optim_b.minimize(self.loss, var_list=var_list_b)
        self.train_op = tf.group(train_w, train_b)
