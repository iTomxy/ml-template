import tensorflow as tf


def struct_loss(X, Y, S, coef=0.5):
    xTy = coef * tf.matmul(X, tf.transpose(Y))
    loss = (1 - S) * xTy - tf.math.log(tf.math.sigmoid(xTy) + 1e-16)
    return tf.reduce_sum(loss)
