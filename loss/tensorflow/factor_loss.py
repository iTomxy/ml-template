import tensorflow as tf


def factor_loss(H, S):
    """S(i,j) in {-1, 1}
    H(i,j) in [-1, 1]
    """
    bit = tf.cast(tf.shape(H)[-1], "float32")
    return tf.nn.l2_loss(bit * S - tf.matmul(H, tf.transpose(H)))
