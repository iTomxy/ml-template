import tensorflow as tf


def sim_mat(label, label2=None, sparse=False):
    """similarity matrix
    S[i][j] = 1 <=> i- & j-th share at lease 1 label
    S[i][j] = 0 otherwise
    """
    label2 = label if (label2 is None) else label2
    if sparse:
        S = tf.cast(tf.equal(label[:, None], label2[None, :]), "float32")
    else:
        S = tf.cast(tf.matmul(label, tf.transpose(label2)) > 0, "float32")
    return S
