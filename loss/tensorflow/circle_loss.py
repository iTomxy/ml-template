import tensorflow as tf
from wheel import *


def circle_loss(X, L, gamma=80, margin=0.4):
    """Eq (1) of [1]
    X: [n, d]
    L: [n, c]
    gamma, margin: from [1]
    fomula:
    L(a) = b + ln[exp(- gamma * b) + sum{exp(gamma * (s_neg_j - s_pos_i + m))}] / gamma
    ref:
    1. Circle Loss: A Unified Perspective of Pair Similarity Optimization
    """
    mask_triplet = tf.cast(_get_triplet_mask(L), "float32")
    S = cos(X)  # X > 0 <- hash_con out from sigmoid
    # S = 0.5 * (1 + S)  # [0, 1]
    S_pos = tf.expand_dims(S, 2)
    S_neg = tf.expand_dims(S, 1)
    # kernel = tf.math.exp(gamma * (S_neg - S_pos + margin))  # boom
    kernel = S_neg - S_pos + margin
    # D = euclidean_dist(X, X)
    # D_pos = tf.expand_dims(D, 2)
    # D_neg = tf.expand_dims(D, 1)
    # kernel = (D_pos + margin - D_neg) * mask_triplet
    big = tf.reduce_max(kernel * mask_triplet, axis=[1, 2])  # [#batch]
    big = tf.maximum(big, 0.0)  # avoid minus -> boom
    kernel = tf.math.exp(gamma * (kernel - big[:, None, None]))  # -big to avoid overflow

    loss = tf.reduce_sum(mask_triplet * kernel, [1, 2])  # [#batch]
    loss = big + tf.math.log(tf.math.exp(- gamma * big) + loss) / gamma  # [#batch]
    # loss = tf.maximum(0.0, loss)
    valid_triplets = tf.cast(tf.greater(loss, 1e-16), "float32")
    n_valid = tf.reduce_sum(valid_triplets)
    loss = tf.reduce_sum(loss) / tf.maximum(n_valid, 1.0)
    return loss
