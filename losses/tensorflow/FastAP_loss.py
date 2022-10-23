import tensorflow as tf
from wheel import *


def FastAP(X, L, R=11):
    """FastAP
    X: [n, d], feature WITHOUT L2 norm
    L: [n, c], label
    R: scalar, num of bins, `L` in Eq (14)
    """
    delta = 4. / (R - 1)
    # z = (z_1, ..., z_R)
    z = tf.lin_space(0., 4., R)[None, :, None]  # [1, R, 1]
    # norm & euclidean -> in [0, 4]
    Xn = tf.math.l2_normalize(X, axis=1)
    D = euclidean_dist(Xn, Xn)  # [n, n]
    # D = hamming(X)
    # gound-truth, similarity matrix
    M = sim_mat(L)  # [n, n]

    Nq_pos = tf.reduce_sum(M, 1, keepdims=True)  # [n, 1]

    # soft histogram
    D_ = tf.expand_dims(D, 1)  # [n, 1, n]
    hist = tf.maximum(0., 1 - tf.math.abs(D_ - z) / delta)  # [n, R, n]
    hist_pos = hist * tf.expand_dims(M, 1)

    # h[i] = (h_1, ..., h_R)
    h = tf.reduce_sum(hist, -1)  # [n, R]
    h_pos = tf.reduce_sum(hist_pos, -1)  # [n, R]

    # H[i] = (H_1, ..., H_R)
    # H[i][j] = sum(h[i][1], ..., h[i][j])
    mask_cumsum = tf.linalg.band_part(
        tf.ones([R, R]), -1, 0)[None, :]  # [1, R, R]
    H = tf.reduce_sum(tf.expand_dims(h, 1) * mask_cumsum, -
                      1)  # [n, R, R] -> [n, R]
    H_pos = tf.reduce_sum(tf.expand_dims(h_pos, 1) * mask_cumsum, -1)

    fast_ap = h_pos * H_pos / tf.maximum(1e-7, H)  # [n, R]
    fast_ap = tf.reduce_sum(fast_ap, 1) / Nq_pos  # [n]
    loss = 1 - tf.reduce_mean(fast_ap)
    return loss
