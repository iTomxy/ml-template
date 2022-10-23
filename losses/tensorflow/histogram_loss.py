import tensorflow as tf
from wheel import *


def histogram_loss(X, L, R=151):
    """hisgogram loss
    X: [n, d], feature WITHOUT L2 norm
    L: [n, c], label
    R: scalar, num of estimating point, same as the paper
    """
    delta = 2. / (R - 1)  # step
    # t = (t_1, ..., t_R)
    t = tf.lin_space(-1., 1., R)[:, None]  # [R, 1]
    # gound-truth, similarity matrix
    M = sim_mat(L)  # [n, n]
    # cosine similarity, in [-1, 1]
    S = cos(X)  # [n, n]

    # get indices of upper triangular (without diag)
    S_hat = S + 2  # shift value to [1, 3] to ensure triu > 0
    S_triu = tf.linalg.band_part(S_hat, 0, -1) * (1 - tf.eye(tf.shape(S)[0]))
    triu_id = tf.where(S_triu > 0)

    # extract triu -> vector of [n(n - 1) / 2]
    S = tf.gather_nd(S, triu_id)[None, :]  # [1, n(n-1)/2]
    M_pos = tf.gather_nd(M, triu_id)[None, :]
    M_neg = 1 - M_pos

    scaled_abs_diff = tf.math.abs(S - t) / delta  # [R, n(n-1)/2]
    # mask_near = tf.cast(scaled_abs_diff <= 1, "float32")
    # delta_ijr = (1 - scaled_abs_diff) * mask_near
    delta_ijr = tf.maximum(0., 1 - scaled_abs_diff)

    def histogram(mask):
        """h = (h_1, ..., h_R)"""
        sum_delta = tf.reduce_sum(delta_ijr * mask, 1)  # [R]
        return sum_delta / tf.maximum(1., tf.reduce_sum(mask))

    h_pos = histogram(M_pos)[None, :]  # [1, R]
    h_neg = histogram(M_neg)  # [R]
    # all 1 in lower triangular (with diag)
    mask_cdf = tf.linalg.band_part(tf.ones([R, R]), -1, 0)
    cdf_pos = tf.reduce_sum(mask_cdf * h_pos, 1)  # [R]

    loss = tf.reduce_sum(h_neg * cdf_pos)
    return loss
