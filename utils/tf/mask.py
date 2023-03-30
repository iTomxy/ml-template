import tensorflow as tf


def top_k_mask(D, k, rand_pick=False):
    """M[i][j] = 1 <=> D[i][j] is oen of the BIGGEST k in i-th row
    Args:
        D: (n, n), distance matrix
        k: param `k` of kNN
        rand_pick: true or false
            - if `True`, only ONE of the top-K element in each row will be selected randomly;
            - if `False`, ALL the top-K elements will be selected as usual.
    Ref:
        - https://cloud.tencent.com/developer/ask/196899
        - https://blog.csdn.net/HackerTom/article/details/103587415
    """
    n_row = tf.shape(D)[0]
    n_col = tf.shape(D)[1]

    k_val, k_idx = tf.math.top_k(D, k)
    if rand_pick:
        c_idx = tf.random_uniform([n_row, 1],
                                  minval=0, maxval=k,
                                  dtype="int32")
        r_idx = tf.range(n_row, dtype="int32")[:, None]
        idx = tf.concat([r_idx, c_idx], axis=1)
        k_idx = tf.gather_nd(k_idx, idx)[:, None]

    idx_offset = (tf.range(n_row) * n_col)[:, None]
    k_idx_linear = k_idx + idx_offset
    k_idx_flat = tf.reshape(k_idx_linear, [-1, 1])

    updates = tf.ones_like(k_idx_flat[:, 0], "int32")
    mask = tf.scatter_nd(k_idx_flat, updates, [n_row * n_col])
    mask = tf.reshape(mask, [-1, n_col])
    # mask = tf.cast(mask, "bool")

    return mask


def count_mask(X):
    """M(i,j) = |{ (s,t) | X(s,t) = X(i,j) }|
    ref: https://blog.csdn.net/HackerTom/article/details/108902880
    """
    x_flat = tf.cast(tf.reshape(X, [-1]), "int32")
    _bin = tf.math.bincount(x_flat)
    mask = tf.reshape(tf.gather(_bin, x_flat), tf.shape(X))
    return tf.cast(mask, X.dtype)
