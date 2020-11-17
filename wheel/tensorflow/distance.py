import tensorflow as tf


def euclidean(A, B=None, sqrt=False):
    if (B is None) or (B is A):
        aTb = tf.matmul(A, tf.transpose(A))
        aTa = bTb = tf.linalg.diag_part(aTb)
    else:
        aTb = tf.matmul(A, tf.transpose(B))
        aTa = tf.linalg.diag_part(tf.matmul(A, tf.transpose(A)))
        bTb = tf.linalg.diag_part(tf.matmul(B, tf.transpose(B)))

    D = aTa[:, None] - 2.0 * aTb + bTb[None, :]
    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    D = tf.maximum(D, 0.0)

    if sqrt:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.cast(tf.equal(D, 0.0), "float32")
        D = D + mask * 1e-16
        D = tf.math.sqrt(D)
        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        D = D * (1.0 - mask)

    return D


def cos(X, Y=None):
    """cosine of every (Xi, Yj) pair
    X, Y: (n, dim)
    """
    X_n = tf.math.l2_normalize(X, axis=1)
    if (Y is None) or (X is Y):
        return tf.matmul(X_n, tf.transpose(X_n))
    Y_n = tf.math.l2_normalize(Y, axis=1)
    _cos = tf.matmul(X_n, tf.transpose(Y_n))
    return tf.clip_by_value(_cos, -1, 1)


def hamming(X, Y=None):
    if Y is None:
        Y = X
    K = tf.cast(tf.shape(X)[1], "float32")
    D = (K - tf.matmul(X, tf.transpose(Y))) / 2
    return tf.clip_by_value(D, 0, K)


def rbf_kernel(X, Y=None, sigma=1.25):
    """K(i,j) = exp(-0.5 * ||xi - xj||^2 / sigma^2)
    sigma: width of RBF kernel, default 1.25 from paper:
        - Learning with Local and Global Consistency
    """
    D = euclidean(X, Y)
    return tf.math.exp(-0.5 * D / sigma**2)
