import tensorflow as tf
from wheel import *


def contrastive_loss(X, L, X2=None, L2=None, margin=2.5, margin2=None, sparse=False):
    """sparse contrastive loss
    X, X2: [n, d], feature
    L, L2: [n, c] one-hot label vec if not sparse, or [n] sparse label id
    if only one margin:
    loss(x1, x2) = {
        || x1 - x2 ||^2              ,  l1 == l2
        max{ 0, m - || x1 - x2 ||^2 },  else
    }
    else (m < m2):
    loss(x1, x2) = {
        max( 0, || x1 - x2 ||^2 - m   ,  l1 == l2
        max{ 0, m2 - || x1 - x2 ||^2 },  else
    }
    """
    if X2 is None:
        # assert L2 is None
        X2, L2 = X, L
    S = sim_mat(L, L2, sparse=sparse)
    D = euclidean(X, X2)
    if margin2 is not None:
        D_pos = tf.math.maximum(0.0, D - margin)
        D_neg = tf.math.maximum(0.0, margin2 - D)
    else:
        D_pos = D
        D_neg = tf.math.maximum(0.0, margin - D)
    loss = S * D_pos + (1 - S) * D_neg
    n_pos = tf.reduce_sum(tf.cast(loss > 1e-16, "float32"))
    return tf.reduce_sum(loss) / (n_pos + 1e-16)
