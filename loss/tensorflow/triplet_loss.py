import tensorflow as tf
from wheel import *
from args import args


"""Ref:
https://omoindrot.github.io/triplet-loss
https://blog.csdn.net/hustqb/article/details/80361171#commentBox
https://blog.csdn.net/hackertom/article/details/103374313
"""


def _triplet_mask(L, L2=None, sparse=False):
    """M(i,j,k) = 1 iff:
    - i != j != k, and
    - sim(i,j), dissim(i,k)
    L: [n, c] if not sparse, else [n]
    L2: (is not None) [m, c] if not sparse, else [m]
    """
    if L2 is None: L2 = L
    n, m = L.shape[0], L2.shape[0]

    I = tf.eye(n, m, dtype="int32")
    neq_id = 1 - I  # [n, m]
    neq_ij = tf.expand_dims(neq_id, 2)  # [n, m, 1]
    neq_ik = tf.expand_dims(neq_id, 1)  # [n, 1, m]
    neq_jk = tf.expand_dims(1 - tf.eye(m, dtype="int32"), 0)  # [1, m, m]
    mask_index = neq_ij * neq_ik * neq_jk  # [n, m, m]
    # print("mask_index:", mask_index.shape)

    S = tf.cast(sim_mat(L, L2, sparse), "int32")
    sim_ij = tf.expand_dims(S, 2)  # [n, m, 1]
    dissim_ik = tf.expand_dims(1 - S, 1)  # [n, 1, m]
    mask_label = sim_ij * dissim_ik  # [n, m, m]
    # print("mask_label:", mask_label.shape)

    mask = mask_index * mask_label
    return mask


def triplet_loss(X, L, X2=None, L2=None, margin=1, dist_fn=euclidean, sparse=False):
    """triplet loss (batch all)
    X, X2: [n, d] & [m, d], feature
    L, L2: [n, c] & [m, c] if not sparse, else [n] & [m], label
    dist_fn: distance function, default to euclidean
    sparse: in form of sparse class ID if true, else one-hot
    """
    if X2 is None:
        # assert L2 is None
        X2, L2 = X, L
    pairwise_dist = dist_fn(X, X2)  # [n, m]
    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)

    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask_triplet = _triplet_mask(L, L2, sparse)

    mask_triplet = tf.cast(mask_triplet, "float32")
    triplet_loss = tf.multiply(mask_triplet, triplet_loss)

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = tf.maximum(triplet_loss, 0.0) / 2.0

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = tf.cast(tf.greater(triplet_loss, 1e-16), "float32")
    num_positive_triplets = tf.reduce_sum(valid_triplets)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = tf.reduce_sum(triplet_loss) / \
        (num_positive_triplets + 1e-16)

    return triplet_loss
