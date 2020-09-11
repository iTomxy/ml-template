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


def knn_loss(pairwise_dist, k_pos, k_neg, margin):
    mask_knn_pos = top_k_mask(-1.0 * pairwise_dist,
                              k_pos, rand_pick=True)  # nearest
    mask_knn_neg = top_k_mask(pairwise_dist, k_neg, rand_pick=True)  # farthest

    mask_knn_pos = tf.cast(mask_knn_pos, "float32")
    mask_knn_neg = tf.cast(mask_knn_neg, "float32")

    dis_pos = pairwise_dist
    dis_neg = tf.maximum(0.0, margin - pairwise_dist)
    knn_loss = (dis_pos * mask_knn_pos + dis_neg * mask_knn_neg) / 2.0

    valid_item = tf.to_float(tf.greater(knn_loss, 1e-16))
    n_valid = tf.reduce_sum(valid_item)
    knn_loss = tf.reduce_sum(knn_loss) / (n_valid + 1e-16)
    return knn_loss


def pseudo_loss(pairwise_dist, labels, pseudo_labels, margin):
    dis_pos = pairwise_dist
    dis_neg = tf.maximum(0.0, margin - pairwise_dist)

    # Unlabeled v.s. Unlabeled
    pseudo_max = tf.argmax(pseudo_labels, axis=-1, name="pseudo_max")
    pseudo_min = tf.argmin(pseudo_labels, axis=-1, name="pseudo_min")
    # pos in terms of pseudo labels: max[i] = max[j]
    mask_pos_uu = tf.equal(tf.expand_dims(pseudo_max, 1),
                           tf.expand_dims(pseudo_max, 0), name="mask_pos_uu")
    # neg in terms of pseudo labels: min[i] = max[j]
    mask_neg_uu = tf.equal(tf.expand_dims(pseudo_min, 1),
                           tf.expand_dims(pseudo_max, 0), name="mask_neg_uu")

    # Unlabeled v.s. Labeled
    mask_pseudo_max = top_k_mask(pseudo_labels, 1)
    mask_pseudo_min = top_k_mask(-1.0 * pseudo_labels, 1)
    mask_label = tf.to_int32(labels)  # tf.cast(labels, "bool")
    mask_label_not = 1 - mask_label  # tf.logical_not(mask_label)

    true_pos = tf.matmul(mask_pseudo_max, tf.transpose(mask_label),
                         name="true_pos") > 0
    false_pos = tf.matmul(mask_pseudo_max, tf.transpose(mask_label_not),
                          name="false_pos") > 0
    false_neg = tf.matmul(mask_pseudo_min, tf.transpose(mask_label),
                          name="false_neg") > 0
    true_neg = tf.matmul(mask_pseudo_min, tf.transpose(mask_label_not),
                         name="true_neg") > 0

    mask_pos_ul = tf.logical_and(true_pos, true_neg, name="mask_pos_ul")
    mask_neg_ul = tf.logical_and(false_pos, false_neg, name="mask_neg_ul")

    # sum up
    mask_pos = tf.to_float(
        tf.concat([mask_pos_uu, mask_pos_ul], axis=1), name="mask_pos")
    mask_neg = tf.to_float(
        tf.concat([mask_neg_uu, mask_neg_ul], axis=1), name="mask_neg")
    pseudo_loss = (mask_pos * dis_pos + mask_neg * dis_neg) / 2.0
    valid_item = tf.to_float(tf.greater(pseudo_loss, 1e-16))
    n_valid = tf.reduce_sum(valid_item)
    pseudo_loss = tf.reduce_sum(pseudo_loss) / (n_valid + 1e-16)

    return pseudo_loss


def DJSRH_loss(fc7, fc7_u, hc, hc_u):
    fc7_ul = tf.concat([fc7_u, fc7], axis=0)
    S = cos(fc7_ul, fc7_ul)  # * 2 - 1
    S = (1 - args.eta) * S + args.eta * \
        tf.matmul(S, tf.transpose(S)) / (args.batch_size + args.batch_size_u)

    hc_ul = tf.concat([hc, hc_u], axis=0)
    BtB = cos(hc_ul, hc_ul)

    return tf.losses.mean_squared_error(args.mu * S, BtB)


def struct_loss(X, Y, S, coef=0.5):
    xTy = coef * tf.matmul(X, tf.transpose(Y))
    loss = (1 - S) * xTy - tf.math.log(tf.math.sigmoid(xTy) + 1e-16)
    return tf.reduce_sum(loss)


def factor_loss(H, S):
    """S(i,j) in {-1, 1}
    H(i,j) in [-1, 1]
    """
    bit = tf.cast(tf.shape(H)[-1], "float32")
    return tf.nn.l2_loss(bit * S - tf.matmul(H, tf.transpose(H)))


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
