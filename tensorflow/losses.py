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
