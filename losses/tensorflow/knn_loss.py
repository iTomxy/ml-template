import tensorflow as tf


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
