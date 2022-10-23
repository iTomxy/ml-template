import tensorflow as tf


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
