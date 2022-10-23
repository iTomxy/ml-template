import numpy as np
# from scipy.special import expit


def _triplet_mask(Rel):
    n, m = Rel.shape(0), Rel.shape(1)
    I = np.eye(n, m)
    neq_id = 1 - I  # [n, m]
    neq_ij = neq_id[:, :, np.newaxis]  # [n, m, 1]
    neq_ik = neq_id[:, np.newaxis, :]  # [n, 1, m]
    neq_jk = (1 - np.eye(m))[np.newaxis, :]  # [1, m, m]
    mask_index = neq_ij * neq_ik * neq_jk

    mask_label = (Rel[:, :, np.newaxis] > Rel[:, np.newaxis, :]).astype(np.float)

    mask = mask_index * mask_label
    return mask


def triplet_loss(Dist, Rel, margin=1):
    D_pos = Dist[:, :, np.newaxis]
    D_neg = Dist[:, np.newaxis, :]
    kernel = D_pos + margin - D_neg

    mask_triplet = _triplet_mask(Rel)
    loss_triplet = (0.5 * mask_triplet * kernel).clip(min=0)

    n_pos = (loss_triplet > 1e-16).float().sum()
    return loss_triplet.sum() / (n_pos + 1e-16)


def nll_triplet_loss(Dist, Rel, margin=1, gamma=5):
    D_pos = Dist[:, :, np.newaxis]
    D_neg = Dist[:, np.newaxis, :]
    kernel = D_pos + margin - D_neg

    mask_triplet = _triplet_mask(Rel)
    loss_triplet = np.log(1 + np.exp(- np.fabs(kernel))) + kernel.clamp(min=0) - kernel
    # loss_triplet = 
    loss_triplet = (mask_triplet * loss_triplet).clip(min=0)
    return loss_triplet.sum()
