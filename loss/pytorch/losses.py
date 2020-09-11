import torch
import torch.nn.functional as F
from wheel import *


"""Ref:
https://omoindrot.github.io/triplet-loss
https://blog.csdn.net/hustqb/article/details/80361171#commentBox
https://blog.csdn.net/hackertom/article/details/103374313
"""


def struct_loss(X, Y, S, coef=0.5):
    xTy = coef * X.mm(Y.T)
    loss = (1 - S) * xTy - (xTy.sigmoid() + 1e-16).log()
    return loss.mean()


def _triplet_mask(L, L2=None, sparse=False):
    if L2 is None:
        L2 = L
    n, m = L.size(0), L2.size(0)
    I = torch.eye(n, m).to(L.device)
    neq_id = 1 - I  # [n, m]
    neq_ij = neq_id.unsqueeze(2)  # [n, m, 1]
    neq_ik = neq_id.unsqueeze(1)  # [n, 1, m]
    neq_jk = (1 - torch.eye(m)).unsqueeze(0).to(L.device)  # [1, m, m]
    mask_index = neq_ij * neq_ik * neq_jk

    S = sim_mat(L, L2, sparse)  # [n, m]
    sim_ij = S.unsqueeze(2)  # [n, m, 1]
    dissim_ik = (1 - S).unsqueeze(1)  # [n, 1, m]
    mask_label = sim_ij * dissim_ik

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
        X2, L2 = X, L
    D = dist_fn(X, X2)
    D_pos = D.unsqueeze(2)
    D_neg = D.unsqueeze(1)
    kernel = D_pos + margin - D_neg

    mask_triplet = _triplet_mask(L, sparse=sparse)
    loss_triplet = 0.5 * mask_triplet * kernel
    # loss_triplet[loss_triplet < 0] = 0.0
    loss_triplet = torch.max(loss_triplet, torch.zeros_like(loss_triplet).to(X.device))

    n_pos = (loss_triplet > 1e-16).float().sum()
    return loss_triplet.sum() / (n_pos + 1e-16)


def l2_regularize(w_list):
    loss = 0
    for w in w_list:
        loss += 0.5 * (w ** 2).sum()
    return loss
