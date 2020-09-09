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


def mAPrs_loss(X, L, X2=None, L2=None, n_bin=None, delta_scale=1, sparse=False):
    """simplified continuous relaxation of tie-aware mAP
    X, X2: [n, bit] & [m, bit], continuous hash in (-1, 1)
    L, L2: [n, c] & [m, c], labels, if `sparse` then [n] & [m]
    n_bin: # of bins
    delta_scale: scaling factor for the \Delta parameter
    sparse: True if the labels are sparse class ID
    ref:
    - https://github.com/kunhe/TALR/blob/master/apr_s_forward.m
    - https://blog.csdn.net/HackerTom/article/details/106181622
    """
    if X2 is None:
        X2, L2 = X, L
    D = hamming(X, X2)
    S = sim_mat(L, L2, sparse)
    bit = X.size(1)
    if n_bin is None:
        n_bin = bit
        if bit >= 32:
            n_bin = bit // 2
    S_inv = 1 - S
    S = S - S.diag().diag()
    delta = bit / n_bin * delta_scale
    t = torch.linspace(0, bit, n_bin + 1).to(D.device)  # histogram centres
    # soft_mask(i,j,k) > 0 means that
    # dist(i,j) lies in the region of the k-th bin
    scaled_abs_diff = (D.unsqueeze(2) - t.view(1, 1, -1)).abs() / delta
    soft_mask = (1 - scaled_abs_diff).clamp(min=0)  # [n, m, n_bin]
    # cp(d): (soft) #pos samples lying in the tie of distance d
    cp = (soft_mask * S.unsqueeze(2)).sum(1)  # [n, n_bin]
    # c(d): (soft) #samples lying in the tie of distance d
    c = cp + (soft_mask * S_inv.unsqueeze(2)).sum(1)  # [n, n_bin]
    # Cp, C: cumulative sum of cp & c, respectively
    Cp = cp.cumsum(1)  # [n, n_bin]
    C = c.cumsum(1)  # [n, n_bin]
    # Cp_1, C_1: C_{d-1}^+, C_{d-1}
    zero = torch.zeros_like(C[:, 0:1])  # [n, 1]
    Cp_1 = torch.cat([zero, Cp[:, :-1]], 1)
    C_1 = torch.cat([zero, C[:, :-1]], 1)
    # Np(i): (hard) #pos samples in the i-th retriavel list
    Np = S.sum(1)  # [n]
    APr_s = (cp * (Cp_1 + Cp + 1) / (C_1 + C + 1)).sum(1) / Np  # [n]
    # deal with invalid terms
    APr_s = APr_s.where(Np > 0, torch.zeros_like(APr_s)) / delta_scale
    return (1 - APr_s).sum()  # to maximize


def NDCGrs_loss(X, L, X2=None, L2=None, n_bin=None, sparse=False):
    """lower bound of tie-aware NDCG
    X, X2: [n, bit] & [m, bit], continuous hash in (-1, 1)
    L, L2: [n, c] & [m, c] labels, if `sparse` then [n] & [m]
    n_bin: # of bins
    delta_scale: scaling factor for the \Delta parameter
    sparse: True if the labels are sparse class ID
    ref:
    - https://github.com/kunhe/TALR/blob/master/ndcgr_s_forward.m
    """
    if X2 is None:
        X2, L2 = X, L
    n, m, bit = X.size(0), X2.size(0), X.size(1)
    if n_bin is None:
        n_bin = bit // 2
    D = hamming(X, X2)
    S = L.mm(L2.T)
    Gain = 2 ** S - 1  # NOTE: mind overflowing, e.g., COCO dataset
    S_unique = S.unique(sorted=True).float()  # ascending
    S = S - S.diag().diag()
    S_mask = (S.unsqueeze(2) == S_unique.view(1, 1, -1)).float()  # [n, m, #s]
    S_mask[:, :, 0] -= S_mask[:, :, 0].diag().diag()
    Discount = 1 / (torch.arange(m).float() + 2).log2()
    Discount = Discount.unsqueeze(0).to(X.device)  # [1, m]
    delta = bit / n_bin
    t = torch.linspace(0, bit, n_bin + 1).to(D.device)  # histogram centres
    # soft_mask(i,j,k) > 0 means that
    # dist(i,j) lies in the region of the k-th bin
    scaled_abs_diff = (D.unsqueeze(2) - t.view(1, 1, -1)).abs()
    soft_mask = (1 - scaled_abs_diff).clamp(min=0)  # [n, m, n_bin]
    # c_ds(i,j,k): #samples in the i-th retrieval list that
    #   of similarity level k and lie in the bin j
    c_ds = torch.zeros(n, t.size(0), S_unique.size(0))
    c_ds = c_ds.to(X.device)  # [n, n_bin, #s]
    for _t in range(t.size(0)):
        for _s in range(S_unique.size(0)):
            _c_ds_mask = soft_mask[:, :, _t] * S_mask[:, :, _s]  # [n, m]
            c_ds[:, _t, _s] = _c_ds_mask.sum(1)  # [n]
    # c_d(i,j): #samples in the i-th retrieval list lying in j-th tie
    c_d = c_ds.sum(2)  # [n, n_bin]
    # C_d: cumsum of c_d
    C_d = c_d.cumsum(1)  # [n, n_bin]
    # C_1d: C_{d-1}
    zero = torch.zeros_like(C_d[:, 0:1])
    C_1d = torch.cat([zero, C_d[:, :-1]], 1)  # [n, n_bin]
    # C_bar = C_{d-1} + (c_d + 1) / 2 + 1
    C_bar = C_1d + (c_d + 1) / 2 + 1  # [n, n_bin]
    # Gns(i): gain of similarity level i
    Gns = 2 ** S_unique.float() - 1  # [#s], mind overflowing
    # G_hat(i,j): sum gain of j-th bin in the i-th retrieval list
    G_hat = (c_ds * Gns.view(1, 1, -1)).sum(2)  # [n, n_bin]
    _DCG = (G_hat / C_bar.log2()).sum(1)  # [n]
    _DCGi = (Gain.sort(1, descending=True)[0] * Discount).sum(1)  # [n]
    NDCGr_s = torch.where(_DCGi > 0, _DCG / _DCGi, torch.zeros_like(_DCG))
    return (1 - NDCGr_s).sum()  # to maximize
