import numpy as np


def prfa(y_true, y_pred):
    """Precision, Recall, F1, Accuracy
    - micro: OP, OR, OF1
    - macro: CP, CR, CF1
    input:
    - y_true: [n, c], ground-truth, in {0, 1}
    - y_pred: [n, c], prediction, in {0, 1}
    output:
    - OP, OR, OF1, CP, CR, CF1, acc
    """
    true = (y_true > 0.5).astype(np.int)
    pred = (y_pred > 0.5).astype(np.int)
    same = (true == pred).astype(np.int)
    n, c = y_true.shape
    gnd = true.sum(0)
    pos = pred.sum(0)
    tp = (true * pred).sum(0)

    OP = tp.sum() / max(pos.sum(), 1)
    OR = tp.sum() / max(gnd.sum(), 1)
    OF1 = (2 * OP * OR) / (OP + OR)

    pos[pos == 0] = 1
    tp[tp == 0] = 1
    CP = (tp / pos).sum() / c
    CR = (tp / gnd).sum() / c
    CF1 = (2 * CP * CR) / (CP + CR)

    acc_pc = same.sum(0) / n
    acc = acc_pc.sum() / c

    return OP, OR, OF1, CP, CR, CF1, acc, acc_pc


def P_R_F1_tie(qH, rH, qL, rL, k=-1, sparse=False):
    """tie-aware precision@k, recall@k, F1@k
    qH, rH: [n, bit] & [m, bit], hash code of query & database samples
    qL, rL: [n, #class] & [m, #class], label of query & database samples
        if sparse, shapes are [n] & [m]
    k: position threshold of `P@k`
    sparse: whether label is sparse class ID or one-hot vector
    ref: https://blog.csdn.net/HackerTom/article/details/107458334
    """
    m = rH.shape[0]
    if (k < 0) or (k > m):
        k = m
    D = hamming(qH, rH)
    Rnk = np.argsort(D)
    S = sim_mat(qL, rL, sparse)
    # find the tie where k lies in: t_{c-1} < k <= t_c
    D_sort = np.vstack([d[r] for d, r in zip(D, Rnk)])
    D_at_k = D_sort[:, k-1:k]  # `-1` for 0-base
    mask_tie = np.equal(D, D_at_k).astype(np.int)
    # r_c, n_c
    nc = mask_tie.sum(1)
    rc = (mask_tie * S).sum(1)
    # find t_{c-1}
    pos = np.arange(m)[np.newaxis, :]  # [1, m]
    tie_pos = np.where(np.equal(D_sort, D_at_k), pos, np.inf)
    tc_1 = np.min(tie_pos, 1) # - 1 + 1  # `+1` to shift 0-base to 1-base
    # R_{c-1}
    mask_pre = (D < D_at_k).astype(np.int)
    Rc_1 = (mask_pre * S).sum(1)
    # P@k, R@k, F1@k
    _common = Rc_1 + (k - tc_1) * rc / nc  # [n]
    Rm = S.sum(1)
    P_at_k = _common / k
    R_at_k = _common / Rm
    F1_at_k = 2 * _common / (k + Rm)
    return P_at_k.mean(), R_at_k.mean(), F1_at_k.mean()
