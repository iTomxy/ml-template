import os
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from args import args


def cos(A, B=None):
    """cosine"""
    An = normalize(A, norm='l2', axis=1)
    if (B is None) or (B is A):
        return np.dot(An, An.T)
    Bn = normalize(B, norm='l2', axis=1)
    return np.dot(An, Bn.T)


def hamming(A, B=None):
    """A, B: [None, bit]
    elements in {-1, 1}
    """
    if B is None: B = A
    bit = A.shape[1]
    return (bit - A.dot(B.T)) // 2


def euclidean(A, B=None, sqrt=False):
    aTb = np.dot(A, B.T)
    if (B is None) or (B is A):
        aTa = np.diag(aTb)
        bTb = aTa
    else:
        aTa = np.diag(np.dot(A, A.T))
        bTb = np.diag(np.dot(B, B.T))
    D = aTa[:, np.newaxis] - 2.0 * aTb + bTb[np.newaxis, :]
    if sqrt:
        D = np.sqrt(D)
    return D


def sim_mat(label, label_2=None, sparse=False):
    if label_2 is None:
        label_2 = label
    if sparse:
        S = label[:, np.newaxis] == label_2[np.newaxis, :]
    else:
        S = np.dot(label, label_2.T) > 0
    return S.astype(label.dtype)


def mAP(qF, rF, qL, rL, what=0, k=-1, sparse=False):
    """calculate mAP for retrieval
    Args:
        qF: query feature/hash matrix
        rF: retrieval feature/hash matrix
        qL: query label matrix
        rL: retrieval label matrix
        what:
            - 0: cos
            - 1: hamming (continuous)
            - 2: euclidean
        k: mAP@k, default `-1` means mAP@ALL
    """
    n_query = qF.shape[0]
    if (k < 0) or (k > rF.shape[0]):
        k = rF.shape[0]
    Gnd = sim_mat(qL, rL, sparse).astype(np.int)
    if what == 0:
        Rank = np.argsort(1 - cos(qF, rF))
    elif what == 1:
        Rank = np.argsort(hamming(qF, rF))
    elif what == 2:
        Rank = np.argsort(euclidean(qF, rF))

    AP = 0.0
    for it in range(n_query):
        gnd = Gnd[it]
        if np.sum(gnd) == 0:
            continue
        rank = Rank[it][:k]
        gnd = gnd[rank]
        if np.sum(gnd) == 0:
            continue
        pos = np.asarray(np.where(gnd == 1.)) + 1.0
        rel_cnt = np.arange(pos.shape[-1]) + 1.0
        AP += np.mean(rel_cnt / pos)

    return AP / n_query


def NDCG(qF, rF, qL, rL, what=0, k=-1, sparse=False):
    """Normalized Discounted Cumulative Gain"""
    n_query = qF.shape[0]
    if (k < 0) or (k > rF.shape[0]):
        k = rF.shape[0]
    Rel = np.dot(qL, rL.T).astype(np.float)
    G = 2 ** Rel - 1
    D = np.log2(2 + np.arange(k))
    # Rank_best = np.argsort(- np.dot(qL, rL.T))
    if what == 0:
        Rank = np.argsort(1 - cos(qF, rF))
    elif what == 1:
        Rank = np.argsort(hamming(qF, rF))
    elif what == 2:
        Rank = np.argsort(euclidean(qF, rF))

    _NDCG = 0
    for g, rnk, rnk_best in zip(G, Rank, Rank_best):
        dcg_best = (g[rnk_best[:k]] / D).sum()
        if dcg_best > 0:
            dcg = (g[rnk[:k]] / D).sum()
            _NDCG += dcg / dcg_best
    return _NDCG / n_query


def ACG(qF, rF, qL, rL, what=0, k=-1, sparse=False):
    """Average Cumulative Gains"""
    n_query = qF.shape[0]
    if (k < 0) or (k > rF.shape[0]):
        k = rF.shape[0]
    Rel = np.dot(qL, rL.T).astype(np.float)
    Gain = Rel
    if what == 0:
        Rank = np.argsort(1 - cos(qF, rF))
    elif what == 1:
        Rank = np.argsort(hamming(qF, rF))
    elif what == 2:
        Rank = np.argsort(euclidean(qF, rF))

    _ACG = 0
    for g, rnk in zip(Gain, Rank):
        _ACG += g[rnk[:k]].mean()
    return _ACG / n_query


def WAP(qF, rF, qL, rL, what=0, k=-1, sparse=False):
    """Weighted Mean Precision"""
    n_query = qF.shape[0]
    if (k < 0) or (k > rF.shape[0]):
        k = rF.shape[0]
    G = np.dot(qL, rL.T).astype(np.int)
    S = (G > 0).astype(np.int)
    pos = np.arange(k) + 1
    if what == 0:
        Rank = np.argsort(1 - cos(qF, rF))
    elif what == 1:
        Rank = np.argsort(hamming(qF, rF))
    elif what == 2:
        Rank = np.argsort(euclidean(qF, rF))

    _WAP = 0.0
    for s, g, rnk in zip(S, G, Rank):
        s, g = s[rnk[:k]], g[rnk[:k]]
        n_rel = s.sum()
        if n_rel > 0:
            acg = np.cumsum(g) / pos
            _WAP += acg * s / n_rel

    return _WAP / n_query


def ap_pc(y_true, y_score):
    """AP per class for multi-label classification
    input:
    - y_true: [n, c], ground-truth, in {0, 1}
    - y_score: [n, c], predict score, in [0, 1]
    output:
    - AP: [c], AP[c] = average precision along the c-th class
    """
    n, c = y_true.shape
    # sort along sample, minus for DESC
    Rank = np.argsort(- y_score, axis=0)
    AP = np.zeros([c])
    for i in range(c):
        gnd = y_true[:, i]
        if gnd.sum() == 0:
            continue
        rank = Rank[:, i]
        gnd = gnd[rank]
        pos = np.asarray(np.where(gnd == 1.)) + 1.0
        rel_cnt = np.arange(pos.shape[-1]) + 1.0
        AP[i] = (rel_cnt / pos).mean()

    return AP


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


def mAP_tie(qH, rH, qL, rL, k=-1, sparse=False):
    """tie-aware mAP
    ref: https://blog.csdn.net/HackerTom/article/details/107458334
    """
    n, m = qH.shape[0], rH.shape[0]
    if (k < 0) or (k > m):
        k = m
    D = hamming(qH, rH)
    Rnk = np.argsort(D)
    S = sim_mat(qL, rL, sparse)
    AP = 0
    pos = np.arange(m)  # 0-base
    # t_fi_1[k]: t_{f(k) - 1}
    t_fi_1 = np.zeros([m])
    # r_fi[k]: #relevant samples in the tie where k lies in
    r_fi = np.zeros([m])
    # n_fi[k]: #samples in the tie where k lies in
    n_fi = np.zeros([m])
    # R_fi_1[k]: prefix sum of r_fi (exclude r_fi[k])
    R_fi_1 = np.zeros([m])
    for d, s, rnk in zip(D, S, Rnk):
        Rm = s.sum()
        if 0 == Rm:
            continue
        d_unique = np.unique(d)  # ascending
        d_sort = d[rnk]
        s_sort = s[rnk]
        _R_fi_1 = 0  # R_{f(i) - 1}
        for _d in d_unique:
            tie_idx = (d_sort == _d)
            t_fi_1[tie_idx] = pos[tie_idx].min() # - 1 + 1  # `+1` to shift 0-base to 1-base
            _r_fi = s[tie_idx].sum()
            r_fi[tie_idx] = _r_fi
            n_fi[tie_idx] = tie_idx.astype(np.int).sum()
            R_fi_1[tie_idx] = _R_fi_1  # exclude `_r_fi`
            _R_fi_1 += _r_fi

        # deal with invalid terms
        n_fi_1, r_fi_1 = n_fi - 1, r_fi - 1
        idx_invalid = (n_fi_1 == 0)
        n_fi_1[idx_invalid] = 1
        r_fi_1[idx_invalid] = 0
        # in computing (i - t_{f(i)-1} - 1),
        # the lastest `-1` is megered: pos = i - 1
        kernel = (R_fi_1 + (pos - t_fi_1) * r_fi_1 / n_fi_1 + 1) * r_fi / n_fi / (pos + 1)
        AP += kernel[:k].sum() / Rm

    return AP / n


def NDCG_tie(qH, rH, qL, rL, k=-1, sparse=False):
    """tie-aware NDCG
    ref: https://blog.csdn.net/HackerTom/article/details/107458334
    """
    n, m = qH.shape[0], rH.shape[0]
    if (k < 0) or (k > m):
        k = m
    Rel = np.dot(qL, rL.T).astype(np.float) / np.maximum(qL.sum(1, keepdims=True), 1)
    G = Rel  # 2 ** Rel - 1
    pos = np.arange(m) + 1  # 1-base
    D_inv = 1 / pos  # np.log2(2 + np.arange(m))
    Rank_best = np.argsort(- np.dot(qL, rL.T))
    Dis = hamming(qF, rF)
    Rank = np.argsort(Dis)

    _NDCG = 0
    n_c = np.zeros([m])
    sum_d = np.zeros([m])
    for g, d, rnk, rnk_best in zip(G, Dis, Rank, Rank_best):
        dcg_best = (g[rnk_best] * D_inv)[:k].sum()
        d_unique = np.unique(d)  # ascending
        d_sort = d[rnk]
        g_sort = g[rnk]
        for _d in d_unique:
            tie_idx = (d_sort == _d)
            tie_pos = pos[tie_idx]
            tc_1 = tie_pos[0] - 1  # i.e. tie_pos.min() - 1
            if tc_1 >= k:  # k <= t_{c-1} < t_c, out of range
                break  # continue
            n_c[tie_idx] = tie_idx.astype(np.int).sum()
            tc = tie_pos[-1]
            sum_d[tie_idx] = (1 / np.arange(tc_1 + 1, min(k, tc) + 1)).sum()

        dcg = (g_sort * sum_d)[:k].sum()
        _NDCG += dcg / dcg_best

    return _NDCG / n


def t_sne(F, L, title="tsne"):
    """T-SNE visualization
    F: [n, d], features
    L: [n], label id
    """
    tsne = TSNE(n_components=2, init="pca", random_state=0)
    F = tsne.fit_transform(F)
    fig = plt.figure()
    plt.title(title)
    l1 = plt.scatter(F[:, 0], F[:, 1], s=25, c=L, marker='.', cmap="rainbow")
    plt.legend(handles=[l1], labels=[title], loc="best")
    # plt.show()
    fig.savefig(os.path.join(args.log_path, "{}.png".format(title)))
    plt.close(fig)


def vis_retrieval(F, L, title="retrieval"):
    """T-SNE visualization
    F: [1 + n, d], features, with the query sample at first
    L: [1 + n, c], one-hot label id
    """
    tsne = TSNE(n_components=2, init="pca", random_state=0)
    F = tsne.fit_transform(F)
    fig = plt.figure()
    plt.title(title)
    S = sim_mat(L[:1], L)[0]  # [1 + n]
    plt.scatter(F[:1, 0], F[:1, 1], s=40, c=S[:1], marker='*', cmap="rainbow")
    plt.scatter(F[1:, 0], F[1:, 1], s=25, c=S[1:], marker='.', cmap="rainbow")
    plt.colorbar()
    # plt.show()
    fig.savefig(os.path.join(args.log_path, "{}.png".format(title)))
    plt.close(fig)


if __name__ == "__main__":
    # test mAP
    qB = np.array([[1, -1, 1, 1],
               [-1, -1, -1, 1],
               [1, 1, -1, 1],
               [1, 1, 1, -1]])
    rB = np.array([[1, -1, 1, -1],
                   [-1, -1, 1, -1],
                   [-1, -1, 1, -1],
                   [1, 1, -1, -1],
                   [-1, 1, -1, -1],
                   [1, 1, -1, 1]])
    query_L = np.array([[0, 1, 0, 0],
                        [1, 1, 0, 0],
                        [1, 0, 0, 1],
                        [0, 1, 0, 1]])
    retrieval_L = np.array([[1, 0, 0, 1],
                            [1, 1, 0, 0],
                            [0, 1, 1, 0],
                            [0, 0, 1, 0],
                            [1, 0, 0, 0],
                            [0, 0, 1, 0]])
    print("mAP test:", mAP(qB, rB, query_L, retrieval_L, what=1))

    # test tie-aware P@k, R@k
    k = 3
    D = np.array([
        [1, 1, 2, 3],
        [13, 13, 11, 12],
        [22, 22, 21, 22]
    ])
    print("D:", D)
    S = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 1, 1, 0]
    ])
    print("S:", S)
