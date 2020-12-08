import numpy as np


def mAP(Dist, S, k=-1):
    """mean Average Precision
    - Dist: distance matrix
    - S: similarity matrix
    - k: mAP@k, default `-1` means mAP@ALL
    ref:
    - https://blog.csdn.net/HackerTom/article/details/89309665
    """
    n, m = Dist.shape
    if (k < 0) or (k > m):
        k = m
    Gnd = S
    Rank = np.argsort(Dist)

    AP = 0.0
    for it in range(n):
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

    return AP / n


def mAP_tie(Dist, S, k=-1):
    """tie-aware mAP
    Dist: [n, m], Hamming distance matrix
    S: [n, m], similarity mattrix
    ref:
    - https://blog.csdn.net/HackerTom/article/details/107458334
    - https://github.com/kunhe/TALR/blob/master/%2Beval/tieAP.m
    """
    n, m = Dist.shape
    if (k < 0) or (k > m):
        k = m
    Rnk = np.argsort(Dist)
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
    for d, s, rnk in zip(Dist, S, Rnk):
        # Rm = s.sum()  # #rel in all
        s_sort = s[rnk]
        Rm = s_sort[:k].sum()  # #rel in top-k
        if 0 == Rm:
            continue
        d_unique = np.unique(d)  # ascending
        d_sort = d[rnk]
        # s_sort = s[rnk]
        _R_fi_1 = 0  # R_{f(i) - 1}
        for _d in d_unique:
            tie_idx = (d_sort == _d)
            t_fi_1[tie_idx] = pos[tie_idx].min() # - 1 + 1  # `+1` to shift 0-base to 1-base
            _r_fi = s_sort[tie_idx].sum()
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
    print("NDCG test:", NDCG(qB, rB, query_L, retrieval_L, what=1))

    # test tie-aware P@k, R@k
    print("tie mAP:", mAP_tie(qB, rB, query_L, retrieval_L, k=-1, sparse=False))
    print("tie NDCG:", NDCG_tie(qB, rB, query_L, retrieval_L, k=-1, sparse=False))
    print("change place")
    print("tie mAP:", mAP_tie(rB, qB, retrieval_L, query_L, k=-1, sparse=False))
    print("tie NDCG:", NDCG_tie(rB, qB, retrieval_L, query_L, k=-1, sparse=False))
