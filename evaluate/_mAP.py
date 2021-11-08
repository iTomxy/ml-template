import copy
import numpy as np


def mAP(Dist, Sim, k=-1):
    """mean Average Precision
    Input:
        Dist: distance matrix
        Sim: 0/1 similarity matrix
        k: mAP@k, int or int tuple/list
           default `-1` means mAP@ALL
    ref:
    - https://blog.csdn.net/HackerTom/article/details/89309665
    """
    if isinstance(k, int):
        k = [k]
    else:
        k = copy.deepcopy(k)
    n, m = Dist.shape
    for kid in range(len(k)):
        if (k[kid] < 0) or (k[kid] > m):
            k[kid] = m
    k = sorted(k)  # ascending
    assert k[0] != 0, "`@0` is meaningless and disallowed for efficiency"
    Gnd = (Sim > 0).astype(np.int32)  # ensure 0/1
    gnd_rs = np.sum(Gnd, axis=1)
    Rank = np.argsort(Dist, axis=-1)

    AP = np.zeros([len(k)], dtype=np.float32)
    for it in range(n):
        gnd = Gnd[it]
        if 0 == gnd_rs[it]:
            continue
        rank = Rank[it]#[:k]
        gnd = gnd[rank]
        # if (k > 0) and (np.sum(gnd) == 0):
        #     continue
        pos = np.asarray(np.where(gnd == 1.)).flatten() + 1.0
        rel_cnt = np.arange(pos.shape[-1]) + 1.0
        # AP += np.mean(rel_cnt / pos)
        p_list = rel_cnt / pos

        _cnt, _p_sum = 0, 0
        for kid, _k in enumerate(k):
            if (0 == _k) or (pos[_cnt] > _k):
                continue
            while (_cnt < pos.shape[0]) and (pos[_cnt] <= _k):
                _p_sum += p_list[_cnt]
                _cnt += 1
            _ap = _p_sum / _cnt
            AP[kid] += _ap
            if _cnt >= pos.shape[0]:
                break

    _mAP = AP / n
    if 1 == _mAP.shape[0]:
        _mAP = _mAP[0]
    return _mAP


def mAP_tie(Dist, Sim, k=-1):
    """tie-aware mAP
    Input:
        Dist: [n, m], Hamming distance matrix
        Sim: [n, m], similarity mattrix
        k: mAP@k, int or int tuple/list
           default `-1` means mAP@ALL
    ref:
    - https://blog.csdn.net/HackerTom/article/details/107458334
    - https://github.com/kunhe/TALR/blob/master/%2Beval/tieAP.m
    """
    if isinstance(k, int):
        k = [k]
    else:
        k = copy.deepcopy(k)
    n, m = Dist.shape
    for kid in range(len(k)):
        if (k[kid] < 0) or (k[kid] > m):
            k[kid] = m
    k = sorted(k)  # ascending
    assert k[0] != 0, "`@0` is meaningless and disallowed for efficiency"
    Rnk = np.argsort(Dist, axis=-1)
    # AP = 0
    AP = np.zeros([len(k)], dtype=np.float32)
    pos = np.arange(m)  # 0-base
    # t_fi_1[k]: t_{f(k) - 1}
    t_fi_1 = np.zeros([m])
    # r_fi[k]: #relevant samples in the tie where k lies in
    r_fi = np.zeros([m])
    # n_fi[k]: #samples in the tie where k lies in
    n_fi = np.zeros([m])
    # R_fi_1[k]: prefix sum of r_fi (exclude r_fi[k])
    R_fi_1 = np.zeros([m])
    for d, s, rnk in zip(Dist, Sim, Rnk):
        # Rm = s.sum()  # #rel in all
        s_sort = s[rnk]
        # Rm = s_sort[:k].sum()  # #rel in top-k
        # if 0 == Rm:
        #     continue
        Rm_list = s_sort.cumsum()
        if 0 == Rm_list[-1]:  # = Rm.max() = s_sort.sum()
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
            n_fi[tie_idx] = tie_idx.sum()
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
        # AP += kernel[:k].sum() / Rm
        kernel_cumsum = np.cumsum(kernel)
        for kid, _k in enumerate(k):
            if Rm_list[_k - 1]:
                # `_k - 1` to shift to 0-base
                AP[kid] += kernel_cumsum[_k - 1] / Rm_list[_k - 1]

    _mAP = AP / n
    if 1 == _mAP.shape[0]:
        _mAP = _mAP[0]
    return _mAP


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

    D = (qB.shape[1] - qB.dot(rB.T)) / 2
    S = (query_L.dot(retrieval_L.T) > 0).astype(np.int32)
    print("mAP test:", mAP(D, S, k=-1))
    # print("NDCG test:", NDCG(qB, rB, query_L, retrieval_L, what=1))

    # test tie-aware P@k, R@k
    # print("tie mAP:", mAP_tie(qB, rB, query_L, retrieval_L, k=-1, sparse=False))
    # print("tie NDCG:", NDCG_tie(qB, rB, query_L, retrieval_L, k=-1, sparse=False))
    # print("change place")
    # print("tie mAP:", mAP_tie(rB, qB, retrieval_L, query_L, k=-1, sparse=False))
    # print("tie NDCG:", NDCG_tie(rB, qB, retrieval_L, query_L, k=-1, sparse=False))
