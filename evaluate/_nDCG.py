import numpy as np


def nDCG(Dist, Rel, k=-1):
    """Normalized Discounted Cumulative Gain
    Input:
        Dist: [n, m], Hamming distance matrix
        Rel: [n, m], relevance mattrix, in {0, 1, 2, ...}
        k: nDCG@k, int or int tuple/list
           default `-1` means nDCG@ALL
    ref:
    - https://blog.csdn.net/HackerTom/article/details/108413141
    - https://github.com/kunhe/TALR/blob/master/%2Beval/NDCG.m
    """
    if isinstance(k, int):
        k = [k]
    else:
        k = sorted(k)  # ascending
    n, m = Dist.shape
    for kid in range(len(k)):
        if (k[kid] < 0) or (k[kid] > m):
            k[kid] = m
    G = 2 ** Rel - 1
    # D = np.log2(2 + np.arange(k))
    D = np.log2(2 + np.arange(m))
    Rank = np.argsort(Dist)

    _nDCG = np.zeros([len(k)], dtype=np.float32)
    for g, d, rnk in zip(G, D, Rank):
        # dcg_best = (np.sort(g)[::-1][:k] / D).sum()
        g_desc = np.sort(g)[::-1]
        dcg_best_list = (g_desc / D).cumsum()
        if 0 == dcg_best_list[0]:  # = dcg_best_list.min() = biggist DCG
            continue
        # if dcg_best > 0:
        #     dcg = (g[rnk[:k]] / D).sum()
        #     _NDCG += dcg / dcg_best
        g_sort = g[rnk]
        dcg_list = (g_sort / D).cumsum()
        for kid, _k in enumerate(k):
            dcg = dcg_list[_k - 1]
            _nDCG[kid] += dcg / dcg_best_list[_k - 1]

    _nDCG /= n
    if 1 == _nDCG.shape[0]:
        _nDCG = _nDCG[0]
    return _nDCG


def nDCG_tie(Dist, Rel, k=-1):
    """tie-aware NDCG
    Dist: [n, m], Hamming distance matrix
    ref:
    - https://blog.csdn.net/HackerTom/article/details/107458334
    - https://github.com/kunhe/TALR/blob/master/%2Beval/tieNDCG.m
    """
    n, m = Dist.shape
    if (k < 0) or (k > m):
        k = m
    G = 2 ** Rel - 1
    pos = np.arange(m) + 1  # 1-base
    D_inv = 1 / np.log2(1 + pos)
    Rank = np.argsort(Dist)

    _NDCG = 0
    n_c = np.zeros([m])
    sum_d = np.zeros([m])
    for g, d, rnk in zip(G, Dist, Rank):
        dcg_best = (np.sort(g)[::-1] * D_inv)[:k].sum()
        if 0 == dcg_best:
            continue
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
            sum_d[tie_idx] = (1 / np.log2(1 + np.arange(tc_1 + 1, min(k, tc) + 1))).sum()

        dcg = ((g_sort[:k] / n_c[:k]) * sum_d[:k]).sum()
        _NDCG += dcg / dcg_best

    return _NDCG / n
