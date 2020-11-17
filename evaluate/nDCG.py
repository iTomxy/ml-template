import numpy as np


def NDCG(Dist, Rel, k=-1):
    """Normalized Discounted Cumulative Gain
    ref: https://github.com/kunhe/TALR/blob/master/%2Beval/NDCG.m
    """
    n, m = Dist.shape
    if (k < 0) or (k > m):
        k = m
    G = 2 ** Rel - 1
    D = np.log2(2 + np.arange(k))
    Rank = np.argsort(Dist)

    _NDCG = 0
    for g, rnk in zip(G, Rank):
        dcg_best = (np.sort(g)[::-1][:k] / D).sum()
        if dcg_best > 0:
            dcg = (g[rnk[:k]] / D).sum()
            _NDCG += dcg / dcg_best
    return _NDCG / n


def NDCG_tie(Dist, Rel, k=-1):
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
