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
    n, m = Dist.shape
    for kid in range(len(k)):
        if (k[kid] < 0) or (k[kid] > m):
            k[kid] = m
    k = sorted(k)  # ascending
    G = 2 ** Rel - 1
    # D = np.log2(2 + np.arange(k))
    D = np.log2(2 + np.arange(m))
    Rank = np.argsort(Dist)

    _nDCG = np.zeros([len(k)], dtype=np.float32)
    for g, d, rnk in zip(G, D, Rank):
        # dcg_best = (np.sort(g)[::-1][:k] / D).sum()
        g_desc = np.sort(g)[::-1]
        if 0 == g_desc[0]:  # biggist DCG
            continue
        dcg_best_list = (g_desc / D).cumsum()
        # if dcg_best > 0:
        #     dcg = (g[rnk[:k]] / D).sum()
        #     _NDCG += dcg / dcg_best
        g_sort = g[rnk]
        dcg_list = (g_sort / D).cumsum()
        for kid, _k in enumerate(k):
            if _k > 0:
                dcg = dcg_list[_k - 1]
                _nDCG[kid] += dcg / dcg_best_list[_k - 1]

    _nDCG /= n
    if 1 == _nDCG.shape[0]:
        _nDCG = _nDCG[0]
    return _nDCG


def nDCG_tie(Dist, Rel, k=-1):
    """tie-aware NDCG
    Input:
        Dist: [n, m], Hamming distance matrix
        Rel: [n, m], relevance mattrix, in {0, 1, 2, ...}
        k: nDCG@k, int or int tuple/list
           default `-1` means nDCG@ALL
    ref:
    - https://blog.csdn.net/HackerTom/article/details/107458334
    - https://github.com/kunhe/TALR/blob/master/%2Beval/tieNDCG.m
    """
    if isinstance(k, int):
        k = [k]
    n, m = Dist.shape
    for kid in range(len(k)):
        if (k[kid] < 0) or (k[kid] > m):
            k[kid] = m
    k = sorted(k)  # ascending
    G = 2 ** Rel - 1
    pos = np.arange(m) + 1  # 1-base
    D_inv = 1 / np.log2(1 + pos)
    Rank = np.argsort(Dist, axis=-1)

    _nDCG = np.zeros([len(k)], dtype=np.float32)
    for g, dist, rnk in zip(G, Dist, Rank):
        g_desc = np.sort(g)[::-1]
        if 0 == g_desc[0]:  # biggist DCG
            continue
        dcg_best_list = (g_desc * D_inv).cumsum()
        dist_unique = np.unique(dist)  # ascending
        dist_sort = dist[rnk]
        g_sort = g[rnk]

        _start_kid = 0
        dcg_list = np.zeros([len(k)], dtype=np.float32)
        for _d in dist_unique:
            tie_idx = (dist_sort == _d)
            tie_pos = pos[tie_idx]
            tc_1 = tie_pos[0] - 1  # i.e. tie_pos.min() - 1
            tc = tie_pos[-1]
            while _start_kid < len(k):
                if k[_start_kid] > tc_1:
                    break
                else:
                    _start_kid += 1
            if _start_kid >= len(k):
                break
            n_c = tie_idx.sum()
            tie_avg_gain = g_sort[tie_idx].sum() / n_c  # != g_sort[tie_idx].mean()
            tie_sum_d_list = (1 / np.log2(1 + np.arange(tc_1 + 1, tc + 1))).cumsum()
            for kid_offset, _k in enumerate(k[_start_kid:]):
                tie_sum_d = tie_sum_d_list[min(_k, tc) - tc_1 - 1]
                tie_dcg = tie_avg_gain * tie_sum_d
                dcg_list[kid_offset + _start_kid] += tie_dcg

        for kid, _k in enumerate(k):
            if _k > 0:
                _ndcg = dcg_list[kid] / dcg_best_list[_k - 1]  # 0-base
                _nDCG[kid] += _ndcg

    _nDCG /= n
    if 1 == _nDCG.shape[0]:
        _nDCG = _nDCG[0]
    return _nDCG
