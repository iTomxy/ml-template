import numpy as np


def mAHP(Dist, Rel, k=-1):
    """mean Average Hierarchical Precision
    APH@k = 1/k * [ sum_i {HP@i} - 1/2 * (HP@1 + HP@k) ]
    ref:
    - https://github.com/cvjena/semantic-embeddings/issues/4
    - Hierarchical Semantic Indexing for Large Scale Image Retrieval
    """
    n, m = Dist.shape
    if (k < 0) or (k > m):
        k = m
    Rank = np.argsort(Dist)

    AHP = 0
    for rel, rnk in zip(Rel, Rank):
        sim_sum_best = np.cumsum(np.sort(rel)[::-1][:k])
        if sim_sum_best[0] > 0:
            sim_sum_real = np.cumsum(rel[rnk[:k]])
            hp = sim_sum_real / sim_sum_best
            AHP += hp.mean() - 0.5 * (hp[0] + hp[-1]) / k
    return AHP / n


def HP_tie(Dist, Rel, k=-1):
    """get tie-aware HP list for mAHP_tie
    return:
    - M: [n, k], with M[i][j] = t-HP@j(q_i, V)
    """
    n, m = Dist.shape
    if (k < 0) or (k > m):
        k = m
    Rank = np.argsort(Dist)
    pos = 1 + np.arange(m)  # 1-base

    t_hp_list = []
    for dis, rnk, rel in zip(Dist, Rank, Rel):
        sim_sum_best = np.cumsum(np.sort(rel)[::-1])  # [m]
        d_sort = dis[rnk]  # [m]
        rel_sort = rel[rnk]
        d_unique = np.unique(dis)  # ascending
        # sim_sum_pre[i]: sim sum of the previous ties
        #       before the tie where i-th elem lies
        sim_sum_pre = np.zeros_like(dis)
        _pre_sum = 0
        # sim_sum_tie[i]: sim sum of the tie where i-th elem lies
        sim_sum_tie = np.zeros_like(dis)
        # tie_n[i]: #samples in the tie where i-th elem lies
        tie_n = np.zeros_like(dis)
        # t_{c-1}[i]: position before the 1st elem of the tie
        #    where the i-th sample lies
        tc_1 = np.zeros_like(dis)
        for _d in d_unique:
            mask_tie = np.equal(d_sort, _d)
            tc_1[mask_tie] = pos[mask_tie][0] - 1
            tie_n[mask_tie] = mask_tie.astype(np.int).sum()
            _ss_tie = rel_sort[mask_tie].sum()
            sim_sum_tie[mask_tie] = _ss_tie
            sim_sum_pre[mask_tie] = _pre_sum
            _pre_sum += _ss_tie
        t_hp = (sim_sum_pre + (pos - tc_1) * (sim_sum_tie / tie_n)) / sim_sum_best
        t_hp_list.append(t_hp[:k])
    return np.vstack(t_hp_list)


def mAHP_tie(Dist, Rel, k=-1):
    """mean Average Hierarchical Precision
    ref: https://blog.csdn.net/HackerTom/article/details/107458334
    """
    t_HP = HP_tie(Dist, Rel, k)
    n, m = Dist.shape
    if (k < 0) or (k > m):
        k = m
    assert k == t_HP.shape[1]
    AHP = t_HP.mean(1) - 0.5 * (t_HP[:, 0] + t_HP[:, -1]) / k
    return AHP.mean()
