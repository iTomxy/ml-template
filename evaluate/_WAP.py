import numpy as np


def WAP(Dist, Rel, k=-1):
    """Weighted mean Average Precision"""
    n, m = Dist.shape
    if (k < 0) or (k > m):
        k = m
    Gain = Rel
    S = (Gain > 0).astype(np.int)
    pos = np.arange(k) + 1
    Rank = np.argsort(Dist)

    _WAP = 0.0
    for s, g, rnk in zip(S, Gain, Rank):
        _rnk = rnk[:k]
        s, g = s[_rnk], g[_rnk]
        n_rel = s.sum()
        if n_rel > 0:
            acg = np.cumsum(g) / pos
            _WAP += (acg * s).sum() / n_rel

    return _WAP / n
