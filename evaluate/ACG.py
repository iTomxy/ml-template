import numpy as np


def ACG(Dist, Rel, k=-1):
    """Average Cumulative Gains"""
    n, m = Dist.shape
    if (k < 0) or (k > m):
        k = m
    Gain = Rel
    Rank = np.argsort(Dist)

    _ACG = 0
    for g, rnk in zip(Gain, Rank):
        _ACG += g[rnk[:k]].mean()
    return _ACG / n
