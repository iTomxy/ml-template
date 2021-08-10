import numpy as np


def top_k_mask(X, k=1, reverse=False):
    """0/1 mask of the top-k elements along each row
    Input:
        X: [n, m], some values
        k: `k` in top-k
        reverse: if `True`, pick the k smallest elements instead
    Output:
        M: [n, m], in {0, 1}, M[i][j] == 1 iff
            X[i][j] is one of the k biggest/smallest elements in i-th row
    """
    assert (k > 0) and (k <= X.shape[1]), "invalid k: {}".format(k)
    asc_idx = np.argsort(X)  # ascending
    if reverse:
        asc_idx = asc_idx[:, :k]  # smallest
    else:
        asc_idx = asc_idx[:, -k:]  # biggest
    mask = np.zeros_like(X, dtype="int")
    for i in range(X.shape[0]):
        mask[i][asc_idx[i]] = 1
    return mask
