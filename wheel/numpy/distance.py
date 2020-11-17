import numpy as np
from sklearn.preprocessing import normalize


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
    return (bit - A.dot(B.T)) * 0.5


def euclidean(A, B=None, sqrt=False):
    if (B is None) or (B is A):
        aTb = A.dot(A.T)
        aTa = bTb = np.diag(aTb)
    else:
        aTb = A.dot(B.T)
        aTa = (A * A).sum(1)
        bTb = (B * B).sum(1)
    D = aTa[:, np.newaxis] - 2.0 * aTb + bTb[np.newaxis, :]
    if sqrt:
        D = np.sqrt(D)
    return D
