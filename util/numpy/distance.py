import math
import numpy as np
from sklearn.preprocessing import normalize


def calc_dist_in_batch(dist_fn, A, B=None, threshold=5000):
    if B is None:
        B = A
    if (A.shape[0] < threshold) and (B.shape[0] < threshold):
        return dist_fn(A, B)
    to_transpose = False
    if A.shape[0] < B.shape[0]:
        A, B = B, A
        to_transpose = True
    to_seq = (B.shape[0] > threshold)
    res = np.zeros([A.shape[0], B.shape[0]]).astype(A.dtype)
    for i in range(0, A.shape[0], threshold):
        a = A[i: i + threshold]
        if to_seq:
            for k in range(0, B.shape[0], threshold):
                b = B[k: k + threshold]
                res[i:i+threshold, k:k+threshold] = dist_fn(a, b)
        else:
            res[i:i+threshold] = dist_fn(a, B)
    if to_transpose:
        res = res.T
    return res


def cos(A, B=None):
    """cosine"""
    An = normalize(A, norm='l2', axis=1)
    if (B is None) or (B is A):
        return np.dot(An, An.T)
    Bn = normalize(B, norm='l2', axis=1)
    S = np.dot(An, Bn.T)
    return np.clip(S, -1, 1)


def hamming(A, B=None):
    """A, B: [None, bit]
    elements in {-1, 1}
    """
    if B is None: B = A
    bit = A.shape[1]
    D = (bit - A.dot(B.T)) * 0.5
    return np.clip(D, 0, bit)


def hamming_real(X, Y=None):
    """continous extension / relaxed version Hamming distance
    X, Y: [None, K], in [-1, 1] (e.g. after tanh)
    """
    if Y is None:
        Y = X
    K = X.shape[1]
    _cos = cos(X, Y)
    D = 0.5 * K * (1 - _cos)
    return np.clip(D, 0, K)


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
    return np.clip(D, 0, math.inf)


if __name__ == "__main__":
    a = np.sign(np.random.randn(5, 2)).astype(np.float32)
    b = np.sign(np.random.randn(7, 2)).astype(np.float32)

    # euclidean
    euc = euclidean(a, b)
    euc_wrap = calc_dist_in_batch(euclidean, a, b, threshold=3)
    # print("euc:\n", euc, "\neuc_wrap:", euc_wrap)
    print(euc.shape, euc_wrap.shape)
    print((euc != euc_wrap).astype(np.int).sum())

    # hamming
    ham = hamming(a, b)
    ham_wrap = calc_dist_in_batch(hamming, a, b, threshold=3)
    # print("ham:\n", ham, "\nham_wrap:", ham_wrap)
    print((ham != ham_wrap).astype(np.int).sum())

    # cos
    cos_d = cos(a, b)
    cos_wrap = calc_dist_in_batch(cos, a, b, threshold=3)
    # print("cos:\n", cos_d, "\ncos_wrap:", cos_wrap)
    print((cos_d != cos_wrap).astype(np.int).sum())
