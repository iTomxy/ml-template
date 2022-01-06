import torch
import torch.nn.functional as F


def calc_dist_in_batch(dist_fn, A, B=None, threshold=5000):
    if B is None:
        B = A
    if (A.size(0) < threshold) and (B.size(0) < threshold):
        return dist_fn(A, B)
    to_transpose = False
    if A.size(0) < B.size(0):
        A, B = B, A
        to_transpose = True
    to_seq = (B.size(0) > threshold)
    res = torch.zeros([A.size(0), B.size(0)]).to(A.dtype).to(A.device)
    for i in range(0, A.size(0), threshold):
        a = A[i: i + threshold]
        if to_seq:
            for k in range(0, B.size(0), threshold):
                b = B[k: k + threshold]
                res[i:i+threshold, k:k+threshold] = dist_fn(a, b)
        else:
            res[i:i+threshold] = dist_fn(a, B)
    if to_transpose:
        res = res.T
    return res


def euclidean(A, B=None, sqrt=False):
    if (B is None) or (B is A):
        aTb = A.mm(A.T)
        aTa = bTb = aTb.diag()
    else:
        aTb = A.mm(B.T)
        aTa = A.mm(A.T).diag()
        bTb = B.mm(B.T).diag()
    D = aTa.view(-1, 1) - 2.0 * aTb + bTb.view(1, -1)
    # D[D < 0] = 0.0
    D = D.clamp(min=0)

    if sqrt:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = D.eq(0).float()
        D = D + mask * 1e-16
        D = D.sqrt()
        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        D = D * (1.0 - mask)

    return D.clamp(min=0)


def cos(X, Y=None):
    """cosine of every (Xi, Yj) pair
    X, Y: (n, dim)
    """
    X_n = F.normalize(X, p=2, dim=1)
    if (Y is None) or (X is Y):
        return X_n.mm(X_n.T)
    Y_n = F.normalize(Y, p=2, dim=1)
    return X_n.mm(Y_n.T).clamp(-1, 1)


def hamming(X, Y=None):
    if Y is None:
        Y = X
    K = X.size(1)
    D = (K - X.mm(Y.T)) / 2
    return D.clamp(0, K)


def hamming_real(X, Y=None):
    """continous extension / relaxed version Hamming distance
    X, Y: [None, K], in [-1, 1] (e.g. after tanh)
    """
    if Y is None:
        Y = X
    K = X.size(1)
    _cos = cos(X, Y)
    D = 0.5 * K * (1 - _cos)
    return D.clamp(0, K)


if __name__ == "__main__":
    a = torch.randn(5, 2).sign().float()
    b = torch.randn(7, 2).sign().float()

    # euclidean
    euc = euclidean(a, b)
    euc_wrap = calc_dist_in_batch(euclidean, a, b, threshold=3)
    # print("euc:\n", euc, "\neuc_wrap:", euc_wrap)
    print(euc.size(), euc_wrap.size())
    print((euc != euc_wrap).int().sum())

    # hamming
    ham = hamming(a, b)
    ham_wrap = calc_dist_in_batch(hamming, a, b, threshold=3)
    # print("ham:\n", ham, "\nham_wrap:", ham_wrap)
    print((ham != ham_wrap).int().sum())

    # cos
    cos_d = cos(a, b)
    cos_wrap = calc_dist_in_batch(cos, a, b, threshold=3)
    # print("cos:\n", cos_d, "\ncos_wrap:", cos_wrap)
    print((cos_d != cos_wrap).int().sum())
