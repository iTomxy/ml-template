import torch
import torch.nn.functional as F


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
    D = D.where(D > 0, torch.zeros_like(D).to(D.device))

    if sqrt:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = D.eq(0).float()
        D = D + mask * 1e-16
        D = D.sqrt()
        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        D = D * (1.0 - mask)

    return D


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
