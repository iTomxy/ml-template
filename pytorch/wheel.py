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
    D[D < 0] = 0.0

    if sqrt:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = D.eq(0).float()
        D = D + mask * 1e-16
        D = D.sqrt()
        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        D = D * (1.0 - mask)

    return D


def sim_mat(label, label2=None, sparse=False):
    """similarity matrix
    S[i][j] = 1 <=> i- & j-th share at lease 1 label
    S[i][j] = 0 otherwise
    """
    if label2 is None:
        label2 = label
    if sparse:
        S = label.view(-1, 1).eq(label2.view(1, -1))
    else:
        S = label.mm(label2.T) > 0
    return S.float()


def cos(X, Y=None):
    """cosine of every (Xi, Yj) pair
    X, Y: (n, dim)
    """
    X_n = F.normalize(X, p=2, dim=1)
    if (Y is None) or (X is Y):
        return X_n.mm(X_n.T)
    Y_n = F.normalize(Y, p=2, dim=1)
    return X_n.mm(Y_n.T)


def hamming(X, Y=None):
    if Y is None:
        Y = X
    K = X.size(1)
    return 0.5 * (K -X.mm(Y.T))
