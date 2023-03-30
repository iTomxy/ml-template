import torch


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
    return S.to(label.dtype)
