import collections
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
    return X_n.mm(Y_n.T).clamp(-1, 1)


def hamming(X, Y=None):
    if Y is None:
        Y = X
    K = X.size(1)
    D = (K - X.mm(Y.T)) / 2
    return D.clamp(0, K)


def check_nan_inf(tensors):
    """tensors: single tensor, or collections of tensor"""
    if isinstance(tensors, torch.Tensor):
        _nan = 1 if torch.isnan(tensors).any() else 0
        _inf = 2 if torch.isinf(tensors).any() else 0
        if _nan or _inf:
            return _nan + _inf
    elif isinstance(tensors, collections.Iterable):
        for _t in tensors:
            res = check_nan_inf(_t)
            if res:
                return res
    else:
        raise Exception("check_nan_inf: unsupported type: {}".format(type(tensors)))
    return 0


def one_hot(label, n_class):
    """convert labels from sparse to one-hot
    Input:
    - label: [n], sparse class ID of n samples
    - n_class: scalar, #classes
    Output:
    - L: [n, c], label in one-hot
    Ref:
    - https://pytorch.org/docs/stable/tensors.html#torch.Tensor.scatter_
    """
    L = torch.zeros(label.size(0), n_class).to(label.device).scatter_(
        1, label.long().unsqueeze(1), 1)
    return L.to(label.dtype)


def count_mask(X):
    """M(i,j) = |{ (s,t) | X(s,t) = X(i,j) }|
    ref: https://blog.csdn.net/HackerTom/article/details/108902880
    """
    x_flat = X.flatten().int()
    _bin = torch.bincount(x_flat)
    mask = torch.gather(_bin, 0, x_flat).view(*X.size())
    return mask.to(X.dtype)


def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False


def freeze_mulit_layers(multi_layers):
    for layer in multi_layers:
        freeze_layer(layer)


def get_on(*tensors, cuda=True):
    """np.ndarray -> torch.Tensor (cuda)"""
    if cuda:
        fn = lambda x: torch.from_numpy(x).cuda()
    else:
        fn = lambda x: torch.from_numpy(x)
    res = list(map(fn, tensors))
    if len(res) == 1:
        res = res[0]
    return res


def get_off(*tensors, cuda=True):
    """torch.Tensor (cuda) -> np.ndarray"""
    res = []
    for t in tensors:
        t = t.data
        if cuda:
            t = t.cpu()
        if 0 == t.ndim:  # scalar
            t = t.item()
        else:
            t = t.numpy()
        res.append(t)
    if len(res) == 1:
        res = res[0]
    return res
