import torch


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
    x_flat = X.flatten()
    _bin = torch.bincount(x_flat)
    mask = torch.gather(_bin, 0, x_flat).view(*X.size())
    return mask.to(X.dtype)
