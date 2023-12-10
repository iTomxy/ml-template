import collections
import torch
import torch.nn.functional as F


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


def freeze_module(*modules):
    for m in modules:
        for param in m.parameters():
            param.requires_grad = False


def activate_module(*modules):
    for m in modules:
        for param in m.parameters():
            param.requires_grad = True


def zero_grad(*modules):
    for m in modules:
        for param in m.parameters():
            param.grad = None


def seed_everything(seed=42):
    """pytorch version seed everything
    Ref:
        - https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
        - https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964
        - https://pytorch.org/docs/stable/notes/randomness.html
        - https://discuss.pytorch.org/t/what-is-the-max-seed-you-can-set-up/145688
    """
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
